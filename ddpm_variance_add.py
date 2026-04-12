import torch
import os
import shutil
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel, DDPMScheduler
from torchvision import utils
from tqdm import tqdm
from PIL import Image
import time
import argparse

# ================= 1. 配置部分 =================
class AugConfig:
    # --- 路径配置 ---
    # 原始数据集根目录 (包含 1, 2, 3, 4 文件夹)
    original_data_dir = "./datasets/train" 
    # 扩充后的目标目录 (建议新建一个，避免直接污染原始输入)
    output_dir = "./ddpm_variance_augmented_v1/train"
    
    # 训练好的模型路径 (DDPM variance模型)
    model_path = "ddpm_variance_22/checkpoint_epoch_9" 
    
    # --- 扩充目标 ---
    # 每一类最终需要的总张数 (原始图片 + 生成图片)
    target_count = 5000  
    
    # --- 采样参数 (参考 test_ddpm_variance.py) ---
    num_inference_steps = 100        
    cfg = 5.0
    variance_scale = 0.00001
    image_size = 256
    batch_size = 8              
    
    # --- DDPM 核心参数 (必须与 train_ddpm_variance.py 一致) ---
    num_classes = 4
    uncond_label = 4                
    cross_attention_dim = 256
    embedding_dim = 256
    num_tokens = 8
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

config = AugConfig()

# ================= 2. LabelEmbedding 类 (必须与训练代码一致) =================
class LabelEmbedding(nn.Module):
    """标签嵌入层，将类别标签转换为cross-attention的encoder_hidden_states"""
    def __init__(self, num_classes, embedding_dim=256, num_tokens=8):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embedding_dim)
        self.dropout = nn.Dropout(0.2)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.SiLU(),
            nn.Linear(embedding_dim * 2, embedding_dim * num_tokens)
        )
        self.pos = nn.Parameter(torch.randn(num_tokens, embedding_dim) * 0.01)
        self.num_tokens = num_tokens
        self.embedding_dim = embedding_dim
    
    def forward(self, labels):
        x = self.embedding(labels)
        # 采样时务必进入 eval 模式以跳过 dropout 和 手动加的随机噪声
        tokens = self.mlp(x).view(-1, self.num_tokens, self.embedding_dim)
        return tokens + self.pos

# ================= 3. 加载模型组件 =================
def load_models():
    print(f"正在从 {config.model_path} 加载 DDPM variance 模型组件...")
    
    # 1. UNet2DConditionModel (6通道输出：3 epsilon + 3 variance)
    model = UNet2DConditionModel.from_pretrained(config.model_path).to(config.device)
    model.eval()

    # 2. Label Projection (必须与训练代码一致)
    label_proj = LabelEmbedding(config.num_classes + 1, 
                                embedding_dim=config.embedding_dim,
                                num_tokens=config.num_tokens).to(config.device)
    proj_path = os.path.join(config.model_path, "label_proj.pt")
    if os.path.exists(proj_path):
        label_proj.load_state_dict(torch.load(proj_path, map_location=config.device))
        print("Label Projection 层加载成功")
    else:
        raise FileNotFoundError(f"未在路径下找到 label_proj.pt，请检查模型目录")

    # 3. 采样器 (DDPMScheduler with learned_range variance)
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
        variance_type="learned_range",  # 关键：告诉 scheduler 我们有方差预测
        clip_sample=True
    )

    return model, label_proj, scheduler

# ================= 4. 执行扩充逻辑 =================
def run_augmentation(dry_run=False):
    model, label_proj, scheduler = load_models()
    
    os.makedirs(config.output_dir, exist_ok=True)
    categories = ['1', '2', '3', '4']

    for idx, cat in enumerate(categories):
        source_cat_dir = os.path.join(config.original_data_dir, cat)
        target_cat_dir = os.path.join(config.output_dir, cat)
        os.makedirs(target_cat_dir, exist_ok=True)
        
        # --- 第一步：确保原始图片已存在于目标目录 ---
        if os.path.exists(source_cat_dir):
            source_imgs = [f for f in os.listdir(source_cat_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for img_name in source_imgs:
                target_path = os.path.join(target_cat_dir, img_name)
                # 如果目标位置不存在该原图，则拷贝
                if not os.path.exists(target_path):
                    shutil.copy(os.path.join(source_cat_dir, img_name), target_path)
        
        # --- 第二步：统计当前总量并计算缺口 ---
        # 统计包含原图和之前已生成的图
        current_imgs = [f for f in os.listdir(target_cat_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        current_count = len(current_imgs)
        needed = config.target_count - current_count
        
        print(f"\n类别 {cat} (标签 {idx}) | 现有: {current_count} | 目标: {config.target_count}")
        
        if needed <= 0:
            print(f"类别 {cat} 已达标，跳过。")
            continue

        if dry_run:
            print(f"Dry-run: 需要生成 {needed} 张图片 (不实际生成)")
            continue

        # --- 第三步：生成补齐图片 ---
        print(f"开始生成剩余的 {needed} 张图片...")
        pbar = tqdm(total=needed, desc=f"Generating {cat}")
        generated_counter = 0
        
        while generated_counter < needed:
            curr_batch_size = min(config.batch_size, needed - generated_counter)
            
            # 采样逻辑 (参考 test_ddpm_variance.py)
            x_t = torch.randn(
                (curr_batch_size, 3, config.image_size, config.image_size),
                device=config.device
            )
            scheduler.set_timesteps(config.num_inference_steps)
            
            with torch.no_grad():
                # 准备 CFG 嵌入
                labels = torch.full((curr_batch_size,), idx, dtype=torch.long, device=config.device)
                cond_emb = label_proj(labels)
                
                uncond_labels = torch.full((curr_batch_size,), config.uncond_label, dtype=torch.long, device=config.device)
                uncond_emb = label_proj(uncond_labels)

                # DDPM 采样循环 (6通道 CFG 处理)
                for t in scheduler.timesteps:
                    # 这里的输入是 curr_batch_size 个样本，我们做 CFG，所以扩展为 2*curr_batch_size
                    latent_model_input = torch.cat([x_t] * 2)
                    prompt_embeds = torch.cat([uncond_emb, cond_emb]) # uncond 在前
                    timesteps = torch.full((2 * curr_batch_size,), t, device=config.device, dtype=torch.long)
                    
                    # 模型一次性跑 2*curr_batch_size 个样本的预测 (输出 6 通道)
                    model_output = model(latent_model_input, timesteps, encoder_hidden_states=prompt_embeds).sample
                    
                    # --- 核心：处理 6 通道 CFG (参考 test_ddpm_variance.py) ---
                    # 拆分噪声部分和方差部分
                    noise_pred_full, var_pred_full = model_output.chunk(2, dim=1)
                    
                    # 对噪声预测做 CFG 引导
                    noise_uncond, noise_cond = noise_pred_full.chunk(2)
                    noise_pred = noise_uncond + config.cfg * (noise_cond - noise_uncond)
                    
                    # 对方差预测做 CFG (方差通常不需要太强的引导，直接取 cond 或者也做微量引导)
                    var_uncond, var_cond = var_pred_full.chunk(2)
                    var_pred = var_cond * config.variance_scale
                    
                    # 重新拼回 6 通道，交给 scheduler 处理
                    final_output = torch.cat([noise_pred, var_pred], dim=1)
                    
                    # 使用 DDPM step 计算。它内部会根据 6 通道自动计算真实方差
                    x_t = scheduler.step(final_output, t, x_t).prev_sample

                # 后处理 (参考 train_ddpm_variance.py)
                images = ((x_t + 1) / 2).clamp(0, 1)
                images = images ** 0.8 
                images_np = (images.cpu().permute(0, 2, 3, 1).numpy() * 255).astype("uint8")
                
                # 保存生成图
                for i in range(curr_batch_size):
                    img = Image.fromarray(images_np[i])
                    # 命名包含 'gen' 前缀，以便区分原图和生成图
                    file_name = f"gen_{cat}_{int(time.time()*1000)}_{i}.png"
                    img.save(os.path.join(target_cat_dir, file_name))
                    generated_counter += 1
                    pbar.update(1)

        pbar.close()

    print(f"\n扩充任务完成！最终数据集位于: {config.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DDPM Variance Dataset Expansion')
    parser.add_argument('--dry-run', action='store_true', help='只计算缺口，不实际生成图片')
    parser.add_argument('--model-path', type=str, help='模型路径 (默认: ddpm_variance_22/checkpoint_epoch_49)')
    parser.add_argument('--output-dir', type=str, help='输出目录 (默认: ./ddpm_augmented_v1/train)')
    parser.add_argument('--target-count', type=int, help='每类目标图片数量 (默认: 5000)')
    parser.add_argument('--batch-size', type=int, help='批量大小 (默认: 8)')
    parser.add_argument('--num-steps', type=int, help='采样步数 (默认: 100)')
    parser.add_argument('--cfg', type=float, help='CFG 引导强度 (默认: 5.0)')
    
    args = parser.parse_args()
    
    # 更新配置
    if args.model_path:
        config.model_path = args.model_path
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.target_count:
        config.target_count = args.target_count
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.num_steps:
        config.num_inference_steps = args.num_steps
    if args.cfg:
        config.cfg = args.cfg
    
    run_augmentation(dry_run=args.dry_run)