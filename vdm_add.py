import torch
import os
import shutil
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel, DPMSolverMultistepScheduler, AutoencoderKL
from torchvision import utils
from tqdm import tqdm
from PIL import Image
import time
from train_ldm import LabelEmbedding
# ================= 1. 配置部分 =================
class AugConfig:
    # --- 路径配置 ---
    # 原始数据集根目录 (包含 1, 2, 3, 4 文件夹)
    original_data_dir = "./Base_datasets/train" 
    # 扩充后的目标目录 (建议新建一个，避免直接污染原始输入)
    output_dir = "./ldm_augmented_v2/train"
    
    # 训练好的模型路径
    model_path = "ldm_udder_v22/checkpoint_epoch_99" 
    vae_path = r"D:\hf_cache\hub\models--stabilityai--sd-vae-ft-mse\snapshots\31f26fdeee1355a5c34592e401dd41e45d25a493"
    
    # --- 扩充目标 ---
    # 每一类最终需要的总张数 (原始图片 + 生成图片)
    target_count = 5000  
    
    # --- 采样参数 ---
    num_inference_steps = 20        
    guidance_scale = 4.0            
    image_size = 512
    batch_size = 8              
    
    # --- LDM 核心参数 (必须与 train_ldm.py 一致) ---
    latent_channels = 4
    latent_size = 64
    scale_factor = 0.18215
    num_classes = 4
    uncond_label = 4                
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

config = AugConfig()

# ================= 2. 加载模型组件 =================
def load_models():
    print(f"🚀 正在从 {config.model_path} 加载 LDM 组件...")
    
    # 1. VAE
    vae = AutoencoderKL.from_pretrained(config.vae_path).to(config.device)
    vae.eval()

    # 2. UNet
    unet = UNet2DConditionModel.from_pretrained(config.model_path).to(config.device)
    unet.eval()

    # 3. Label Projection
    label_proj = LabelEmbedding(config.num_classes+1).to(config.device)
    proj_path = os.path.join(config.model_path, "label_proj.pt")
    if os.path.exists(proj_path):
        label_proj.load_state_dict(torch.load(proj_path, map_location=config.device))
        print("✅ Label Projection 层加载成功")
    else:
        raise FileNotFoundError(f"❌ 未在路径下找到 label_proj.pt，请检查模型目录")

    # 4. 采样器
    scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.02,
        beta_schedule="scaled_linear",
        algorithm_type="dpmsolver++",
        solver_order=2,
    )

    return vae, unet, label_proj, scheduler

# ================= 3. 执行扩充逻辑 =================
def run_augmentation():
    vae, unet, label_proj, scheduler = load_models()
    
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
        
        print(f"\n📂 类别 {cat} | 原始+现有: {current_count} | 目标: {config.target_count}")
        
        if needed <= 0:
            print(f"✅ 类别 {cat} 已达标，跳过。")
            continue

        # --- 第三步：生成补齐图片 ---
        print(f"🔥 开始生成剩余的 {needed} 张图片...")
        pbar = tqdm(total=needed, desc=f"Generating {cat}")
        generated_counter = 0
        
        while generated_counter < needed:
            curr_batch_size = min(config.batch_size, needed - generated_counter)
            
            # 采样逻辑 (与 train_ldm.py 一致)
            latents = torch.randn(
                (curr_batch_size, config.latent_channels, config.latent_size, config.latent_size),
                device=config.device
            )
            scheduler.set_timesteps(config.num_inference_steps)
            
            with torch.no_grad():
                # 准备 CFG 嵌入
                labels = torch.full((curr_batch_size,), idx, dtype=torch.long, device=config.device)
                cond_emb = label_proj(labels)
                
                uncond_labels = torch.full((curr_batch_size,), config.uncond_label, dtype=torch.long, device=config.device)
                uncond_emb = label_proj(uncond_labels)

                # 扩散采样循环
                for t in scheduler.timesteps:
                    latent_model_input = torch.cat([latents] * 2)
                    t_input = torch.cat([t.unsqueeze(0).to(config.device)] * 2 * curr_batch_size)
                    emb_input = torch.cat([uncond_emb, cond_emb])
                    labels_input = torch.cat([uncond_labels, labels])

                    noise_pred = unet(latent_model_input, t_input, encoder_hidden_states=emb_input, class_labels=labels_input).sample
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + config.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    latents = scheduler.step(noise_pred, t, latents).prev_sample

                # VAE 解码
                images = vae.decode(latents / config.scale_factor).sample
                images = (images / 2 + 0.5).clamp(0, 1)
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

    print(f"\n🎉 扩充任务完成！最终数据集位于: {config.output_dir}")

if __name__ == "__main__":
    run_augmentation()