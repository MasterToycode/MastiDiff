import os
import torch
import torch.nn as nn
from torchvision import utils
from diffusers import UNet2DConditionModel, DDPMScheduler
from tqdm.auto import tqdm

class Config:
    # Scheduler 细节
    beta_schedule = "squaredcos_cap_v2"
    clip_sample = True
    # 采样配置
    cfg = 6
    sampler = "DDPMScheduler" 
    variance_scale = 0.00001 
    NUM_CLASSES = 4
    CHECKPOINT_PATH = r"ddpm_variance_21\checkpoint_epoch_49" # 替换为你最新的权重文件夹路径
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    image_size=256
    uncond_label = 4
config = Config()

# --- 1. 必须保留的标签投影类 (需与训练代码一致) ---
class LabelEmbedding(nn.Module):
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

# --- 2. 采样配置 ---

@torch.no_grad()
def run_test_sample():
    # 1. 加载模型
    print(f"正在从 {Config.CHECKPOINT_PATH} 加载模型...")
    model = UNet2DConditionModel.from_pretrained(Config.CHECKPOINT_PATH).to(Config.DEVICE)
    model.eval()

    # 2. 加载标签投影层
    label_proj = LabelEmbedding(Config.NUM_CLASSES + 1, embedding_dim=256).to(Config.DEVICE)
    label_proj.load_state_dict(torch.load(os.path.join(Config.CHECKPOINT_PATH, "label_proj.pt"), map_location=Config.DEVICE))
    label_proj.eval()

    # 3. 配置采样器 (DPM-Solver 速度快且稳定)
    # 注意：训练时用 squaredcos_cap_v2，采样器也必须匹配
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
        variance_type="learned_range",
        clip_sample=True
    )
    scheduler.set_timesteps(100) 
    # 备份与准备
    model.eval()
    label_proj.eval()
    labels = torch.tensor([0, 1, 2, 3], device=Config.DEVICE)
    x_t = torch.randn(4, 3, Config.image_size, Config.image_size, device=Config.DEVICE)
    
    # 准备 CFG 嵌入
    uncond_labels = torch.full((4,), Config.uncond_label, device=Config.DEVICE)
    cond_emb = label_proj(labels)
    uncond_emb = label_proj(uncond_labels)

    # 6. 采样循环
    for t in tqdm(scheduler.timesteps, desc="Sampling"):
        with torch.no_grad():
            # 这里的输入是 4 个样本，我们做 CFG，所以扩展为 8
            latent_model_input = torch.cat([x_t] * 2)
            prompt_embeds = torch.cat([uncond_emb, cond_emb]) # 习惯上 uncond 在前
            timesteps = torch.full((8,), t, device=Config.DEVICE, dtype=torch.long)
            
            # 模型一次性跑 8 个样本的预测 (输出 6 通道)
            model_output = model(latent_model_input, timesteps, encoder_hidden_states=prompt_embeds).sample
            
            # --- 核心：处理 6 通道 CFG ---
            # 拆分噪声部分和方差部分
            noise_pred_full, var_pred_full = model_output.chunk(2, dim=1)
            
            # 对噪声预测做 CFG 引导
            noise_uncond, noise_cond = noise_pred_full.chunk(2)
            noise_pred = noise_uncond + config.cfg * (noise_cond - noise_uncond)
            
            # 对方差预测做 CFG (方差通常不需要太强的引导，直接取 cond 或者也做微量引导)
            var_uncond, var_cond = var_pred_full.chunk(2)
            var_pred = var_cond * Config.variance_scale
            
            # 重新拼回 6 通道，交给 scheduler 处理
            final_output = torch.cat([noise_pred, var_pred], dim=1)
            
            # 2. 使用 DDPM step 计算。它内部会根据 6 通道自动计算真实方差
            x_t = scheduler.step(final_output, t, x_t).prev_sample

    # 7. 后处理并保存
    images = ((x_t + 1) / 2).clamp(0, 1)
    #images = images ** 0.9 
    save_name = f"test_cfg_{Config.cfg}.png"
    utils.save_image(images, save_name, nrow=2)
    print(f"测试图已保存至: {save_name}")

if __name__ == "__main__":
    run_test_sample()