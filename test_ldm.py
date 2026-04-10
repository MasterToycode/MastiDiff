import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel, AutoencoderKL, DPMSolverMultistepScheduler
from torchvision import utils
import numpy as np
# 从 train_ldm 导入 LabelEmbedding 类
from train_ldm import LabelEmbedding

# 告诉程序你的自定义缓存库在哪里
os.environ["HF_HOME"] = r"D:\hf_cache" 
# 开启离线模式（1代表开启，0代表关闭），这不是文件夹名
os.environ["HF_HUB_OFFLINE"] = "1"
# === 1. 配置参数 ===
# 请指向你最新的模型路径
checkpoint_dir = "ldm_udder_v22/checkpoint_epoch_99" 
output_path = "cfg_test_comparison.png"

num_classes = 4
uncond_label = 4
latent_size = 64
latent_channels = 4
cross_attention_dim = 512
vae_model = r"D:\hf_cache\hub\models--stabilityai--sd-vae-ft-mse\snapshots\31f26fdeee1355a5c34592e401dd41e45d25a493"
scale_factor = 0.18215

# 要测试的 CFG 强度列表
cfg_scales = [1.0, 2.0, 3.0,4.0, 5.0,6.0, 8.0] 
# 采样步数 (DPM-Solver 下 30-50 步通常足够)
num_inference_steps = 20

device = "cuda" if torch.cuda.is_available() else "cpu"

# === 2. 加载模型 ===
print("正在加载模型...")
vae = AutoencoderKL.from_pretrained(vae_model).to(device).eval()
unet = UNet2DConditionModel.from_pretrained(checkpoint_dir).to(device).eval()

# 加载 label_proj (使用新的 LabelEmbedding 类)
label_proj = LabelEmbedding(num_classes + 1).to(device).eval()
label_proj_path = os.path.join(checkpoint_dir, "label_proj.pt")
if os.path.exists(label_proj_path):
    label_proj.load_state_dict(torch.load(label_proj_path, map_location=device))
    print("✅ 已成功加载 label_proj 权重")
else:
    print("❌ 错误：未找到 label_proj.pt 文件！")
    exit()

# 配置调度器
scheduler = DPMSolverMultistepScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.02,
    beta_schedule="scaled_linear",
    algorithm_type="dpmsolver++"
)

# === 3. 采样核心函数 ===
def sample_with_cfg(cfg_scale):
    scheduler.set_timesteps(num_inference_steps)
    
    # 每次采样 4 个类别的图像 (0, 1, 2, 3)
    labels = torch.arange(num_classes).to(device)
    # 使用固定的种子以便对比
    torch.manual_seed(100)
    latents = torch.randn(num_classes, latent_channels, latent_size, latent_size).to(device)
    
    for t in scheduler.timesteps:
        # 准备条件嵌入
        with torch.no_grad():
            # 1. 条件部分 (Classes 0-3)
            # LabelEmbedding 直接接收整数标签，返回形状为 (-1, 8, 512) 的 tokens
            cond_emb = label_proj(labels)
            
            # 2. 无条件部分 (Class 4)
            uncond_labels = torch.full_like(labels, uncond_label)
            uncond_emb = label_proj(uncond_labels)
            
            # 合并输入进行并行推理
            latent_model_input = torch.cat([latents] * 2)
            t_input = torch.cat([t.unsqueeze(0)] * (num_classes * 2)).to(device)
            emb_input = torch.cat([cond_emb, uncond_emb])
            class_input = torch.cat([labels, uncond_labels])
            
            # 预测噪声
            noise_pred_full = unet(latent_model_input, t_input, encoder_hidden_states=emb_input, class_labels=class_input).sample
            
            # 分离条件与无条件结果
            noise_pred_cond, noise_pred_uncond = noise_pred_full.chunk(2)
            
            # CFG 引导公式
            noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
            
            # 步进
            latents = scheduler.step(noise_pred, t, latents).prev_sample
            
    # 解码图像
    with torch.no_grad():
        imgs = vae.decode(latents / scale_factor).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
    return imgs

# === 4. 执行对比循环 ===
print(f"开始对比测试，CFG 范围: {cfg_scales}")
all_results = []

for cfg in cfg_scales:
    print(f"正在使用 CFG = {cfg} 进行采样...")
    imgs = sample_with_cfg(cfg)
    all_results.append(imgs)

# 合并所有图像形成大网格
# 每行代表一个 CFG 强度，每列是一个类别
comparison_grid = torch.cat(all_results, dim=0)
utils.save_image(comparison_grid, output_path, nrow=num_classes)
print(f"🚀 对比图已保存至: {output_path}")