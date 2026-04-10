import torch
import os
import shutil
from diffusers import UNet2DModel, DDIMScheduler
from torchvision import utils
from tqdm import tqdm
from PIL import Image

# ================= 1. 配置部分 =================
class AugConfig:
    # 模型路径：指向包含 .safetensors 的 EMA 目录
    model_path = "ddpm-udder-results6/ema_model_epoch_119" 
    
    # 路径配置
    original_train_dir = "./Base_datasets_augmented_3/train"      # 原始切分后的训练集
    aug_train_dir = "./Base_datasets_augmented_4/train" # 扩充后的保存路径
    
    # 补齐目标
    target_count = 5000  
    
    # 采样配置（同步你提供的 DDIM 配置）
    num_inference_steps = 200       
    image_size = 256
    batch_size = 16                 # 批量生成以提高效率，显存小可调低
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 训练参数匹配
    beta_start = 0.0001
    beta_end = 0.02               
    beta_schedule = "scaled_linear"
    clip_sample = True

config = AugConfig()

# ================= 2. 加载 EMA 模型与采样器 =================
print(f"正在加载 EMA 模型: {config.model_path}")
try:
    model = UNet2DModel.from_pretrained(config.model_path, use_safetensors=True)
    model.to(config.device)
    model.eval()
    print("✅ EMA 模型加载成功！")
except Exception as e:
    print(f"❌ 加载失败: {e}")
    exit()

scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=config.beta_start,
    beta_end=config.beta_end,
    beta_schedule=config.beta_schedule,
    clip_sample=config.clip_sample,
    prediction_type="epsilon"
)
import time
# ================= 3. 执行扩充逻辑 =================
os.makedirs(config.aug_train_dir, exist_ok=True)
# 映射：文件夹名 '1' 对应模型标签 0，依此类推
categories = ['1', '2', '3', '4']

for idx, cat in enumerate(categories):
    source_cat_dir = os.path.join(config.original_train_dir, cat)
    target_cat_dir = os.path.join(config.aug_train_dir, cat)
    os.makedirs(target_cat_dir, exist_ok=True)
    
    # 1. 拷贝原始图片
    source_imgs = [f for f in os.listdir(source_cat_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for img_name in source_imgs:
        shutil.copy(os.path.join(source_cat_dir, img_name), os.path.join(target_cat_dir, img_name))
    
    current_count = len(source_imgs)
    needed = config.target_count - current_count
    
    print(f"\n类别 {cat} (Label {idx}) | 原始: {current_count} | 需补齐: {max(0, needed)}")
    
    if needed <= 0:
        continue

    # 2. 批量生成差额部分
    pbar = tqdm(total=needed, desc=f"Generating Class {cat}")
    generated_so_far = 0
    batch_time = int(time.time())
    while generated_so_far < needed:
        # 计算当前 batch 大小
        current_batch = min(config.batch_size, needed - generated_so_far)
        
        # 准备标签和噪声
        labels = torch.full((current_batch,), idx, dtype=torch.long, device=config.device)
        latents = torch.randn((current_batch, 3, config.image_size, config.image_size), device=config.device)
        
        scheduler.set_timesteps(config.num_inference_steps)
        
        # DDIM 采样循环
        with torch.no_grad():
            for t in scheduler.timesteps:
                model_output = model(latents, t, class_labels=labels).sample
                latents = scheduler.step(model_output, t, latents).prev_sample
        
        # 后处理并保存
        images = (latents / 2 + 0.5).clamp(0, 1)
        images = (images.cpu().permute(0, 2, 3, 1).numpy() * 255).astype("uint8")
        
        for i in range(current_batch):
            img = Image.fromarray(images[i])
            # 文件名：类别_时间戳_序号
            file_name = f"cat{cat}_{batch_time}_{generated_so_far:04d}.png"
            img.save(os.path.join(target_cat_dir, file_name))
            generated_so_far += 1
            pbar.update(1)
            
    pbar.close()

print("\n" + "="*40)
print(f"🎉 数据集扩充平衡任务已完成！")
print(f"存储路径: {config.aug_train_dir}")