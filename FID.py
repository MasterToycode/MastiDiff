import torch
import os
import numpy as np
from diffusers import UNet2DModel, DDIMScheduler
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from tqdm import tqdm

# ================= 1. 配置部分 =================
class EvalConfig:
    # 模型路径（指向你的 EMA 文件夹）
    model_path = "ddpm-udder-results6/ema_model_epoch_119" 
    # 真实数据根目录
    real_data_root = "./datasets/train" # 使用测试集作为基准
    
    image_size = 256
    batch_size = 16
    num_generate_images = 2000  # 生成 2000 张图像进行评估
    device = "cuda" if torch.cuda.is_available() else "cpu"
    beta_start = 0.0001 # 训练时的 beta_start
    beta_end = 0.02               
    beta_schedule = "sloped_linear"
    clip_sample = True

config = EvalConfig()

# ================= 2. 加载模型与采样器 =================
print(f"正在加载模型: {config.model_path}")
model = UNet2DModel.from_pretrained(config.model_path, use_safetensors=True).to(config.device)
model.eval()

# 采样器配置（必须与你发现效果好的参数一致）
scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=config.beta_start,
    beta_end=config.beta_end, 
    beta_schedule=config.beta_schedule,
    clip_sample=config.clip_sample, 
    prediction_type="epsilon"
)
scheduler.set_timesteps(100) # 评价时可适当降低步数以提速

# ================= 3. 准备指标计算器 =================
# FID 需要特征提取 (默认使用 InceptionV3 的 2048 维特征)
fid_metric = FrechetInceptionDistance(feature=2048).to(config.device)
# IS 计算
is_metric = InceptionScore().to(config.device)

# ================= 4. 处理真实图像 (基准) =================
print("正在提取真实图像特征...")
transform = transforms.Compose([
    transforms.Resize((config.image_size, config.image_size)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x * 255).to(torch.uint8)) # Metrics 要求 uint8 格式
])

real_dataset = datasets.ImageFolder(config.real_data_root, transform=transform)
real_loader = DataLoader(real_dataset, batch_size=config.batch_size, shuffle=True)

with torch.no_grad():
    for i, (images, _) in enumerate(tqdm(real_loader)):
        fid_metric.update(images.to(config.device), real=True)
        if i * config.batch_size >= config.num_generate_images:
            break

# ================= 5. 生成图像并计算 =================
print(f"正在生成 {config.num_generate_images} 张图像进行评估...")
num_batches = config.num_generate_images // config.batch_size

with torch.no_grad():
    for i in tqdm(range(num_batches)):
        # 均匀分配 4 个类别
        labels = torch.tensor([0, 1, 2, 3] * (config.batch_size // 4)).to(config.device)
        noise = torch.randn((config.batch_size, 3, config.image_size, config.image_size)).to(config.device)
        
        # DDIM 采样过程
        curr_images = noise
        for t in scheduler.timesteps:
            model_output = model(curr_images, t, class_labels=labels).sample
            curr_images = scheduler.step(model_output, t, curr_images).prev_sample
        
        # 后处理并转为 uint8
        curr_images = (curr_images / 2 + 0.5).clamp(0, 1)
        curr_images_uint8 = (curr_images * 255).to(torch.uint8)
        
        # 更新指标
        fid_metric.update(curr_images_uint8, real=False)
        is_metric.update(curr_images_uint8)

# ================= 6. 输出并保存最终分数 =================
print("\n" + "="*30)
# 强制转换为 Python 原生 float，确保 json 可序列化
fid_val = float(fid_metric.compute())
is_mean_res, is_std_res = is_metric.compute()
is_mean_val = float(is_mean_res)
is_std_val = float(is_std_res)

print(f"✅ FID Score: {fid_val:.4f}")
print(f"✅ Inception Score: {is_mean_val:.4f} ± {is_std_val:.4f}")
print("="*30)

# --- 保存机制 ---
import datetime
import json

run_name = os.path.basename(config.model_path)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
save_filename = f"FID_result_{run_name}.txt"

# 1. 保存 TXT
with open(save_filename, "a", encoding="utf-8") as f:
    f.write(f"\n--- 测试时间: {timestamp} ---\n")
    f.write(f"模型路径: {config.model_path}\n")
    f.write(f"FID Score: {fid_val:.4f}\n")
    f.write(f"IS Score: {is_mean_val:.4f} ± {is_std_val:.4f}\n")
    f.write("-" * 40 + "\n")

# 2. 保存 JSON (确保这里所有的值都是 float 而非 Tensor)
json_data = {
    "timestamp": timestamp,
    "model": config.model_path,
    "fid": fid_val,            # 确认是 float
    "is_mean": is_mean_val,    # 确认是 float
    "is_std": is_std_val,      # 确认是 float
    "params": {
        "beta_end": float(scheduler.config.beta_end), 
        "clip": bool(scheduler.config.clip_sample),
        "steps": int(len(scheduler.timesteps))
    }
}

history_file = "all_experiments_log.json"

# 2. 核心逻辑：读取旧数据并追加
history = []
if os.path.exists(history_file):
    try:
        with open(history_file, "r", encoding="utf-8") as jf:
            # 读取现有列表
            content = jf.read().strip()
            if content:
                history = json.loads(content)
                # 确保读取到的是列表格式
                if not isinstance(history, list):
                    history = [history]
    except (json.JSONDecodeError, Exception) as e:
        print(f"⚠️ 读取旧日志出错 (可能是文件损坏): {e}")
        history = []

# 3. 将新结果放入列表末尾
history.append(json_data)

# 4. 写回文件 (使用 indent 保持人类可读)
with open(history_file, "w", encoding="utf-8") as jf:
    json.dump(history, jf, indent=4, ensure_ascii=False)

print(f"📊 实验结果已追加至: {history_file} (当前累计 {len(history)} 条记录)")