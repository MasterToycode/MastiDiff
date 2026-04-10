import torch
import os
import datetime
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from tqdm import tqdm

# ================= 1. 配置路径 =================
REAL_DIR = "datasets/train"      # 真实数据根目录
FAKE_DIR = "Base_datasets_augmented_4/train"      # 已生成的 5000 张/类 根目录
SAVE_FILE = "FID_lpips.txt"              # 保存的文件名

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
IMAGE_SIZE = 256 

# ================= 2. 定义命名函数替代 Lambda (解决 PicklingError) =================
def to_uint8_tensor(x):
    return (x * 255).to(torch.uint8)

# 将变换逻辑封装
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Lambda(to_uint8_tensor) # 使用命名函数
])

def get_loader(path):
    dataset = datasets.ImageFolder(path, transform=transform)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ================= 3. 主程序入口 =================
def main():
    real_loader = get_loader(REAL_DIR)
    fake_loader = get_loader(FAKE_DIR)

    # 初始化指标计算器
    fid_metric = FrechetInceptionDistance(feature=2048).to(DEVICE)
    is_metric = InceptionScore().to(DEVICE)

    # 执行计算
    with torch.no_grad():
        # A. 提取真实图像特征 (仅用于 FID)
        print(f"正在读取真实图像特征: {REAL_DIR}")
        for imgs, _ in tqdm(real_loader, desc="Real Images"):
            fid_metric.update(imgs.to(DEVICE), real=True)

        # B. 提取已生成图像特征 (用于 FID 和 IS)
        print(f"正在读取生成图像特征并计算 IS: {FAKE_DIR}")
        for imgs, _ in tqdm(fake_loader, desc="Fake Images"):
            imgs_gpu = imgs.to(DEVICE)
            fid_metric.update(imgs_gpu, real=False)
            is_metric.update(imgs_gpu)

    # 获取最终数值
    print("正在计算最终指标...")
    fid_score = fid_metric.compute().item()
    is_mean, is_std = is_metric.compute()
    is_mean_val = is_mean.item()
    is_std_val = is_std.item()

    # 追加保存 (Append 模式)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(SAVE_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n" + "="*50 + "\n")
        f.write(f"测试时间: {timestamp}\n")
        f.write(f"评估文件夹: {FAKE_DIR}\n")
        f.write(f"🎯 FID Score: {fid_score:.4f}\n")
        f.write(f"✅ Inception Score (IS): {is_mean_val:.4f} ± {is_std_val:.4f}\n")
        f.write("="*50 + "\n")

    print(f"\n✅ 计算完成！结果已追加至: {SAVE_FILE}")
    print(f"FID: {fid_score:.4f} | IS: {is_mean_val:.4f}")

if __name__ == '__main__':
    main()