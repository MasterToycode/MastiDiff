import torch
from torchvision import transforms, utils
from PIL import Image
from diffusers import AutoencoderKL
import os
# 告诉程序你的自定义缓存库在哪里
os.environ["HF_HOME"] = r"D:\hf_cache" 
# 开启离线模式（1代表开启，0代表关闭），这不是文件夹名
os.environ["HF_HUB_OFFLINE"] = "1"
def test_vae():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae_model = r"D:\hf_cache\hub\models--stabilityai--sd-vae-ft-mse\snapshots\31f26fdeee1355a5c34592e401dd41e45d25a493"
    scale_factor = 0.18215
    image_path = "datasets/train/4/171_3.png" # 替换为你真实存在的图片路径

    # 1. 加载 VAE
    vae = AutoencoderKL.from_pretrained(vae_model).to(device)
    
    # 2. 预处理 (保持与训练一致)
    preprocess = transforms.Compose([
    transforms.RandomResizedCrop(512, scale=(0.95, 1.0), ratio=(1, 1)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
])
    img = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(img).unsqueeze(0).to(device)

    # 3. 编解码过程 (模拟训练时的 Latent 缩放)
    with torch.no_grad():
        # 编码到潜空间并应用缩放因子
        latents = vae.encode(input_tensor).latent_dist.sample() * scale_factor
        # 从潜空间解码回像素
        rec_image = vae.decode(latents / scale_factor).sample

    # 4. 反归一化并保存
    rec_image = (rec_image / 2 + 0.5).clamp(0, 1)
    input_image = (input_tensor / 2 + 0.5).clamp(0, 1)
    
    comparison = torch.cat([input_image, rec_image], dim=-1)
    utils.save_image(comparison, "vae_test.png")
    print("测试完成！请检查 vae_test.png。左边是原图，右边是 VAE 还原图。")

if __name__ == "__main__":
    test_vae()