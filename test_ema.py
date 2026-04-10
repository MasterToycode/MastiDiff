import torch
import os
from diffusers import UNet2DModel, DDIMScheduler
from torchvision import utils
from tqdm import tqdm

# ================= 1. 配置部分 =================
class TestConfig:
    # 路径配置：指向你保存 EMA 模型的文件夹（包含 .safetensors 的那个目录）
    model_path = "ddpm-udder-results6/ema_model_epoch_119" 
    
    # 结果保存目录
    output_dir = "test_results_ema3"
    
    # 采样配置
    num_samples_per_class = 4      # 每个类别生成 4 张，总共 16 张
    num_inference_steps = 300       # DDIM 步数，数值越大，雪花感通常越少（建议 50-250）
    image_size = 256
    device = "cuda" if torch.cuda.is_available() else "cpu"
    beta_start = 0.0001 # 训练时的 beta_start
    beta_end = 0.02               
    beta_schedule = "scaled_linear"
    clip_sample = True

config = TestConfig()
os.makedirs(config.output_dir, exist_ok=True)

# ================= 2. 加载 EMA 模型 =================
print(f"正在从 Safetensors 文件夹加载模型: {config.model_path}")

try:
    # 使用 from_pretrained 自动加载结构和权重
    model = UNet2DModel.from_pretrained(
        config.model_path, 
        use_safetensors=True
    )
    model.to(config.device)
    model.eval() # 切换到推理模式
    print("✅ EMA 模型加载成功！")
except Exception as e:
    print(f"❌ 加载失败，请检查路径是否正确。错误信息: {e}")
    exit()

# ================= 3. 设置 DDIM 采样器 =================
# 这里的参数必须与你训练最后阶段的配置（Scaled Linear）完全一致
scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=config.beta_start,
    beta_end=config.beta_end,                # 匹配你训练时的减小后的噪声强度
    beta_schedule=config.beta_schedule,  # 匹配你训练时的平滑调度
    clip_sample=config.clip_sample,              
    prediction_type="epsilon"
)

# ================= 4. 执行生成逻辑 =================
def run_inference():
    # 准备类别标签 [0,1,2,3, 0,1,2,3 ...]
    classes = [0, 1, 2, 3] * config.num_samples_per_class
    labels = torch.tensor(classes).to(config.device)
    num_images = len(labels)
    
    # 生成起始随机噪声
    # 种子设定（可选，取消注释可固定结果）: torch.manual_seed(42)
    image = torch.randn(
        (num_images, 3, config.image_size, config.image_size), 
        device=config.device
    )
    
    # 设置采样步数
    scheduler.set_timesteps(config.num_inference_steps)
    
    print(f"开始生成 {num_images} 张图像 (步数: {config.num_inference_steps})...")
    
    with torch.no_grad():
        for t in tqdm(scheduler.timesteps, desc="DDIM Sampling"):
            # 1. 模型预测当前步的噪声
            model_output = model(image, t, class_labels=labels).sample
            
            # 2. 计算上一步的图像 (去噪)
            # DDIM 的 step 函数会自动处理确定性采样，消除 DDPM 的随机雪花感
            image = scheduler.step(model_output, t, image).prev_sample
            
    # 后处理：将 [-1, 1] 映射回 [0, 1]
    image = (image / 2 + 0.5).clamp(0, 1)
    
    # 保存为网格图
    # nrow=4 代表每行显示 4 张（刚好对应 4 个类别）
    save_path = os.path.join(config.output_dir, "ema_final_comparison.png")
    utils.save_image(image, save_path, nrow=4)
    
    print(f"\n✨ 生成完成！")
    print(f"文件保存路径: {save_path}")
    print(f"提示：如果图片仍有轻微噪点，可尝试将 num_inference_steps 设为 250。")

if __name__ == "__main__":
    run_inference()