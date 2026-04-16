import torch
import os
import shutil
import torch.nn as nn
from diffusers import UNet2DConditionModel, DDPMScheduler
from tqdm import tqdm
from PIL import Image
import time
import torch.backends.cudnn as cudnn
from concurrent.futures import ThreadPoolExecutor

# ================= 1. 配置与加速开关 =================
class AugConfig:
    original_data_dir = "./Base_datasets/train" 
    output_dir = "./ldm_and_ddpm_augmented/train"
    model_path = r"ddpm_variance_22\checkpoint_epoch_9" 
    
    target_count = 5000  
    num_inference_steps = 50        
    guidance_scale = 5.0            
    image_size = 256                
    batch_size = 16                 
    variance_scale = 0.00001        
    
    device = "cuda"
    use_fp16 = True
    use_channels_last = True        
    max_workers = 8                 # 用于异步保存图片的 CPU 线程数

config = AugConfig()

# 开启 cuDNN 自动内核优化
cudnn.benchmark = True

# ================= 2. 模型组件 =================
class LabelEmbedding(nn.Module):
    def __init__(self, num_classes=5, embedding_dim=256, num_tokens=8):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embedding_dim)
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
        tokens = self.mlp(x).view(-1, self.num_tokens, self.embedding_dim)
        return tokens + self.pos

def load_models():
    dtype = torch.float16 if config.use_fp16 else torch.float32
    print(f"🚀 初始化模型 | FP16: {config.use_fp16} | Channels Last: {config.use_channels_last}")
    
    model = UNet2DConditionModel.from_pretrained(config.model_path, torch_dtype=dtype).to(config.device)
    
    # 内存布局优化：显卡处理更高效
    if config.use_channels_last:
        model = model.to(memory_format=torch.channels_last)
    
    # 标签投影层
    label_proj = LabelEmbedding(num_classes=5).to(config.device).to(dtype)
    proj_path = os.path.join(config.model_path, "label_proj.pt")
    if os.path.exists(proj_path):
        label_proj.load_state_dict(torch.load(proj_path, map_location=config.device))
    
    model.eval()
    label_proj.eval()
    
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
        variance_type="learned_range"
    )
    return model, label_proj, scheduler

# 异步保存函数
def save_image_async(img_np, path):
    Image.fromarray(img_np).save(path)

# ================= 3. 执行逻辑 =================
def run():
    model, label_proj, scheduler = load_models()
    dtype = torch.float16 if config.use_fp16 else torch.float32
    
    os.makedirs(config.output_dir, exist_ok=True)
    categories = ['1', '2', '3', '4']

    # 使用线程池加速 I/O
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        for idx, cat in enumerate(categories):
            target_dir = os.path.join(config.output_dir, cat)
            os.makedirs(target_dir, exist_ok=True)
            
            # 基础数据拷贝逻辑
            source_dir = os.path.join(config.original_data_dir, cat)
            if os.path.exists(source_dir):
                for img_f in os.listdir(source_dir):
                    if not os.path.exists(os.path.join(target_dir, img_f)):
                        shutil.copy(os.path.join(source_dir, img_f), target_dir)
            
            existing = len([f for f in os.listdir(target_dir) if f.lower().endswith(('.png', '.jpg'))])
            needed = config.target_count - existing
            
            if needed <= 0:
                print(f"✅ 类别 {cat} 已满，跳过。")
                continue
            
            print(f"\n📂 类别 {cat} | 缺口: {needed} | 启动生成...")
            pbar = tqdm(total=needed, desc=f"Cat {cat}")
            
            while needed > 0:
                cur_bs = min(config.batch_size, needed)
                
                x_t = torch.randn((cur_bs, 3, config.image_size, config.image_size), 
                                  device=config.device, dtype=dtype)
                if config.use_channels_last:
                    x_t = x_t.to(memory_format=torch.channels_last)
                
                scheduler.set_timesteps(config.num_inference_steps)
                
                with torch.no_grad():
                    # 准备 Embedding
                    labels = torch.full((cur_bs,), idx, dtype=torch.long, device=config.device)
                    uncond_labels = torch.full((cur_bs,), 4, dtype=torch.long, device=config.device)
                    cond_emb = label_proj(labels)
                    uncond_emb = label_proj(uncond_labels)

                    # 扩散循环
                    for t in scheduler.timesteps:
                        # 开启混合精度自动转换
                        with torch.cuda.amp.autocast(enabled=config.use_fp16):
                            inp = torch.cat([x_t] * 2)
                            emb = torch.cat([uncond_emb, cond_emb])
                            ts = torch.full((cur_bs * 2,), t, device=config.device, dtype=torch.long)
                            
                            # 模型推理
                            out = model(inp, ts, encoder_hidden_states=emb).sample
                            
                            # 处理 6 通道输出
                            noise_p, var_p = out.chunk(2, dim=1)
                            n_uncond, n_cond = noise_p.chunk(2)
                            noise_pred = n_uncond + config.guidance_scale * (n_cond - n_uncond)
                            
                            v_uncond, v_cond = var_p.chunk(2)
                            var_pred = v_cond * config.variance_scale
                            
                            final_out = torch.cat([noise_pred, var_pred], dim=1)
                        
                        # Scheduler step
                        x_t = scheduler.step(final_out, t, x_t).prev_sample

                    # 图像后处理 (转回 FP32 保证画质)
                    imgs = ((x_t.float() + 1) / 2).clamp(0, 1) ** 0.8
                    imgs_np = (imgs.cpu().permute(0, 2, 3, 1).numpy() * 255).astype("uint8")
                    
                    # 异步提交保存任务，不阻塞下一波显卡计算
                    for i in range(cur_bs):
                        save_path = os.path.join(target_dir, f"gen_{cat}_{time.time_ns()}_{i}.png")
                        executor.submit(save_image_async, imgs_np[i], save_path)
                
                needed -= cur_bs
                pbar.update(cur_bs)
            pbar.close()

    print(f"\n🎉 所有类别扩充完成！路径: {config.output_dir}")

if __name__ == "__main__":
    run()