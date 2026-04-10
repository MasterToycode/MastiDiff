import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, utils
from torchvision.datasets import ImageFolder
from diffusers import UNet2DConditionModel, DDPMScheduler, AutoencoderKL
from diffusers.training_utils import EMAModel
from accelerate import Accelerator
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
from diffusers import DPMSolverMultistepScheduler 
from diffusers.optimization import get_cosine_schedule_with_warmup

# 告诉程序你的自定义缓存库在哪里
os.environ["HF_HOME"] = r"D:\hf_cache" 
# 开启离线模式（1代表开启，0代表关闭），这不是文件夹名
os.environ["HF_HUB_OFFLINE"] = "1"

# --- 1. 配置类 ---
class Config:
    data_root = "./datasets"
    output_dir = "ldm_udder_v22"  
    initial_checkpoint="ldm_udder_v21/checkpoint_epoch_99"
    image_size = 512
    train_batch_size = 1
    num_epochs = 100
    learning_rate = 5e-6  
    lr_warmup_steps = 100
    num_classes = 4
    uncond_label = 4
    p_uncond = 0.2
    save_model_epochs = 25
    save_image_epochs = 25  
    
    # LDM 核心配置
    vae_model = r"D:\hf_cache\hub\models--stabilityai--sd-vae-ft-mse\snapshots\31f26fdeee1355a5c34592e401dd41e45d25a493" 
    latent_channels = 4
    latent_size = 64 
    scale_factor = 0.18215 

    # Scheduler 细节
    beta_start = 0.00085
    beta_end = 0.02
    beta_schedule = "scaled_linear"
    clip_sample = False

    #CFG 引导强度
    cfg=4
    cross_attention_dim = 512

config = Config()

# --- 2. 辅助函数：绘图与权重同步 ---

def save_loss_plot(loss_history, output_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, alpha=0.3, color='blue', label='Batch Loss')
    if len(loss_history) > 50:
        smooth_loss = np.convolve(loss_history, np.ones(50)/50, mode='valid')
        plt.plot(smooth_loss, color='red', label='Smoothed Loss')
    plt.title("Training Loss Curve")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()

def sync_ema_to_model(model, ema):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in ema.shadow_params:
                param.copy_(ema.shadow_params[name].to(param.device))


# 定义拉普拉斯算子，用于提取图像边缘
def laplacian_kernel(x):
    # 使用 3x3 算子提取高频特征
    kernel = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=x.dtype, device=x.device)
    # 转换成单通道计算边缘（取 RGB 均值）
    return F.conv2d(x.mean(dim=1, keepdim=True), kernel, padding=1)


class LabelEmbedding(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.embedding = nn.Embedding(num_classes,512)

        self.dropout = nn.Dropout(0.5)
        
        self.mlp = nn.Sequential(
            nn.Linear(512,1024),
            nn.SiLU(),
            nn.Linear(1024,512*8)
        )

        self.pos = nn.Parameter(torch.randn(8,512) * 0.01)

    def forward(self,labels):
        
        x = self.embedding(labels)
        x = self.dropout(x)
        tokens = self.mlp(x).view(-1,8,512)
        noise = torch.randn_like(x) * 0.01
        x = x + noise
        tokens = self.dropout(tokens)
        return tokens + self.pos
from torchvision.models import vgg16, VGG16_Weights


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features.eval()
        
        self.stage1 = vgg[:4]   # 捕获边缘和细节
        self.stage2 = vgg[4:9]  # 捕获纹理
        self.stage3 = vgg[9:16] # 捕获结构骨架
        
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, pred, target):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)
        
        pred_norm = (pred - mean) / std
        target_norm = (target - mean) / std
        
        feat1_pred = self.stage1(pred_norm)
        feat1_target = self.stage1(target_norm)
        
        feat2_pred = self.stage2(feat1_pred)
        feat2_target = self.stage2(feat1_target)
        
        feat3_pred = self.stage3(feat2_pred)
        feat3_target = self.stage3(feat2_target) 
        
        loss = 1.2 * F.l1_loss(feat1_pred, feat1_target) + \
               0.7 * F.l1_loss(feat2_pred, feat2_target) + \
               0.2 * F.l1_loss(feat3_pred, feat3_target)
        return loss
    

def adjust_lr(optimizer, base_lr, epoch, total_epochs, warmup_steps=0, current_step=0):
    """动态学习率分阶段调节，支持warmup"""
    # Warmup阶段
    if warmup_steps > 0 and current_step < warmup_steps:
        warmup_factor = current_step / max(1, warmup_steps)
        lr_factor = 0.1 + 0.9 * warmup_factor  
    else:
        # 分阶段调节
        progress = epoch / total_epochs
        if progress < 0.1:
            lr_factor = 0.5      # 冷启动：稳定阶段
        elif progress < 0.7:
            lr_factor = 1.0      # 主学习阶段
        else:
            lr_factor = 0.3      # 精调阶段
    
    new_lr = base_lr * lr_factor
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr
    return new_lr


def train_loop():
    accelerator = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=16)
    os.makedirs(config.output_dir, exist_ok=True)

    # A. VAE 准备
    vae = AutoencoderKL.from_pretrained(config.vae_model).to(accelerator.device)
    vae.requires_grad_(False)

    # B. 数据准备
    preprocess = transforms.Compose([
    transforms.RandomResizedCrop(config.image_size, scale=(0.8, 1.0), ratio=(1, 1)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
])

    dataset = ImageFolder(root=os.path.join(config.data_root, 'train'), transform=preprocess)
    # 权重采样平衡类别
    targets = torch.tensor(dataset.targets)
    class_sample_count = np.array([len(np.where(targets == t)[0]) for t in np.unique(targets)])
    print(f"各类别样本数量: {class_sample_count}")
    weight = 1. / torch.from_numpy(class_sample_count).float()
    samples_weight = torch.tensor([weight[t] for t in targets])
    sampler = WeightedRandomSampler(weights=samples_weight, num_samples=len(samples_weight), replacement=True)
    sample_counts = torch.bincount(targets)
    print(sample_counts)
    train_dataloader = DataLoader(dataset, batch_size=config.train_batch_size, sampler=sampler, num_workers=4,persistent_workers=True,pin_memory=True)

    # C. UNet (LDM 架构)
    model = UNet2DConditionModel(
        sample_size=config.latent_size,
        in_channels=config.latent_channels,
        out_channels=config.latent_channels,
        layers_per_block=2,
        block_out_channels=(128, 256, 512, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        num_class_embeds=config.num_classes + 1,
        cross_attention_dim=512,
        dropout=0.5,   # ⭐ 加这一行
    )

    #label_proj = nn.Linear(config.num_classes + 1, 512).to(accelerator.device)
    label_proj = LabelEmbedding(config.num_classes+1)

    if config.initial_checkpoint and os.path.exists(config.initial_checkpoint):
        print(f"==> 正在从 {config.initial_checkpoint} 加载模型权重...")
        # 加载 UNet
        model = UNet2DConditionModel.from_pretrained(config.initial_checkpoint).to(accelerator.device)
        print("✅ UNet 权重加载成功！")

        # 加载 label_proj.pt 
        label_proj_path = os.path.join(config.initial_checkpoint, "label_proj.pt")
        if os.path.exists(label_proj_path):
            label_proj.load_state_dict(torch.load(label_proj_path, map_location=accelerator.device))
            print(f"✅ 成功加载 label_proj 权重：{label_proj_path}")
        else:
            print("⚠️ 未找到 label_proj.pt，将使用随机初始化的嵌入层。")
    else:
        print("==> 未指定权重或路径不存在，将从随机初始化开始训练。")
    
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000, 
        beta_start=config.beta_start, 
        beta_end=config.beta_end, 
        beta_schedule=config.beta_schedule,
        clip_sample=config.clip_sample
    )

    
    base_lr = config.learning_rate

    optimizer = torch.optim.AdamW(
                                    list(model.parameters()) +
                                    list(label_proj.parameters()),
                                    lr=base_lr
    )

    model, label_proj, optimizer, train_dataloader = accelerator.prepare(
        model, label_proj, optimizer, train_dataloader
    )

    # --- 修复C：创建cosine scheduler ---
    # 计算总训练步数
    num_training_steps = config.num_epochs * len(train_dataloader)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=num_training_steps,
    )

    ema_model = EMAModel(model.parameters(), decay=0.99, model_cls=UNet2DConditionModel, model_config=model.config)  
    ema_model.to(accelerator.device)

    # 创建PerceptualLoss实例（移到循环外部）
    perceptual_loss_fn = PerceptualLoss().to(accelerator.device)

    # 日志记录
    loss_history = []
    log_file = open(os.path.join(config.output_dir, "train_log.txt"), "w")
    
    # 步数计数器，用于warmup
    global_step = 0

    main_pbar = tqdm(
        range(config.num_epochs), 
        desc="Total Training Progress", 
        position=0, 
        disable=not accelerator.is_local_main_process
    )

    for epoch in main_pbar:
        model.train()
        epoch_losses = []
        
        # leave=False 确保该进度条跑完 100% 后自动从屏幕消失
        # position=1 确保它显示在总进度条的下方
        sub_pbar = tqdm(
            train_dataloader, 
            desc=f"Epoch {epoch}", 
            leave=False, 
            position=1, 
            disable=not accelerator.is_local_main_process
        )
        
        for step, (images, labels) in enumerate(sub_pbar):
            with torch.no_grad():
                latents = vae.encode(images).latent_dist.sample() * config.scale_factor

            # 先生成timesteps，然后再计算offset_scale
            timesteps = torch.randint(0, 1000, (latents.shape[0],), device=latents.device).long()
            
            noise = torch.randn_like(latents)
            offset_scale = 0.005 * (1 - timesteps.float() / 1000).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            noise = noise + offset_scale * torch.randn_like(noise)

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            mask = torch.bernoulli(torch.full(labels.shape, config.p_uncond)).to(accelerator.device)
            train_labels = torch.where(mask > 0, torch.tensor(config.uncond_label).to(accelerator.device), labels)

            with accelerator.accumulate(model):
               # 1. 准备 One-hot
                #labels_one_hot = F.one_hot(train_labels, num_classes=config.num_classes + 1).float()
                # 2. 通过投影层映射到 512 维，并增加序列维度给 Cross-Attention
                # 现在不再是稀疏的 0/1，而是具有可学习权重的特征向量
                #encoder_hidden_states = label_proj(labels_one_hot).unsqueeze(1) 
                label_tokens = label_proj(train_labels)
                encoder_hidden_states = label_tokens

                # 3. 传入模型
                noise_pred = model(
                    noisy_latents, 
                    timesteps, 
                    encoder_hidden_states=encoder_hidden_states, 
                    class_labels=train_labels
                ).sample
                # --- Min-SNR 5 策略：平衡不同 timestep 的学习权重 ---
                alphas_cumprod = noise_scheduler.alphas_cumprod.to(accelerator.device)
                snr = alphas_cumprod[timesteps] / (1 - alphas_cumprod[timesteps])
                mse_loss_weights = torch.stack([snr, 10 * torch.ones_like(snr)], dim=1).min(dim=1)[0] / snr

                # 计算 MSE Loss
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="none")
                loss = (loss.mean(dim=[1, 2, 3]) * mse_loss_weights).mean()

                # --- 修复A：在图像空间计算感知损失和边缘损失 ---
                # 计算预测的x0（去噪后的潜在空间）
                sqrt_alpha_cumprod = alphas_cumprod[timesteps].sqrt().view(-1, 1, 1, 1)
                sqrt_one_minus_alpha = (1 - alphas_cumprod[timesteps]).sqrt().view(-1, 1, 1, 1)
                
                # 预测的x0
                x0_pred = (noisy_latents - sqrt_one_minus_alpha * noise_pred) / (sqrt_alpha_cumprod + 1e-8)
                # 真实的x0就是原始latents
                x0_true = latents
                
                decoded_pred = vae.decode(x0_pred / config.scale_factor).sample
                # 解码到图像空间
                with torch.no_grad():
                    decoded_true = vae.decode(x0_true / config.scale_factor).sample
                    
                    # 将范围从[-1,1]调整到[0,1]
                    decoded_pred = (decoded_pred / 2 + 0.5).clamp(0, 1)
                    decoded_true = (decoded_true / 2 + 0.5).clamp(0, 1)
                
                # 在计算感知损失前下采样
                # align_corners=False 和 antialias=True 能提供更平滑的下采样
                decoded_pred_256 = F.interpolate(
                    decoded_pred, size=(256, 256), mode="bicubic", align_corners=False, antialias=True
                )
                decoded_true_256 = F.interpolate(
                    decoded_true, size=(256, 256), mode="bicubic", align_corners=False, antialias=True
                )

                perceptual_loss_val = perceptual_loss_fn(decoded_pred_256, decoded_true_256)
                # 在图像空间计算感知损失
                
                # 在图像空间计算边缘损失
                pred_edge_img = laplacian_kernel(decoded_pred)
                true_edge_img = laplacian_kernel(decoded_true)
                edge_loss = F.mse_loss(pred_edge_img, true_edge_img)

                # --- 修复B：调整感知损失权重策略 ---
                # 初期关闭感知损失，逐步增加
                if epoch < 5:
                    perceptual_weight = 0.0
                else:
                    perceptual_weight = 0.15 * (epoch - 5) / max(1, config.num_epochs - 5)
                    perceptual_weight = min(perceptual_weight, 0.15)
                
                # 边缘损失权重逐步增加
                edge_weight = 0.5 + 0.5 * (epoch / config.num_epochs)
                
                tloss = loss + edge_weight * edge_loss + perceptual_weight * perceptual_loss_val 

                accelerator.backward(tloss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0) 
                optimizer.step()
                # --- 修复C：在每个optimizer.step()后调用scheduler.step() ---
                lr_scheduler.step()
                optimizer.zero_grad()
                ema_model.step(model.parameters())
                
                # 更新步数计数器
                global_step += 1

            # 获取当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            
            loss_val = tloss.detach().item()
            tedge_loss=edge_loss.detach().item()
            perceptual_loss_val_item = perceptual_loss_val.detach().item()
            loss_history.append(loss_val)
            epoch_losses.append(loss_val) 

            sub_pbar.set_postfix({"loss": f"{loss_val:.4f}", "edge_loss": f"{tedge_loss:.4f}", "perceptual_loss":f"{perceptual_loss_val_item:.4f}", "lr": f"{current_lr:.2e}"})

        # --- 3. 周期结束后的清理与打印 ---
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        
        if accelerator.is_local_main_process:
            # 使用 tqdm.write 打印，不会破坏进度条结构
            # 这一行会留在屏幕上，记录每一轮的最终结果
            tqdm.write(f"🌟 Epoch {epoch:03d} Finished | Avg Loss: {avg_loss:.6f}" )
            
            # 更新主进度条的显示信息
            main_pbar.set_postfix({"last_avg_loss": f"{avg_loss:.4f}"})
            
            # 写入日志文件
            log_file.write(f"Epoch {epoch}: Avg Loss = {avg_loss:.6f}\n")
            log_file.flush()

        # 周期性采样与绘图
        if accelerator.is_main_process:
            if (epoch + 1) % config.save_image_epochs == 0:
                save_ldm_sample(accelerator, model, label_proj,ema_model, vae, epoch, config)

            if (epoch + 1) % config.save_model_epochs == 0:
                save_path = os.path.join(config.output_dir, f"checkpoint_epoch_{epoch}")
                unwrapped_model = accelerator.unwrap_model(model)
                
                # === 修改点 C：保存时不覆盖当前训练权重 ===
                # 1. 先备份当前训练中的参数到 CPU
                current_weights = [p.detach().cpu().clone() for p in unwrapped_model.parameters()]
                
                # 2. 将 EMA 权重拷贝到模型用于保存
                ema_model.copy_to(unwrapped_model.parameters()) 
                unwrapped_model.save_pretrained(save_path)
                
                # 3. 立即恢复原来的训练权重
                for p, sw in zip(unwrapped_model.parameters(), current_weights):
                    p.data.copy_(sw.to(accelerator.device))
                
                # 4. 清理
                del current_weights
                torch.cuda.empty_cache()

                # 保存 label_proj (保持不变)
                unwrapped_proj = accelerator.unwrap_model(label_proj)
                torch.save(unwrapped_proj.state_dict(), os.path.join(save_path, "label_proj.pt"))
                print(f"✅ 已安全保存 EMA 检查点并恢复训练权重")

    log_file.close()
    if accelerator.is_main_process:
        save_loss_plot(loss_history, config.output_dir)
        print("🚀 训练完成，Loss 曲线和权重已保存！")


def save_ldm_sample(accelerator, model, label_proj, ema_model, vae, epoch, config):
    scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        beta_schedule=config.beta_schedule,
        algorithm_type="dpmsolver++", 
        solver_order=2,               
    )
    scheduler.set_timesteps(30) 
    
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_proj = accelerator.unwrap_model(label_proj)
    
    # 权重备份（防止 EMA 覆盖训练权重导致后续训练失效）
    orig_params = [p.detach().cpu().clone() for p in unwrapped_model.parameters()]
    ema_model.copy_to(unwrapped_model.parameters())
    unwrapped_model.eval()
    unwrapped_proj.eval()
    
    labels = torch.tensor([0, 1, 2, 3]).to(accelerator.device)
    latents = torch.randn(4, config.latent_channels, config.latent_size, config.latent_size).to(accelerator.device)
    
    for t in scheduler.timesteps:
        with torch.no_grad():
            def get_projected_emb(labels_tensor):

                tokens = unwrapped_proj(labels_tensor)

                return tokens

            # 条件与无条件推理
            cond_emb = get_projected_emb(labels)
            noise_pred_cond = unwrapped_model(latents, t, cond_emb, class_labels=labels).sample
            
            uncond_labels = torch.full_like(labels, config.uncond_label)
            uncond_emb = get_projected_emb(uncond_labels)
            noise_pred_uncond = unwrapped_model(latents, t, uncond_emb, class_labels=uncond_labels).sample
            
            # CFG 融合
            noise_pred = noise_pred_uncond + config.cfg * (noise_pred_cond - noise_pred_uncond)
            
            # 步进
            latents = scheduler.step(noise_pred, t, latents).prev_sample
            

    with torch.no_grad():
        # 解码前确保数值不会过载
        images = vae.decode(latents / config.scale_factor).sample
        images = (images / 2 + 0.5).clamp(0, 1)
    
    utils.save_image(images, f"{config.output_dir}/sample_epoch_{epoch}.png", nrow=2)
    
    # 恢复训练权重
    for p, orig_p in zip(unwrapped_model.parameters(), orig_params):
        p.data.copy_(orig_p.data)
    
    # 显式删除引用并清空缓存
    del orig_params
    torch.cuda.empty_cache()

    # --- 修复D：添加采样监控信息 ---
    print(f"采样完成：epoch {epoch}，采样步数：30，CFG强度：{config.cfg}")

if __name__ == "__main__":
    train_loop()