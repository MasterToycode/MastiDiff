import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, utils
from torchvision.datasets import ImageFolder
from diffusers import UNet2DConditionModel, DDPMScheduler
from diffusers.training_utils import EMAModel
from accelerate import Accelerator
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt 
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.utils import is_xformers_available

os.environ["HF_HOME"] = r"D:\hf_cache" 
# 开启离线模式（1代表开启，0代表关闭），这不是文件夹名
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# --- 1. 配置类 ---
class Config:
    data_root = "./datasets"
    output_dir = "ddpm_variance_13"  
    initial_checkpoint = r"ddpm_variance_12\checkpoint_epoch_59"
    image_size = 256
    train_batch_size = 13
    num_epochs = 100
    learning_rate = 1e-5
    lr_warmup_steps = 100
    num_classes = 4
    uncond_label = 4
    p_uncond = 0.1
    save_model_epochs = 10
    save_image_epochs = 10
    weight_offset=0.005
    
    # DDPM 像素空间配置
    in_channels = 3  # RGB图像
    out_channels = 6  # 3 epsilon + 3 variance（像素空间RGB）
    use_variance_prediction = True
    
    # UNet 配置（像素空间，带交叉注意力）
    unet_model = None  # 不使用预训练UNet，从头训练
    cross_attention_dim = 256  
    attention_head_dim = 32  # 注意力头维度
    num_class_embeds = None  # 不使用num_class_embeds，改用交叉注意力
    block_out_channels = (128, 128, 256, 256, 512, 512)
    layers_per_block = 2
    down_block_types = (
    "DownBlock2D",
    "DownBlock2D",
    "CrossAttnDownBlock2D",
    "CrossAttnDownBlock2D",
    "DownBlock2D",
    "DownBlock2D",
)

    up_block_types = (
        "UpBlock2D",
        "UpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    )
    
    # Scheduler 细节
    beta_schedule = "squaredcos_cap_v2"
    clip_sample = True
    
    # 采样配置
    cfg = 2
    sampler = "DDPMScheduler"  

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


# laplacian_kernel 函数已由 LaplacianLoss 类替代，使用 register_buffer 缓存 kernel


class LabelEmbedding(nn.Module):
    """标签嵌入层，将类别标签转换为cross-attention的encoder_hidden_states"""
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
        x = self.dropout(x)
        # ✅ 先加噪声，再送入 MLP
        noise = torch.randn_like(x) * 0.0001
        x = x + noise
        tokens = self.mlp(x).view(-1, self.num_tokens, self.embedding_dim)
        tokens = self.dropout(tokens)
        return tokens + self.pos

from torchvision.models import vgg16, VGG16_Weights


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features.eval()
        self.stage1 = vgg[:4]
        self.stage2 = vgg[4:9]
        self.stage3 = vgg[9:16]
        
        # 优化：直接将 VGG 期望的 [0,1] 均值标准差转换为适应 [-1, 1] 输入的参数
        # VGG 均值 0.485, 0.456, 0.406 -> 在 [-1, 1] 空间下对应约为 -0.03, -0.088, -0.188
        vgg_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        vgg_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        
        self.register_buffer('shift', (2 * vgg_mean - 1))
        self.register_buffer('scale', (2 * vgg_std))
        
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, pred, target):
        # 直接使用 [-1, 1] 的张量进行快速归一化
        pred_norm = (pred - self.shift) / self.scale
        target_norm = (target - self.shift) / self.scale
        
        feat1_pred = self.stage1(pred_norm)
        feat1_target = self.stage1(target_norm)
        feat2_pred = self.stage2(feat1_pred)
        feat2_target = self.stage2(feat1_target)
        feat3_pred = self.stage3(feat2_pred)
        feat3_target = self.stage3(feat2_target) 
        
        return 1.0 * F.l1_loss(feat1_pred, feat1_target) + \
               0.8 * F.l1_loss(feat2_pred, feat2_target) + \
               0.6 * F.l1_loss(feat3_pred, feat3_target)


class LaplacianLoss(nn.Module):
    """拉普拉斯边缘损失，使用register_buffer缓存kernel避免重复创建"""
    def __init__(self):
        super().__init__()
        # 创建3x3拉普拉斯kernel并缓存
        kernel = torch.tensor([[[[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]]])
        self.register_buffer('kernel', kernel)
    
    def forward(self, x):
        # 转换成单通道计算边缘（取RGB均值）
        return F.conv2d(x.mean(dim=1, keepdim=True), self.kernel, padding=1)
    
    
def adjust_lr(optimizer, base_lr, epoch, total_epochs, warmup_steps=500, current_step=0):
    """动态学习率分阶段调节，支持warmup"""
    # Warmup阶段
    if warmup_steps > 0 and current_step < warmup_steps:
        warmup_factor = current_step / max(1, warmup_steps)
        lr_factor = 0.1 + 0.9 * warmup_factor  
    else:
        # 分阶段调节
        progress = epoch / total_epochs
        if progress < 0.1:
            lr_factor = 1.0      # 冷启动：稳定阶段
        elif progress < 0.7:
            lr_factor = 1.0      # 主学习阶段
        else:
            lr_factor = 0.5      # 精调阶段
    
    new_lr = base_lr * lr_factor
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr
    return new_lr


def train_loop():
    accelerator = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=1)
    os.makedirs(config.output_dir, exist_ok=True)

    # A. 像素空间数据准备（无VAE）
    # 直接在像素空间操作，无需VAE编码
    
    # B. 数据准备
    preprocess = transforms.Compose([
    transforms.RandomResizedCrop(config.image_size, scale=(0.95, 1.0), ratio=(1, 1)),
    transforms.RandomHorizontalFlip(),
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
    train_dataloader = DataLoader(dataset, batch_size=config.train_batch_size, sampler=sampler, num_workers=4, persistent_workers=True, pin_memory=True)

    # 标签嵌入层，将类别标签转换为cross-attention的encoder_hidden_states
    label_proj = LabelEmbedding(config.num_classes + 1, embedding_dim=config.cross_attention_dim)

    # 检查点加载逻辑：区分从头训练 vs 恢复训练
    if config.initial_checkpoint and os.path.exists(config.initial_checkpoint):
        print(f"==> 从检查点 {config.initial_checkpoint} 恢复训练...")
        
        # 加载像素空间UNet（6通道输出，带交叉注意力）
        print("==> 加载像素空间UNet（带交叉注意力）...")
        model = UNet2DConditionModel.from_pretrained(
            config.initial_checkpoint,
            local_files_only=True
        )
        print("PASS: UNet加载成功！")
        
        # 加载 label_proj 权重
        label_proj_path = os.path.join(config.initial_checkpoint, "label_proj.pt")
        if os.path.exists(label_proj_path):
            label_proj.load_state_dict(torch.load(label_proj_path, map_location=accelerator.device))
            print(f"PASS: 成功加载 label_proj 权重：{label_proj_path}")
        else:
            print("⚠️ 未找到 label_proj.pt，将使用随机初始化的嵌入层。")
        
    else:
        # 从头训练流程
        print("==> 从头训练，创建像素空间UNet...")
        
        # 创建像素空间UNet（输出6通道：3 epsilon + 3 variance，带交叉注意力）
        print("==> 创建像素空间UNet（带交叉注意力）...")
        model = UNet2DConditionModel(
            sample_size=config.image_size,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            layers_per_block=config.layers_per_block,
            block_out_channels=config.block_out_channels,
            down_block_types=config.down_block_types,
            up_block_types=config.up_block_types,
            cross_attention_dim=config.cross_attention_dim,
            attention_head_dim=config.attention_head_dim,
            num_class_embeds=config.num_class_embeds,
            dropout=0.2,
        )
        print("PASS: UNet创建成功！")

    # F. 启用内存优化
    print("==> 启用内存优化...")
    if is_xformers_available():
        model.enable_xformers_memory_efficient_attention()
        print("[OK] XFormers内存优化已启用")
    else:
        print("[WARNING] XFormers不可用，继续使用标准注意力")

    # 启用梯度检查点
    model.enable_gradient_checkpointing()
    print("[OK] 梯度检查点已启用")

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000, 
        beta_schedule=config.beta_schedule,
        clip_sample=config.clip_sample,
        prediction_type="epsilon",
        variance_type = "learned_range"
    )

    
    base_lr = config.learning_rate

    optimizer = torch.optim.AdamW(
                                    list(model.parameters()) + list(label_proj.parameters()),
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

    ema_model = EMAModel(model.parameters(), decay=0.999, model_cls=UNet2DConditionModel, model_config=model.config,inv_gamma=1.0, power=3/4)  
    ema_model.to(accelerator.device)

    # --- 预计算噪声调度器静态张量（优化GPU利用率）---
    # 这些张量在每个训练步骤中都是静态的，提前计算避免重复计算
    dev = accelerator.device
    # 基础张量
    _ac = noise_scheduler.alphas_cumprod.to(dev)           # ᾱ_t
    _b = noise_scheduler.betas.to(dev)                    # β_t
    _a = 1.0 - _b                                         # α_t
    _ac_prev = torch.cat([torch.ones(1, device=dev), _ac[:-1]])  # ᾱ_{t-1}
    _bt = _b * (1 - _ac_prev) / (1 - _ac)                 # β̃_t
    # 对数张量
    _log_b = torch.log(_b.clamp(min=1e-20))
    _log_bt = torch.log(_bt.clamp(min=1e-20))
    # SNR和平方根张量
    _snr = _ac / (1 - _ac)
    _sqrt_ac = _ac.sqrt()
    _sqrt_ac_prev = _ac_prev.sqrt()
    _sqrt_1m_ac = (1 - _ac).sqrt()
    _sqrt_a = _a.sqrt()  # α_t 的平方根

    # 创建PerceptualLoss实例（移到循环外部）
    perceptual_loss_fn = PerceptualLoss().to(accelerator.device)
    # 创建LaplacianLoss实例（使用register_buffer缓存kernel）
    laplacian_fn = LaplacianLoss().to(accelerator.device)

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
            # 像素空间，无需VAE编码
            x_0 = images  # 形状 [B, 3, H, W]，范围 [-1, 1]

            # 先生成timesteps，然后再计算offset_scale
            timesteps = torch.randint(0, 1000, (x_0.shape[0],), device=x_0.device).long()
            
            noise = torch.randn_like(x_0)
            offset_scale = Config.weight_offset * (1 - timesteps.float() / 1000).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            noise = noise + offset_scale * torch.randn_like(noise)

            noisy_x_t = noise_scheduler.add_noise(x_0, noise, timesteps)

            mask = torch.bernoulli(torch.full(labels.shape, config.p_uncond)).to(accelerator.device, non_blocking=True)
            train_labels = torch.where(mask > 0, torch.tensor(config.uncond_label).to(accelerator.device, non_blocking=True), labels)

            with accelerator.accumulate(model):
                # 通过标签嵌入层生成cross-attention的encoder_hidden_states
                label_tokens = label_proj(train_labels)
                encoder_hidden_states = label_tokens
                
                # 传入模型：像素空间，使用encoder_hidden_states和class_labels进行条件
                model_out = model(
                    noisy_x_t,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample
                # --- Lhybrid 混合损失：拆分噪声预测和方差预测 ---
                # --- [优化版] 混合损失计算：一次性提取所有 timestep 相关参数 ---
                # 1. 提取基础索引张量 (合并索引操作，减少 GPU kernel 启动次数)
                
                # --- 拆分模型输出 ---
                noise_pred, v_pred = model_out.chunk(2, dim=1)
                
                curr_b = _b[timesteps].view(-1, 1, 1, 1)
                curr_a_sqrt = _sqrt_a[timesteps].view(-1, 1, 1, 1)
                curr_ac = _ac[timesteps].view(-1, 1, 1, 1)
                curr_ac_prev = _ac_prev[timesteps].view(-1, 1, 1, 1)
                curr_ac_prev_sqrt = _sqrt_ac_prev[timesteps].view(-1, 1, 1, 1)
                curr_sqrt_ac = _sqrt_ac[timesteps].view(-1, 1, 1, 1)
                curr_sqrt_1m_ac = _sqrt_1m_ac[timesteps].view(-1, 1, 1, 1)
                
                one_minus_ac = 1.0 - curr_ac

                # 2. 预计算 mu 的公共系数 (w1 对应 x0, w2 对应 xt)
                # 这两个系数在计算 mu_true 和 mu_pred 时完全一致，复用可节省计算量
                w1 = curr_ac_prev_sqrt * curr_b / one_minus_ac
                w2 = curr_a_sqrt * (1.0 - curr_ac_prev) / one_minus_ac
                
                
            
                # 4. 计算 Lvlb (方差学习损失)
                # 计算真实后验均值 mu_true
                mu_true = w1 * x_0 + w2 * noisy_x_t

                # 计算模型预测的均值 mu_pred (基于预测的噪声推导 x0)
                x0_pred_from_noise = (noisy_x_t - curr_sqrt_1m_ac * noise_pred) / curr_sqrt_ac.clamp(min=1e-8)
                mu_pred = w1 * x0_pred_from_noise + w2 * noisy_x_t

                # 计算 KL 散度
                v_clamped = torch.sigmoid(v_pred)
                log_variance = (
                    v_clamped * _log_b[timesteps].view(-1, 1, 1, 1)
                    + (1 - v_clamped) * _log_bt[timesteps].view(-1, 1, 1, 1)
                )
                log_var_true = torch.log(_bt[timesteps].clamp(min=1e-20)).view(-1, 1, 1, 1).expand_as(log_variance)
                log_var_diff = (log_var_true - log_variance).clamp(max=20) # 限制上限防止溢出
                kl = 0.5 * (
                    log_variance - log_var_true
                    + torch.exp(log_var_diff)
                    + (mu_true.detach() - mu_pred.detach()) ** 2 / torch.exp(log_variance).clamp(min=1e-8)
                    - 1
                )
                loss_vlb = kl.mean(dim=[1, 2, 3]).mean() / np.log(2.0)
                
                progress = epoch / Config.num_epochs

                # sigmoid schedule（最稳）
                lambda_vlb = 0.001 * torch.sigmoid(torch.tensor((progress - 0.5) * 10)).item()

                # clamp
                lambda_vlb = min(lambda_vlb, 0.001)

                # 3. 计算 Lsimple (噪声预测 MSE)
                snr = _snr[timesteps]
                mse_loss_weights = torch.stack([snr, 10 * torch.ones_like(snr)], dim=1).min(dim=1)[0] / snr
                loss_simple = F.mse_loss(noise_pred.float(), noise.float(), reduction="none")
                loss_simple = (loss_simple.mean(dim=[1, 2, 3]) * mse_loss_weights).mean()

                loss = loss_simple + loss_vlb * lambda_vlb

                # --- 修复A：在图像空间计算感知损失和边缘损失 ---
                # 计算预测的x0（去噪后的潜在空间）
                sqrt_alpha_cumprod = _sqrt_ac[timesteps].view(-1, 1, 1, 1)
                sqrt_one_minus_alpha = _sqrt_1m_ac[timesteps].view(-1, 1, 1, 1)
                
                # 预测的x0
                x0_pred = (noisy_x_t - sqrt_one_minus_alpha * noise_pred) / sqrt_alpha_cumprod.clamp(min=1e-3)
                x0_pred = x0_pred.clamp(-1, 1) # 必须增加这一行！
                # 真实的x0就是原始图像
                x0_true = x_0
                
                # 像素空间，直接使用x0_pred和x0_true，范围[-1,1] -> 转换为[0,1]
                decoded_pred = (x0_pred + 1) / 2  # 范围[0,1]
                decoded_pred = decoded_pred.clamp(0, 1)
                decoded_true = (x0_true + 1) / 2  # 范围[0,1]
                decoded_true = decoded_true.clamp(0, 1)
                
                # 图像已经是256x256，无需下采样，直接计算感知损失
                perceptual_loss_val = perceptual_loss_fn(decoded_pred, decoded_true)
                # 在图像空间计算感知损失
                
                # 在图像空间计算边缘损失
                pred_edge_img = laplacian_fn(decoded_pred)
                true_edge_img = laplacian_fn(decoded_true)
                edge_loss = F.mse_loss(pred_edge_img, true_edge_img)

                aux_loss = (
                    0.0001 * perceptual_loss_val +
                    0.001 * edge_loss
                ).mean()

                tloss = loss + aux_loss

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

            sub_pbar.set_postfix({"loss": f"{loss_val:.4f}", "edge_loss": f"{tedge_loss:.4f}", "perceptual_loss":f"{perceptual_loss_val_item:.4f}","lvb_loss":f"{loss_vlb:.4f}","lr": f"{current_lr:.2e}"})

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
                save_ldm_sample(accelerator, model, ema_model, epoch, config, label_proj)

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

                # 保存 label_proj 权重
                torch.save(label_proj.state_dict(), os.path.join(save_path, "label_proj.pt"))
                print(f"✅ 已安全保存 EMA 检查点并恢复训练权重，包括 label_proj")

    log_file.close()
    if accelerator.is_main_process:
        save_loss_plot(loss_history, config.output_dir)
        print("🚀 训练完成，Loss 曲线和权重已保存！")


def save_ldm_sample(accelerator, model, ema_model, epoch, config, label_proj):
    # 1. 切换为 DDPMScheduler，并开启 learned_range 模式
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule=config.beta_schedule,
        prediction_type="epsilon",
        variance_type="learned_range", # 关键：告诉 scheduler 我们有方差预测
        clip_sample=config.clip_sample
    )
    # 采样步数：DDPM 建议设多一点，如果为了快，可以用 DDIMScheduler (写法类似)
    scheduler.set_timesteps(200) 

    unwrapped_model = accelerator.unwrap_model(model)
    
    # 备份与准备
    orig_params = [p.detach().cpu().clone() for p in unwrapped_model.parameters()]
    ema_model.copy_to(unwrapped_model.parameters())
    unwrapped_model.eval()
    label_proj.eval()
    
    device = next(unwrapped_model.parameters()).device
    labels = torch.tensor([0, 1, 2, 3], device=device)
    x_t = torch.randn(4, 3, config.image_size, config.image_size, device=device)
    
    # 准备 CFG 嵌入
    uncond_labels = torch.full((4,), config.uncond_label, device=device)
    cond_emb = label_proj(labels)
    uncond_emb = label_proj(uncond_labels)
    
    # 采样循环
    for t in tqdm(scheduler.timesteps, desc=f"Epoch {epoch} Sampling"):
        with torch.no_grad():
            # 这里的输入是 4 个样本，我们做 CFG，所以扩展为 8
            latent_model_input = torch.cat([x_t] * 2)
            prompt_embeds = torch.cat([uncond_emb, cond_emb]) # 习惯上 uncond 在前
            timesteps = torch.full((8,), t, device=device, dtype=torch.long)
            
            # 模型一次性跑 8 个样本的预测 (输出 6 通道)
            model_output = unwrapped_model(latent_model_input, timesteps, encoder_hidden_states=prompt_embeds).sample
            
            # --- 核心：处理 6 通道 CFG ---
            # 拆分噪声部分和方差部分
            noise_pred_full, var_pred_full = model_output.chunk(2, dim=1)
            
            # 对噪声预测做 CFG 引导
            noise_uncond, noise_cond = noise_pred_full.chunk(2)
            noise_pred = noise_uncond + config.cfg * (noise_cond - noise_uncond)
            
            # 对方差预测做 CFG (方差通常不需要太强的引导，直接取 cond 或者也做微量引导)
            var_uncond, var_cond = var_pred_full.chunk(2)
            var_pred = var_cond # 采样时通常直接信任有条件分支的方差预测
            
            # 重新拼回 6 通道，交给 scheduler 处理
            final_output = torch.cat([noise_pred, var_pred], dim=1)
            
            # 2. 使用 DDPM step 计算。它内部会根据 6 通道自动计算真实方差
            x_t = scheduler.step(final_output, t, x_t).prev_sample

    # 结果后处理
    images = (x_t + 1) / 2
    images = images.clamp(0, 1)
    utils.save_image(images, f"{config.output_dir}/sample_epoch_{epoch}.png", nrow=2)
    
    # 权重恢复
    for p, orig_p in zip(unwrapped_model.parameters(), orig_params):
        p.data.copy_(orig_p.to(device))
    label_proj.train()
    print(f"采样完成：使用 DDPMScheduler (Learned Variance), CFG={config.cfg}")

if __name__ == "__main__":
    # Windows 上设置多进程启动方法为 spawn，确保 DataLoader 正常工作
    torch.multiprocessing.set_start_method('spawn', force=True)
    train_loop()
