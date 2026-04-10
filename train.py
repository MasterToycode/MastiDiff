import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from torchvision.datasets import ImageFolder
from diffusers import UNet2DModel, DDPMScheduler,DDIMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from tqdm.auto import tqdm
from diffusers.training_utils import EMAModel
from torch.utils.data import WeightedRandomSampler
import numpy as np

class CharbonnierLoss(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = torch.sqrt((x - y) ** 2 + self.eps ** 2)
        return torch.mean(diff)

# --- 1. 超参数配置 ---
class Config:
    data_root = "./datasets"       # 数据集根目录
    image_size = 256               # 图像分辨率
    train_batch_size = 6           # 根据显存调整
    num_epochs = 120               # 训练总轮数
    learning_rate = 5e-6           # 学习率
    lr_warmup_steps = 300          # 学习率预热步数
    save_image_epochs = 30          # 每隔几轮采样一次图像预览
    save_model_epochs = 30         # 每隔几轮保存一次模型
    mixed_precision = "fp16"       # 使用半精度加速 (需显卡支持)
    output_dir = "ddpm-udder-results7" # 输出目录
    beta_start = 0.0001 # 训练时的 beta_start
    beta_end = 0.02               
    beta_schedule = "scaled_linear"
    clip_sample = False
    

config = Config()
os.makedirs(config.output_dir, exist_ok=True)

# --- 2. 数据加载 ---

def get_data():
    preprocess = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1), # 轻微抖动，模拟不同光照下的医学图像
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),  # 归一化到 [-1, 1]
    ])
    
    # 1. 加载原始数据集
    train_dataset = ImageFolder(root=os.path.join(config.data_root, 'train'), transform=preprocess)
    
    # 2. 获取每个样本的类别标签 (targets 列表)
    targets = np.array(train_dataset.targets)
    
    # 3. 计算每个类别的样本总数
    # np.unique 会返回 [0, 1, 2, 3] (对应你的文件夹 1, 2, 3, 4)
    class_sample_count = np.array([len(np.where(targets == t)[0]) for t in np.unique(targets)])
    print(f"各类别样本数量: {class_sample_count}")
    
    # 4. 计算类别权重: 权重 = 1 / 样本数
    # 样本越少的类，权重越高
    class_weights = 1. / class_sample_count
    
    # 5. 为数据集中的每一个样本分配对应的权重
    samples_weight = torch.from_numpy(class_weights[targets]).double()
    
    # 6. 创建采样器
    # replacement=True 表示允许重复采样（这是过采样的核心）
    # num_samples 可以设为 len(samples_weight)，即维持一个 epoch 的总数不变
    sampler = WeightedRandomSampler(
        weights=samples_weight, 
        num_samples=len(samples_weight), 
        replacement=True
    )
    
    # 7. 返回 DataLoader
    # 注意：设置了 sampler 之后，shuffle 必须为 False (或者不写，默认为 False)
    return DataLoader(
        train_dataset, 
        batch_size=config.train_batch_size, 
        sampler=sampler,      # 引入权重采样
        shuffle=False,        # 必须设为 False
        pin_memory=True, 
        num_workers=4
    )

# --- 3. 模型定义 ---
# 这是一个带类别条件的 UNet。
# 注意：UNet2DModel 默认支持 class_labels (num_class_embeds)
model = UNet2DModel(
    sample_size=config.image_size,
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(128, 128, 256, 256, 512, 512),
    down_block_types=(
        "DownBlock2D", "DownBlock2D", "DownBlock2D", 
        "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"
    ),
    up_block_types=(
        "UpBlock2D", "AttnUpBlock2D", "UpBlock2D", 
        "UpBlock2D", "UpBlock2D", "UpBlock2D"
    ),
    num_class_embeds=4, # 关键：设置 4 个类别的嵌入
)

noise_scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    beta_start=config.beta_start,
    beta_end=config.beta_end, 
    beta_schedule=config.beta_schedule, # 这种曲线对高分辨率图像的细节学习更有利
    clip_sample=config.clip_sample, 
    prediction_type="epsilon"
)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
ema_model = EMAModel(
    model.parameters(), 
    decay=0.99,
    model_cls=UNet2DModel, # 必须加上这一行
    model_config=model.config # 建议加上这一行，保存时会自动带上模型配置
)

criterion = CharbonnierLoss()

# --- 4. 训练流程 ---
def train_loop():
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision, # 混合精度训练
        gradient_accumulation_steps=4, # 梯度累积步数
    )
    
    train_dataloader = get_data()
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )
    ema_model.to(accelerator.device)
    model_prepared, opt_prepared, dl_prepared, lr_prepared = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    # 新增：加载现有权重的逻辑
    initial_checkpoint = "ddpm-udder-results6/model_epoch_119.pth" # 填写加载的权重路径

    if os.path.exists(initial_checkpoint):
        print(f"正在加载预训练权重: {initial_checkpoint}")
        # 使用 accelerator.unwrap_model 是为了处理多卡/混合精度下的模型包装问题
        state_dict = torch.load(initial_checkpoint, map_location=accelerator.device)
        accelerator.unwrap_model(model_prepared).load_state_dict(state_dict)
        print("权重加载成功！将从该起点继续训练。")
    else:
        print("未找到预训练权重，将从头开始训练。")

    loss_history = []# 记录损失值

    global_step = 0
    for epoch in range(config.num_epochs):
        model_prepared.train() 
        progress_bar = tqdm(total=len(dl_prepared), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch+1}")

        for step, batch in enumerate(dl_prepared):
            # 1. 准备数据
            clean_images, class_labels = batch
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # 【新增修改】：添加全局偏移噪声
            # 这能强制模型学习更高的动态范围，让黑白对比更锐利，减少发灰发糊
            offset_noise = 0.1 * torch.randn(bs, 3, 1, 1).to(clean_images.device)
            noise = noise + offset_noise

            # 2. 采样随机时间步 t
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device
            ).long()

            # 3. 前向加噪
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            # --- 开启梯度累积块 ---
            with accelerator.accumulate(model_prepared):
                # 4. 模型预测
                noise_pred = model_prepared(noisy_images, timesteps, class_labels=class_labels).sample
                loss = F.mse_loss(noise_pred, noise)
        
                # 反向传播
                accelerator.backward(loss)

                # 5. 检查是否达到了累积步数（即是否要执行真正的参数更新）
                if accelerator.sync_gradients:
                    # 梯度裁剪：防止 Loss 剧烈跳动导致的梯度爆炸
                    accelerator.clip_grad_norm_(model_prepared.parameters(), 1.0) 
                    
                    opt_prepared.step()
                    lr_prepared.step()
                    opt_prepared.zero_grad()
                    
                    # 【重要修改】：只在梯度同步（参数更新）时更新 EMA
                    # 这样 EMA 记录的是每次稳定更新后的轨迹，而不是中间不完整的状态
                    ema_model.step(model_prepared.parameters())

            # 记录日志 (注意：每步都记录，或者只在同步时记录)
            current_loss = loss.detach().item()
            loss_history.append(current_loss)

            progress_bar.update(1)
            progress_bar.set_postfix(loss=current_loss, step=global_step)
            global_step += 1

        # --- 5. 周期性采样预览 ---
        if accelerator.is_main_process:
            if (epoch + 1) % config.save_image_epochs == 0:
                print(f"\n生成预览图 (Epoch {epoch}) - 使用 EMA 权重和 DDIM...")

                # --- [关键步骤 A]：备份并切换 ---
                # 使用 torch.no_grad() 确保备份过程不计算梯度
                with torch.no_grad():
                    orig_params = [p.clone().detach() for p in model_prepared.parameters()]
                    ema_model.copy_to(model_prepared.parameters())
                
                # 配置采样器（确保参数与训练代码中的 noise_scheduler 严格一致）
                eval_scheduler = DDIMScheduler(
                    num_train_timesteps=1000,
                    beta_start=config.beta_start,        # 确保这与你训练时的设置一致
                    beta_end=config.beta_end,               # 确保这与你训练时的设置一致
                    beta_schedule=config.beta_schedule, # 确保这与你训练时的设置一致
                    clip_sample=config.clip_sample,
                    prediction_type="epsilon"
                )
                eval_scheduler.set_timesteps(350) 

                sample_labels = torch.tensor([0, 1, 2, 3]).to(accelerator.device)
                sample_images = torch.randn(4, 3, config.image_size, config.image_size).to(accelerator.device)
                
                for t in tqdm(eval_scheduler.timesteps, desc="Sampling"):
                    with torch.no_grad():
                        model_output = model_prepared(sample_images, t, class_labels=sample_labels).sample
                    sample_images = eval_scheduler.step(model_output, t, sample_images).prev_sample
                
                # --- [关键步骤 B]：恢复原始权重 ---
                with torch.no_grad():
                    for p, orig_p in zip(model_prepared.parameters(), orig_params):
                        p.data.copy_(orig_p.data)
                    # 显式删除备份以释放显存
                    del orig_params 
                
                print("EMA 权重已恢复。")

                # 存盘
                sample_images = (sample_images / 2 + 0.5).clamp(0, 1)
                from torchvision import utils
                utils.save_image(sample_images, f"{config.output_dir}/sample_epoch_{epoch:03d}.png", nrow=2)

            # 保存模型权重 (EMA + 原始)
            if (epoch + 1) % config.save_model_epochs == 0:
                # 1. 保存原始权重（用于下次加载继续训练）
                torch.save(model_prepared.state_dict(), f"{config.output_dir}/model_epoch_{epoch:03d}.pth")
                
                # 2. 保存 EMA 权重（这是最终成品）
                # 注意：save_pretrained 是 diffusers 专有方法，它会创建一个文件夹存 json 和 bin
                ema_save_path = os.path.join(config.output_dir, f"ema_model_epoch_{epoch:03d}")
                ema_model.save_pretrained(ema_save_path)
    


    # 整个 Epoch 循环结束后 (跳出循环)，再执行绘图 ---
    if accelerator.is_main_process:
        import matplotlib.pyplot as plt
        import numpy as np

        print("训练已完成，正在生成最终 Loss 曲线...")
        
        plt.figure(figsize=(12, 6))
        # 绘制原始数据
        plt.plot(loss_history, color='blue', alpha=0.2, label='Batch Loss')
        
        # 绘制平滑曲线（为了美观和分析趋势）
        if len(loss_history) > 100:
            window_size = 100
            smooth_loss = np.convolve(loss_history, np.ones(window_size)/window_size, mode='valid')
            plt.plot(smooth_loss, color='red', linewidth=2, label='Smoothed Trend')
        
        plt.title(f"Final Training Loss Curve ({config.num_epochs} Epochs)")
        plt.xlabel("Total Steps")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 保存最终结果
        plt.savefig(os.path.join(config.output_dir, "final_loss_curve.png"))
        plt.close()
        
        # 保存原始数据，万一以后需要用 Excel 重新画图
        np.save(os.path.join(config.output_dir, "final_loss_data.npy"), np.array(loss_history))
        print(f"最终曲线已保存至: {config.output_dir}/final_loss_curve.png")


if __name__ == "__main__":
    train_loop()