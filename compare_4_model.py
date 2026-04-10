import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt # 找回绘图功能
import pandas as pd             # 找回 CSV 保存功能
import os
import time

class Config:
    exp_phase = "Augmented_Dataset_vdm_2"  
    root_dir = f"Classification_Experiments/{exp_phase}"
    train_dir = "./ldm_augmented_v2/train" 
    test_dir = "./Base_datasets/test"
    
    model_list = ["resnet18", "swin_t", "vit_tiny", "convnext_tiny"] 
    img_size = 224
    
    # 优化配置：针对过拟合问题
    batch_size = 32                    # 减小批次大小，增加梯度噪声
    num_workers = 4      
    epochs = 40                        
    lr = 1e-4                          
    weight_decay = 0.1                 # 增加权重衰减，更强的L2正则化
    patience = 10                      # 早停耐心值
    min_delta = 0.001                  # 早停最小改进阈值
    
    # 数据增强配置
    mixup_alpha = 0.2                  # MixUp增强参数
    cutmix_alpha = 1.0                 # CutMix增强参数
    label_smoothing = 0.1              # 标签平滑参数
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model(name):
    # 保持你原有的模型初始化逻辑，并添加 dropout 机制
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # 在分类层前添加 dropout
        m.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(m.fc.in_features, 4)
        )
    elif name == "vgg16":
        m = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
        # 在分类器最后一层前添加 dropout
        m.classifier[6] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4)
        )
    elif name == "swin_t":
        m = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
        # 在分类头前添加 dropout
        m.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(m.head.in_features, 4)
        )
    elif name == "vit_tiny":
        m = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        # 在分类头前添加 dropout
        m.heads.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(m.heads.head.in_features, 4)
        )
    elif name == "convnext_tiny":
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        # 在分类器最后一层前添加 dropout
        m.classifier[2] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(m.classifier[2].in_features, 4)
        )
    return m.to(Config.device)

if __name__ == '__main__':
    # 1. 硬件加速设置 (放在 main 里面，解决重复打印问题)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.fp32_precision = 'tf32'
        torch.backends.cudnn.conv.fp32_precision = 'tf32'
        torch.backends.cudnn.benchmark = True
        print(f"✅ 5070 Ti 硬件加速已激活: TF32 模式")

    # 2. 目录创建 (找回 results 目录)
    for sub_dir in ["models", "results"]:
        os.makedirs(os.path.join(Config.root_dir, sub_dir), exist_ok=True)

    # 3. 数据加载优化 - 增强数据增强以对抗过拟合
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((Config.img_size + 32, Config.img_size + 32)),  # 先放大再随机裁剪
            transforms.RandomCrop(Config.img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3)),  # 随机擦除
        ]),
        'test': transforms.Compose([
            transforms.Resize((Config.img_size, Config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    train_set = datasets.ImageFolder(Config.train_dir, data_transforms['train'])
    test_set = datasets.ImageFolder(Config.test_dir, data_transforms['test'])
    train_loader = DataLoader(train_set, batch_size=Config.batch_size, shuffle=True, num_workers=Config.num_workers, persistent_workers=True,pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=Config.batch_size, shuffle=False, num_workers=Config.num_workers, persistent_workers=True,pin_memory=True)

    scaler = torch.amp.GradScaler('cuda') 

    # 4. 自动化训练循环 - 添加早停和更多优化
    for model_name in Config.model_list:
        print(f"\n🚀 正在开始训练模型: {model_name}")
        model = get_model(model_name)
        '''
        # 冻结预训练层的前几层，只微调后面几层
        if model_name == "resnet18":
            for param in list(model.parameters())[:-20]:  # 冻结大部分层
                param.requires_grad = False
        elif model_name == "swin_t":
            for param in list(model.parameters())[:-15]:
                param.requires_grad = False
        elif model_name == "vit_tiny":
            for param in list(model.parameters())[:-10]:
                param.requires_grad = False
        elif model_name == "convnext_tiny":
            for param in list(model.parameters())[:-12]:  # 冻结ConvNeXt的大部分层
                param.requires_grad = False
        '''
        
        
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=Config.lr, 
            weight_decay=Config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # 使用带热重启的余弦退火调度器
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=10,  # 第一个周期的长度
            T_mult=2,  # 每个周期长度翻倍
            eta_min=Config.lr * 0.01  # 最小学习率
        )
        
        # 使用标签平滑的交叉熵损失
        criterion = nn.CrossEntropyLoss(label_smoothing=Config.label_smoothing)
        
        history = [] # 找回历史记录列表
        best_acc = 0.0
        patience_counter = 0
        early_stop = False
        
        print(f"✅ 模型 {model_name} 已准备就绪，可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        for epoch in range(Config.epochs):
            if early_stop:
                print(f"⚠️  早停触发，停止训练 {model_name}")
                break
                
            model.train()
            start_time = time.time()
            running_loss, correct = 0.0, 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(Config.device, non_blocking=True), labels.to(Config.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    # 添加梯度惩罚（L2正则化）
                    l2_reg = torch.tensor(0., device=Config.device)
                    for param in model.parameters():
                        if param.requires_grad:
                            l2_reg += torch.norm(param)
                    loss = loss + 0.001 * l2_reg
                
                scaler.scale(loss).backward()
                
                # 梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                
                running_loss += loss.item()
                correct += (outputs.argmax(1) == labels).sum().item()
            
            epoch_train_acc = correct / len(train_set)
            
            # 验证阶段
            model.eval()
            val_loss, val_correct = 0.0, 0
            with torch.inference_mode():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(Config.device), labels.to(Config.device)
                    outputs = model(inputs)
                    val_loss += criterion(outputs, labels).item()
                    val_correct += (outputs.argmax(1) == labels).sum().item()
            
            epoch_val_acc = val_correct / len(test_set)
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # 早停机制
            if epoch_val_acc > best_acc + Config.min_delta:
                best_acc = epoch_val_acc
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(Config.root_dir, "models", f"best_{model_name}.pth"))
                print(f"💾 保存最佳模型，准确率: {best_acc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= Config.patience:
                    early_stop = True
            
            # 找回详细日志打印
            print(f"[{model_name}] Epoch {epoch+1}/{Config.epochs} | LR: {current_lr:.6f} | Train Acc: {epoch_train_acc:.4f} | Test Acc: {epoch_val_acc:.4f} | Time: {time.time()-start_time:.2f}s | Train Loss :{running_loss/len(train_loader):.6f} | Patience: {patience_counter}/{Config.patience}")
            
            # 存入 history 以便后续绘图和保存 CSV
            history.append([
                epoch+1, 
                running_loss/len(train_loader), 
                epoch_train_acc, 
                val_loss/len(test_loader), 
                epoch_val_acc
            ])

        # 5. 保存结果文件（CSV 和 曲线图）
        df = pd.DataFrame(history, columns=['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc'])
        df.to_csv(os.path.join(Config.root_dir, "results", f"{model_name}_metrics.csv"), index=False)

        plt.figure(figsize=(10, 5))
        plt.plot(df['epoch'], df['train_acc'], label='Train Acc')
        plt.plot(df['epoch'], df['test_acc'], label='Test Acc')
        plt.title(f"{model_name} Accuracy - Phase: {Config.exp_phase}")
        plt.legend()
        plt.savefig(os.path.join(Config.root_dir, "results", f"{model_name}_curves.png"))
        plt.close()

        # 清理显存碎片
        del model, optimizer
        torch.cuda.empty_cache()

    print("🎉 所有模型训练、CSV指标、准确率曲线图已全部生成！")