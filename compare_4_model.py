import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
import numpy as np

# ==========================================
# 1. 配置类：增加训练策略参数
# ==========================================
class Config:
    exp_phase = "Augmented_Dataset_ddpm_variance_V2"  
    root_dir = f"Classification_Experiments/{exp_phase}"
    train_dir = "./ddpm_augmented_v1/train" 
    test_dir = "./Base_datasets/test"
    
    model_list = ["resnet18", "swin_t", "vit_tiny", "convnext_tiny"] 
    img_size = 224
    
    batch_size = 32
    num_workers = 4      
    epochs = 50                        
    base_lr = 1e-4                     # 分类头学习率
    backbone_lr_mult = 0.2             # 主干网络学习率倍率 
    weight_decay = 0.05                # 适当的 AdamW 权重衰减
    patience = 12                      
    
    # 数据增强/防过拟合
    mixup_alpha = 0.2                  
    label_smoothing = 0.1              
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. Mixup 实现 (防止过拟合核心)
# ==========================================
def mixup_data(x, y, alpha=1.0):
    '''返回混合后的输入、两份标签对以及混合比例 lam'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(Config.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ==========================================
# 3. 模型获取与“语义化”冻结逻辑
# ==========================================
def get_model(name, freeze_level=1):
    """
    freeze_level: 
    0: 全微调 (Fine-tuning)
    1: 冻结浅层 (Frozen Stages)
    2: 仅训练分类头 (Linear Probe)
    """
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.fc.in_features, 4))
        if freeze_level >= 1:
            # 冻结：conv1, bn1, layer1, layer2 (保留深层 layer3/4 训练)
            for child in [m.conv1, m.bn1, m.layer1, m.layer2]:
                for param in child.parameters(): param.requires_grad = False
        if freeze_level >= 2:
            for child in [m.layer3, m.layer4]:
                for param in child.parameters(): param.requires_grad = False

    elif name == "swin_t":
        m = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
        m.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.head.in_features, 4))
        if freeze_level >= 1:
            # Swin 包含 features[0]~[7]，前4个是早期 Stage
            for i in range(4):
                for param in m.features[i].parameters(): param.requires_grad = False
                
    elif name == "vit_tiny": # torchvision 里的 vit_b_16
        m = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        m.heads.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.heads.head.in_features, 4))
        if freeze_level >= 1:
            # 冻结前 6 个 Transformer Encoder Block
            for param in m.conv_proj.parameters(): param.requires_grad = False
            for i in range(6):
                for param in m.encoder.layers[i].parameters(): param.requires_grad = False

    elif name == "convnext_tiny":
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        m.classifier[2] = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.classifier[2].in_features, 4))
        if freeze_level >= 1:
            # 冻结前一半的 Stage
            for i in range(4):
                for param in m.features[i].parameters(): param.requires_grad = False

    return m.to(Config.device)

# ==========================================
# 4. 主程序
# ==========================================
if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.fp32_precision = 'tf32'
        torch.backends.cudnn.benchmark = True
        print(f"✅ 硬件加速已激活")

    for sub_dir in ["models", "results"]:
        os.makedirs(os.path.join(Config.root_dir, sub_dir), exist_ok=True)

    # 升级数据增强：使用 RandAugment
    train_transform = transforms.Compose([
        transforms.Resize((Config.img_size, Config.img_size)),
        transforms.RandAugment(num_ops=2, magnitude=9), # 自动增强
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2), 
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((Config.img_size, Config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_set = datasets.ImageFolder(Config.train_dir, train_transform)
    test_set = datasets.ImageFolder(Config.test_dir, test_transform)
    train_loader = DataLoader(train_set, batch_size=Config.batch_size, shuffle=True, num_workers=Config.num_workers,persistent_workers=True, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=Config.batch_size, shuffle=False, num_workers=Config.num_workers, persistent_workers=True,pin_memory=True)

    scaler = torch.amp.GradScaler('cuda') 

    for model_name in Config.model_list:
        print(f"\n🚀 模型训练开始: {model_name}")
        model = get_model(model_name, freeze_level=1)
        
        # 差异化学习率设置
        backbone_params = []
        head_params = []
        for n, p in model.named_parameters():
            if p.requires_grad:
                if any(x in n for x in ['fc', 'head', 'classifier']):
                    head_params.append(p)
                else:
                    backbone_params.append(p)

        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': Config.base_lr * Config.backbone_lr_mult},
            {'params': head_params, 'lr': Config.base_lr}
        ], weight_decay=Config.weight_decay)
        
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=1)
        criterion = nn.CrossEntropyLoss(label_smoothing=Config.label_smoothing)
        
        best_acc, patience_counter, history = 0.0, 0, []

        for epoch in range(Config.epochs):
            model.train()
            start_time = time.time()
            running_loss, correct, total = 0.0, 0, 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(Config.device), labels.to(Config.device)
                
                # 应用 Mixup
                inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, Config.mixup_alpha)
                
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 梯度裁剪
                scaler.step(optimizer)
                scaler.update()
                
                running_loss += loss.item()
                # 计算训练准确率（Mixup下以主标签为准，仅作参考）
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (lam * predicted.eq(labels_a).sum().item() + (1 - lam) * predicted.eq(labels_b).sum().item())

            # 验证阶段
            model.eval()
            val_loss, val_correct = 0.0, 0
            with torch.inference_mode():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(Config.device), labels.to(Config.device)
                    outputs = model(inputs)
                    val_loss += criterion(outputs, labels).item()
                    val_correct += (outputs.argmax(1) == labels).sum().item()
            
            epoch_train_acc = correct / total
            epoch_val_acc = val_correct / len(test_set)
            scheduler.step()
            
            # 早停与保存
            if epoch_val_acc > best_acc:
                best_acc = epoch_val_acc
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(Config.root_dir, "models", f"best_{model_name}.pth"))
                print(f"⭐ Best Model Saved! Acc: {best_acc:.4f}")
            else:
                patience_counter += 1

            print(f"[{model_name}] Epoch {epoch+1:02d} | Train Acc: {epoch_train_acc:.4f} | Val Acc: {epoch_val_acc:.4f} | LR: {optimizer.param_groups[1]['lr']:.6f} | Time: {time.time()-start_time:.1f}s")
            
            history.append([epoch+1, running_loss/len(train_loader), epoch_train_acc, val_loss/len(test_loader), epoch_val_acc])
            if patience_counter >= Config.patience:
                print(f"🛑 Early Stopping triggered.")
                break

        # 保存 CSV 和 曲线图
        df = pd.DataFrame(history, columns=['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc'])
        df.to_csv(os.path.join(Config.root_dir, "results", f"{model_name}_metrics.csv"), index=False)

        plt.figure(figsize=(10, 5))
        plt.plot(df['epoch'], df['train_acc'], 'b-', label='Train')
        plt.plot(df['epoch'], df['test_acc'], 'r-', label='Test')
        plt.title(f"{model_name} Learning Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(Config.root_dir, "results", f"{model_name}_curves.png"))
        plt.close()

        del model, optimizer
        torch.cuda.empty_cache()

    print("🎉 All Experiments Completed!")