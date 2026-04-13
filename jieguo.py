import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ================= 1. 核心配置 (请确保路径正确) =================
class Config:
    # 实验配置 - 可以根据需要修改这些值
    base_experiment = "Base_Dataset_2"          # 原始数据集实验
    aug_experiment = "Augmented_Dataset_ddpm_variance_V2"    # 数据增强实验
    save_dir = "Classification_Experiments/Final_Analysis_Report_ddpm_variance_V2" 
    
    # 模型配置 - 与 compare_4_model.py 保持一致
    model_ids = ["resnet18", "swin_t", "vit_tiny", "convnext_tiny"]
    model_names = ["ResNet-18", "Swin-T", "ViT-Tiny", "ConvNeXt-Tiny"]
    
    # 测试集路径
    test_dataset_path = "./new_base_datasets/test"
    
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ['1', '2', '3', '4']

# 自动生成路径
BASE_RESULTS = f"Classification_Experiments/{Config.base_experiment}/results"
AUG_RESULTS = f"Classification_Experiments/{Config.aug_experiment}/results"
BASE_MODELS_DIR = f"Classification_Experiments/{Config.base_experiment}/models"
AUG_MODELS_DIR = f"Classification_Experiments/{Config.aug_experiment}/models"
SAVE_DIR = Config.save_dir

os.makedirs(SAVE_DIR, exist_ok=True)

device = Config.device
class_names = Config.class_names
model_ids = Config.model_ids
model_names = Config.model_names

def get_model(name):
    # 保持你原有的模型初始化逻辑，并添加 dropout 机制
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # 在分类层前添加 dropout
        m.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(m.fc.in_features, 4)
        )
    elif name == "vgg16":
        m = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
        # 在分类器最后一层前添加 dropout
        m.classifier[6] = nn.Sequential(
            nn.Dropout(0.4),
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
            nn.Dropout(0.4),
            nn.Linear(m.heads.head.in_features, 4)
        )
    elif name == "convnext_tiny":
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        # 在分类器最后一层前添加 dropout
        m.classifier[2] = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(m.classifier[2].in_features, 4)
        )
    return m.to(Config.device)

# ================= 3. 数据分析与曲线绘制 =================
def run_full_analysis():
    summary_list = []
    print("🚀 正在读取 CSV 指标并生成对比曲线...")

    for m_id, m_name in zip(model_ids, model_names):
        base_csv = os.path.join(BASE_RESULTS, f"{m_id}_metrics.csv")
        aug_csv = os.path.join(AUG_RESULTS, f"{m_id}_metrics.csv")

        if not os.path.exists(base_csv) or not os.path.exists(aug_csv):
            print(f"⚠️ 跳过 {m_name}: 找不到 CSV 文件。")
            continue

        df_b = pd.read_csv(base_csv)
        df_a = pd.read_csv(aug_csv)

        # 绘制 Loss/Acc 对比图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'Comparison: {m_name}', fontsize=16, fontweight='bold')

        ax1.plot(df_b['epoch'], df_b['test_loss'], 'r--', label='Base Test Loss')
        ax1.plot(df_a['epoch'], df_a['test_loss'], 'g-', label='Augmented Test Loss')
        ax1.set_title('Loss'); ax1.legend()

        ax2.plot(df_b['epoch'], df_b['test_acc']*100, 'r--', label='Base Test Acc')
        ax2.plot(df_a['epoch'], df_a['test_acc']*100, 'g-', label='Augmented Test Acc')
        ax2.set_title('Accuracy (%)'); ax2.legend()

        plt.savefig(os.path.join(SAVE_DIR, f"{m_id}_comparison.png"))
        plt.close()

        # 记录最高分用于柱状图
        summary_list.append({
            "Model": m_name,
            "Base": round(df_b['test_acc'].max() * 100, 2),
            "Augmented": round(df_a['test_acc'].max() * 100, 2)
        })
        


    # 生成最终汇总表
    if summary_list:
        summary_df = pd.DataFrame(summary_list)
        summary_df.to_csv(os.path.join(SAVE_DIR, "Performance_Table.csv"), index=False)
        print(f"📊 汇总表已保存至: {SAVE_DIR}/Performance_Table.csv")

        # --- 新增：保存对比柱状图 ---
        plt.figure(figsize=(12, 7))
        x = np.arange(len(summary_df['Model']))
        width = 0.35

        # 绘制两条柱子：原始数据 vs 扩充数据
        rects1 = plt.bar(x - width/2, summary_df['Base'], width, label='Original Dataset', color='#C0C4C8')
        rects2 = plt.bar(x + width/2, summary_df['Augmented'], width, label='DDPM_VARIANCE Augmented', color='#2E86C1')

        # 添加装饰
        plt.ylabel('Best Test Accuracy (%)', fontsize=12)
        plt.title('Final Performance Comparison: Impact of Data Augmentation', fontsize=14, pad=20)
        plt.xticks(x, summary_df['Model'], fontsize=11)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.ylim(0, 110) # 留出空间显示百分比标签

        # 在柱子上方标注具体的数值
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                plt.annotate(f'{height}%',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3点纵向偏移
                            textcoords="offset points",
                            ha='center', va='bottom', fontweight='bold')

        autolabel(rects1)
        autolabel(rects2)

        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, "Final_Bar_Comparison.png"), dpi=300)
        plt.close()
        print(f"📊 汇总对比柱状图已保存至: {SAVE_DIR}/Final_Bar_Comparison.png")

# ================= 4. 混淆矩阵生成 (推理模式) =================
def plot_all_cms():
    print("🧠 正在生成混淆矩阵 (仅推理)...")
    
    # 准备测试集加载器
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # 使用配置中的测试集路径
    test_set = datasets.ImageFolder(Config.test_dataset_path, test_transform)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    for m_id, m_name in zip(model_ids, model_names):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        for idx, (dir_path, title) in enumerate([(BASE_MODELS_DIR, "Base"), (AUG_MODELS_DIR, "Aug")]):
            weight_path = os.path.join(dir_path, f"best_{m_id}.pth")
            ax = ax1 if idx == 0 else ax2
            
            if not os.path.exists(weight_path):
                continue
            
            # 加载权重
            model = get_model(m_id)
            model.load_state_dict(torch.load(weight_path, map_location=device))
            model.eval()

            all_preds, all_labels = [], []
            with torch.no_grad():
                for imgs, lbls in test_loader:
                    out = model(imgs.to(device))
                    all_preds.extend(out.argmax(1).cpu().numpy())
                    all_labels.extend(lbls.numpy())

            cm = confusion_matrix(all_labels, all_preds)
            sns.heatmap(cm / cm.sum(axis=1)[:, None], annot=True, fmt='.2%', cmap='Blues', ax=ax,
                        xticklabels=class_names, yticklabels=class_names, cbar=False)
            ax.set_title(f"{m_name} ({title})")

        plt.savefig(os.path.join(SAVE_DIR, f"{m_id}_confusion_matrix.png"))
        plt.close()
        print(f"✅ {m_name} 混淆矩阵完成。")

if __name__ == "__main__":
    run_full_analysis()
    plot_all_cms()
    print(f"\n✨ 全部结果已存入: {SAVE_DIR}")