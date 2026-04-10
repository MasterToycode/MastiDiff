import torch
import lpips
from PIL import Image
import torchvision.transforms as transforms
import os
import random
from tqdm import tqdm
import datetime
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 使用 VGG 作为后端的 LPIPS 评估器 (学术界最通用)
loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
SAVE_FILE = "FID_lpips.txt"             

def calculate_lpips_diversity(img_dir, num_pairs=2500):
    """
    计算文件夹内图片的多样性得分 (LPIPS)
    """
    files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg'))]
    if len(files) < 2: return 0
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    distances = []
    print(f"正在计算 {img_dir} 的多样性得分...")
    
    for _ in tqdm(range(num_pairs)):
        # 随机抽取两张不同的图片进行对比
        path1, path2 = random.sample(files, 2)
        
        img1 = transform(Image.open(path1).convert('RGB')).unsqueeze(0).to(device)
        img2 = transform(Image.open(path2).convert('RGB')).unsqueeze(0).to(device)

        with torch.no_grad():
            dist = loss_fn_vgg(img1, img2)
            distances.append(dist.item())

    avg_dist = sum(distances) / len(distances)
    return avg_dist

def calculate_lpips_for_all_classes(base_dir, num_pairs_per_class=1000):
    """
    计算整个数据集的LPIPS多样性得分（所有类别）
    """
    # 获取所有类别文件夹
    class_dirs = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            class_dirs.append(item_path)
    
    if not class_dirs:
        print(f"警告：在 {base_dir} 中没有找到类别文件夹")
        return None
    
    print(f"找到 {len(class_dirs)} 个类别文件夹")
    
    all_distances = []
    class_scores = {}
    
    for class_dir in class_dirs:
        class_name = os.path.basename(class_dir)
        print(f"\n处理类别: {class_name}")
        
        # 计算当前类别的LPIPS
        class_score = calculate_lpips_diversity(class_dir, num_pairs=num_pairs_per_class)
        class_scores[class_name] = class_score
        
        # 收集所有图片路径用于跨类别比较
        class_files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) 
                      if f.endswith(('.png', '.jpg'))]
        
        if len(class_files) >= 2:
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            
            # 跨类别比较：与其他类别的图片比较
            for other_class_dir in class_dirs:
                if other_class_dir == class_dir:
                    continue
                    
                other_class_name = os.path.basename(other_class_dir)
                other_files = [os.path.join(other_class_dir, f) for f in os.listdir(other_class_dir) 
                             if f.endswith(('.png', '.jpg'))]
                
                if not other_files:
                    continue
                
                # 随机选择一些跨类别对
                num_cross_pairs = min(200, len(class_files) * len(other_files) // 10)
                num_cross_pairs = max(50, num_cross_pairs)  # 至少50对
                
                for _ in range(num_cross_pairs):
                    img1_path = random.choice(class_files)
                    img2_path = random.choice(other_files)
                    
                    img1 = transform(Image.open(img1_path).convert('RGB')).unsqueeze(0).to(device)
                    img2 = transform(Image.open(img2_path).convert('RGB')).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        dist = loss_fn_vgg(img1, img2)
                        all_distances.append(dist.item())
    
    # 计算整体平均LPIPS
    if all_distances:
        overall_score = sum(all_distances) / len(all_distances)
    else:
        overall_score = 0
    
    return {
        'overall_score': overall_score,
        'class_scores': class_scores,
        'num_classes': len(class_dirs),
        'total_pairs': len(all_distances)
    }


if __name__ == "__main__":
    # 计算整个数据集的LPIPS
    base_dir = r"Base_datasets_augmented_3\train"
    results = calculate_lpips_for_all_classes(base_dir, num_pairs_per_class=1000)
    
    if results:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(SAVE_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n" + "="*60 + "\n")
            f.write(f"测试时间: {timestamp}\n")
            f.write(f"评估数据集: {base_dir}\n")
            f.write(f"类别数量: {results['num_classes']}\n")
            f.write(f"总比较对数: {results['total_pairs']}\n")
            f.write(f"🎯 整体 LPIPS 多样性得分: {results['overall_score']:.4f}\n")
            f.write("\n各类别得分:\n")
            for class_name, score in results['class_scores'].items():
                f.write(f"  类别 {class_name}: {score:.4f}\n")
            f.write("="*60 + "\n")
        
        print(f"\n✅ 计算完成！结果已追加至: {SAVE_FILE}")
        print(f"\n🌟 整体平均 LPIPS 多样性得分: {results['overall_score']:.4f}")
        print(f"📊 各类别得分:")
        for class_name, score in results['class_scores'].items():
            print(f"   类别 {class_name}: {score:.4f}")
    else:
        print("❌ 计算失败，请检查数据集路径")