import os
import shutil
import random
from tqdm import tqdm

def split_dataset(source_dir, target_dir, split_ratio=0.8):
    """
    source_dir: 原始文件夹 'fenlei'，包含子文件夹 1, 2, 3, 4
    target_dir: 目标文件夹 'datasets'
    split_ratio: 训练集比例
    """
    categories = ['1', '2', '3', '4']
    
    # 创建目标目录结构
    for phase in ['train', 'test']:
        for cat in categories:
            os.makedirs(os.path.join(target_dir, phase, cat), exist_ok=True)

    print(f"开始切分数据，目标比例：Train:{int(split_ratio*10)} / Test:{10-int(split_ratio*10)}")

    for cat in categories:
        cat_path = os.path.join(source_dir, cat)
        if not os.path.exists(cat_path):
            print(f"⚠️ 警告：找不到类别文件夹 {cat_path}，跳过...")
            continue
            
        # 获取该文件夹下所有图片
        all_images = [f for f in os.listdir(cat_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
        
        # 打乱顺序
        random.shuffle(all_images)
        
        # 计算切分位置
        split_point = int(len(all_images) * split_ratio)
        train_images = all_images[:split_point]
        test_images = all_images[split_point:]
        
        # 拷贝图片到对应文件夹
        print(f"正在处理类别 {cat}: 总计 {len(all_images)} 张...")
        
        # 拷贝训练集
        for img in tqdm(train_images, desc=f"  Category {cat} Train"):
            shutil.copy(os.path.join(cat_path, img), os.path.join(target_dir, 'train', cat, img))
            
        # 拷贝测试集
        for img in tqdm(test_images, desc=f"  Category {cat} Test "):
            shutil.copy(os.path.join(cat_path, img), os.path.join(target_dir, 'test', cat, img))

    print("\n✅ 数据切分完成！")
    print(f"数据已保存至: {os.path.abspath(target_dir)}")

if __name__ == "__main__":
    # 设置路径
    # 假设你的当前目录下有 fenlei 文件夹
    SRC = "D:/biye_sheji/fenlei" 
    DST = "./Base_datasets"
    
    # 执行切分
    split_dataset(SRC, DST, split_ratio=0.8)