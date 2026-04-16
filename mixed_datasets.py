import os
import shutil
import random
from tqdm import tqdm

class MixConfig:
    base_dir = "./Base_datasets/train"
    ldm_dir = "./ldm_augmented_v2/train"
    ddpm_var_dir = "./ddpm_augmented_v1/train"
    
    target_total = 5000  # 强制目标总数
    alpha = 0.8          # LDM 在生成图中占的比例
    
    output_dir = "./hybrid_final_5000_0.8/train"
    categories = ['1', '2', '3', '4']
    seed = 42

random.seed(MixConfig.seed)

def run_exact_mix():
    if os.path.exists(MixConfig.output_dir):
        print(f"⚠️ 警告: 目标目录 {MixConfig.output_dir} 已存在，请清理后再运行以防干扰。")
    
    os.makedirs(MixConfig.output_dir, exist_ok=True)
    
    for cat in MixConfig.categories:
        print(f"\n📏 正在校准类别 {cat} 的总数...")
        target_cat_dir = os.path.join(MixConfig.output_dir, cat)
        os.makedirs(target_cat_dir, exist_ok=True)

        # 1. 拷贝原图
        base_cat_path = os.path.join(MixConfig.base_dir, cat)
        base_imgs = [f for f in os.listdir(base_cat_path) if not f.startswith("gen_")]
        for img in base_imgs:
            shutil.copy2(os.path.join(base_cat_path, img), target_cat_dir)
        
        n_base = len(base_imgs)
        n_needed = MixConfig.target_total - n_base # 核心：计算总缺口
        
        print(f"   基础原图: {n_base} | 需要补齐: {n_needed}")

        if n_needed <= 0:
            print(f"   ✅ 原图已超过或达到 5000，无需补齐。")
            continue

        # 2. 按照比例分配 LDM 和 DDPM-Var 的配额
        n_ldm = int(n_needed * MixConfig.alpha)
        n_ddpm = n_needed - n_ldm # 通过减法强制总数等于 n_needed

        # 3. 准备生成图池子
        ldm_pool = [f for f in os.listdir(os.path.join(MixConfig.ldm_dir, cat)) if f.startswith("gen_")]
        ddpm_pool = [f for f in os.listdir(os.path.join(MixConfig.ddpm_var_dir, cat)) if f.startswith("gen_")]

        # 4. 随机抽样
        sampled_ldm = random.sample(ldm_pool, min(len(ldm_pool), n_ldm))
        sampled_ddpm = random.sample(ddpm_pool, min(len(ddpm_pool), n_ddpm))

        # 5. 最终拷贝
        for f in sampled_ldm:
            shutil.copy2(os.path.join(MixConfig.ldm_dir, cat, f), 
                         os.path.join(target_cat_dir, f"mix_L_{f}"))
        
        for f in sampled_ddpm:
            shutil.copy2(os.path.join(MixConfig.ddpm_var_dir, cat, f), 
                         os.path.join(target_cat_dir, f"mix_D_{f}"))

        # 6. 最终数量校验
        final_count = len(os.listdir(target_cat_dir))
        if final_count == MixConfig.target_total:
            print(f"   🎯 校验通过: 最终总数精确等于 {final_count}")
        else:
            print(f"   ❌ 校验失败: 当前总数 {final_count}，请检查池子样本量是否充足。")

if __name__ == "__main__":
    run_exact_mix()