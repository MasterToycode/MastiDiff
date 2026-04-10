import os
import torch
import torch.nn as nn
from torchvision import utils
from diffusers import UNet2DConditionModel, DPMSolverMultistepScheduler

# --- 1. 必须保留的标签投影类 (需与训练代码一致) ---
class LabelEmbedding(nn.Module):
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
        # 采样时务必进入 eval 模式以跳过 dropout 和 手动加的随机噪声
        tokens = self.mlp(x).view(-1, self.num_tokens, self.embedding_dim)
        return tokens + self.pos

# --- 2. 采样配置 ---
CHECKPOINT_PATH = r"ddpm_variance_8\checkpoint_epoch_19" # 替换为你最新的权重文件夹路径
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CFG_SCALE = 1.5  # 如果之前图片“炸了”，尝试调低到 1.0 或 1.5
IMAGE_SIZE = 256
NUM_CLASSES = 4
UNCOND_LABEL = 4  # 训练时设定的无条件标签

@torch.no_grad()
def run_test_sample():
    # 1. 加载模型
    print(f"正在从 {CHECKPOINT_PATH} 加载模型...")
    model = UNet2DConditionModel.from_pretrained(CHECKPOINT_PATH).to(DEVICE)
    model.eval()

    # 2. 加载标签投影层
    label_proj = LabelEmbedding(NUM_CLASSES + 1, embedding_dim=256).to(DEVICE)
    label_proj.load_state_dict(torch.load(os.path.join(CHECKPOINT_PATH, "label_proj.pt"), map_location=DEVICE))
    label_proj.eval()

    # 3. 配置采样器 (DPM-Solver 速度快且稳定)
    # 注意：训练时用 squaredcos_cap_v2，采样器也必须匹配
    scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        beta_schedule="squaredcos_cap_v2", 
        algorithm_type="dpmsolver++"
    )
    scheduler.set_timesteps(50) # 50步采样

    # 4. 准备条件 (每个类别生成一张图)
    classes = torch.arange(NUM_CLASSES, device=DEVICE)
    cond_emb = label_proj(classes)
    uncond_labels = torch.full((NUM_CLASSES,), UNCOND_LABEL, device=DEVICE)
    uncond_emb = label_proj(uncond_labels)

    # 5. 生成初始噪声
    x_t = torch.randn(NUM_CLASSES, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)

    # 6. 采样循环
    for t in scheduler.timesteps:
        # 扩展输入以支持 CFG
        latent_model_input = torch.cat([x_t] * 2)
        # 注意：这里拼接了 cond 和 uncond
        combined_emb = torch.cat([cond_emb, uncond_emb])
        t_input = torch.full((NUM_CLASSES * 2,), t, device=DEVICE, dtype=torch.long)

        # 模型预测 (输出6通道，只取前3通道噪声)
        output = model(latent_model_input, t_input, encoder_hidden_states=combined_emb).sample
        eps_cond, eps_uncond = output[:NUM_CLASSES, :3], output[NUM_CLASSES:, :3]
        
        # 应用 CFG
        noise_pred = eps_uncond + CFG_SCALE * (eps_cond - eps_uncond)

        # 步进
        x_t = scheduler.step(noise_pred, t, x_t).prev_sample

    # 7. 后处理并保存
    images = ((x_t + 1) / 2).clamp(0, 1)
    save_name = f"test_cfg1_{CFG_SCALE}.png"
    utils.save_image(images, save_name, nrow=2)
    print(f"测试图已保存至: {save_name}")

if __name__ == "__main__":
    run_test_sample()