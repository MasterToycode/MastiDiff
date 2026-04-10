import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from diffusers import UNet2DConditionModel
from train_ddpm_variance import LabelEmbedding

# ===============================
# 配置
# ===============================

checkpoint_dir = r"ddpm_variance_10\checkpoint_epoch_99"

num_classes = 4
uncond_label = 4

device = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================
# 加载模型
# ===============================

model = UNet2DConditionModel.from_pretrained(checkpoint_dir).to(device).eval()

label_proj = LabelEmbedding(num_classes + 1).to(device).eval()

try:
    label_proj.load_state_dict(
        torch.load(f"{checkpoint_dir}/label_proj.pt", map_location=device)
    )
    print("✅ label_proj loaded")
except:
    print("⚠️ label_proj 未找到权重，使用随机初始化")


# ===============================
# 提取 embedding
# ===============================

with torch.no_grad():

    labels = torch.arange(num_classes + 1).to(device)

    tokens = label_proj(labels)   # (C, 8, 512)

    emb_mean = tokens.mean(dim=1)  # (C,512)

    emb_flat = tokens.view(num_classes + 1, -1)  # (C,4096)

emb_mean = emb_mean.cpu().numpy()
emb_flat = emb_flat.cpu().numpy()

# ===============================
# 1 t-SNE
# ===============================

print("Running t-SNE...")

tsne = TSNE(n_components=2, perplexity=3, random_state=42)
emb_2d = tsne.fit_transform(emb_mean)

plt.figure(figsize=(6,6))
plt.scatter(emb_2d[:,0], emb_2d[:,1],
            c=np.arange(num_classes+1),
            cmap="tab10",
            s=150)

for i in range(num_classes+1):
    plt.text(emb_2d[i,0], emb_2d[i,1], f"class {i}")

plt.title("Label Embedding t-SNE")
plt.show()


# ===============================
# 2 距离矩阵
# ===============================

print("Embedding distance matrix:")

emb_torch = torch.tensor(emb_flat)

dist = torch.cdist(emb_torch, emb_torch)

print(dist)

plt.imshow(dist.numpy())
plt.colorbar()
plt.title("Embedding Distance Matrix")
plt.show()


# ===============================
# 3 Cosine similarity
# ===============================

cos = cosine_similarity(emb_flat)

print("Cosine similarity:")

print(cos)

plt.imshow(cos)
plt.colorbar()
plt.title("Cosine Similarity Matrix")
plt.show()


# ===============================
# 4 Token 结构
# ===============================

tokens_np = tokens.cpu().numpy()

plt.figure(figsize=(8,4))

for c in range(num_classes+1):

    token_norm = np.linalg.norm(tokens_np[c], axis=1)

    plt.plot(token_norm, label=f"class {c}")

plt.legend()
plt.title("Token Norm Distribution")
plt.xlabel("Token Index")
plt.ylabel("Norm")
plt.show()


# ===============================
# 5 PCA
# ===============================

pca = PCA(n_components=2)

pca_2d = pca.fit_transform(emb_flat)

plt.figure(figsize=(6,6))

plt.scatter(pca_2d[:,0], pca_2d[:,1],
            c=np.arange(num_classes+1),
            cmap="tab10",
            s=150)

for i in range(num_classes+1):
    plt.text(pca_2d[i,0], pca_2d[i,1], f"class {i}")

plt.title("Label Embedding PCA")
plt.show()


print("Explained variance ratio:", pca.explained_variance_ratio_)