import numpy as np
import matplotlib.pyplot as plt
import torch

from evaluation.metrics import format_class_label

@torch.no_grad()
def collect_latents(model, dataloader, device="cuda"):
    model.eval()
    model.to(device)

    zs = []
    labels_all = []

    for batch in dataloader:
        # move batch to device while preserving tuple structure
        batch = tuple(x.to(device) for x in batch)

        z, y = model.extract_latents(batch)

        zs.append(z.cpu())
        labels_all.append(y.cpu())

    return torch.cat(zs), torch.cat(labels_all)

# -------------------------------------------------------------
# Metric 6: Embedding/Latent Space Visualization (UMAP or PCA)
# -------------------------------------------------------------

def plot_umap(latent_vectors, labels, class_names=None, save_path=None):
    import matplotlib.pyplot as plt
    import numpy as np
    import umap

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1)
    emb = reducer.fit_transform(latent_vectors)

    plt.figure(figsize=(7, 6))

    classes = np.unique(labels)

    for c in classes:
        idx = labels == c
        name = format_class_label(class_names[c]) if class_names else f"Class {c}"
        plt.scatter(emb[idx, 0], emb[idx, 1], label=name, s=20, alpha=0.8)

    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title("Latent Space (UMAP)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=600, bbox_inches="tight")
    
    plt.close()
