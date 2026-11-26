import torch
import torch.nn.functional as F

class GradCAM1D:
    def __init__(self, lightning_module, target_layer):
        """
        lightning_module: SpectraClassifierModule
        target_layer: conv layer inside SpectraBranch
        """
        self.model = lightning_module
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self.bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out  # (B, C, L)

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]  # (B, C, L)

    def remove_hooks(self):
        self.fwd_hook.remove()
        self.bwd_hook.remove()

    def generate_from_batch(self, batch, class_idx=None, device="cuda"):
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        self.model.to(device)

        inputs, labels = self.model.unpack_batch(batch)

        if "xanes" in inputs:
            inputs["xanes"] = inputs["xanes"].to(device).unsqueeze(1)
        if "exafs" in inputs:
            inputs["exafs"] = inputs["exafs"].to(device).unsqueeze(1)

        logits = self.model(**inputs)  # (B, num_classes)

        # If class_idx not provided: per-sample argmax
        if class_idx is None:
            class_idx = logits.argmax(dim=1)

        # Make scalar loss over batch
        selected_logits = logits[torch.arange(logits.size(0)), class_idx]
        loss = selected_logits.sum()
        loss.backward()

        # GradCAM
        # activations, gradients: (B, C, L)
        weights = self.gradients.mean(dim=2, keepdim=True)  # (B, C, 1)
        cam = (weights * self.activations).sum(dim=1)       # (B, L)
        cam = F.relu(cam)

        # Normalize per sample
        cam_min = cam.min(dim=1, keepdim=True)[0]
        cam_max = cam.max(dim=1, keepdim=True)[0]
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam.detach().cpu()

def generate_cam_for_sample(model, sample, target_layer, device="cuda"):
    """
    sample: (xanes, label) or (xanes, exafs, label)
    """
    cam = GradCAM1D(model, target_layer)

    batch = tuple(x.unsqueeze(0) for x in sample)
    cam_map = cam.generate_from_batch(batch, device=device)

    cam.remove_hooks()
    return cam_map[0].numpy()

def plot_cam(x, cam, title="Grad-CAM", trim=10, save_path=None):
    import matplotlib.pyplot as plt

    x = x[trim:-trim]
    cam = cam[trim:-trim]
    idx = range(len(x))

    plt.rcParams.update({"font.size": 12, "legend.fontsize": 12})

    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(idx, x, label="Spectrum")
    ax.fill_between(idx, 0, cam, alpha=0.4, label="CAM")
    ax.set_title(title, fontsize=12)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(fontsize=12)

    if save_path is not None:
        plt.savefig(save_path, dpi=600, bbox_inches="tight")

    plt.close()
