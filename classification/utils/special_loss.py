import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------
# Supervised Contrastive Loss
# ------------------------------------------------------------

class SupervisedContrastiveLoss(nn.Module):
    """
    SupCon Loss: pulls same-class embeddings together,
    pushes different-class embeddings apart.
    
    Optionally accepts a pair_weights matrix to up-weight
    specific class confusions (e.g. random <-> layered).
    
    Args:
        temperature:  softmax temperature (lower = sharper separation, 
                      typical range 0.07–0.5)
        pair_weights: (num_classes, num_classes) tensor of floats.
                      Entry [i,j] scales the loss contribution when
                      the anchor is class i and a negative is class j.
                      None = uniform weighting.
    """
    def __init__(self, temperature=0.1, pair_weights=None):
        super().__init__()
        self.temperature = temperature
        self.pair_weights = pair_weights  # (num_classes, num_classes) or None

    def forward(self, projections, labels):
        """
        Args:
            projections: (B, D) L2-normalized embeddings from projection head
            labels:      (B,)   integer class labels
        Returns:
            scalar loss
        """
        device = projections.device
        B = projections.shape[0]

        # Cosine similarity matrix (B, B) — safe because projections are L2-normalized
        sim = projections @ projections.T / self.temperature  # (B, B)

        # Mask out self-similarity on diagonal
        self_mask = torch.eye(B, dtype=torch.bool, device=device)
        sim.masked_fill_(self_mask, float('-inf'))

        # Positive mask: same class, not self
        labels_col = labels.unsqueeze(1)   # (B, 1)
        labels_row = labels.unsqueeze(0)   # (1, B)
        pos_mask = (labels_col == labels_row) & ~self_mask   # (B, B)

        # If a sample has no positives in the batch, skip it
        has_positives = pos_mask.any(dim=1)

        # Log-softmax over all non-self pairs
        log_probs = sim - torch.logsumexp(sim, dim=1, keepdim=True)  # (B, B)

        # Optional: weight negative pairs by class confusion severity
        if self.pair_weights is not None:
            pw = self.pair_weights.to(device)
            # Build a (B, B) weight matrix based on anchor vs negative class
            weight_matrix = pw[labels][:, labels]  # (B, B)
            # Only apply weights to negative pairs; positives stay unweighted
            neg_mask = ~pos_mask & ~self_mask
            scale = torch.ones(B, B, device=device)
            scale[neg_mask] = weight_matrix[neg_mask]
            log_probs = log_probs * scale

        # Mean log-prob over positives per anchor, then mean over batch
        loss = -(log_probs * pos_mask).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1)
        loss = loss[has_positives].mean()

        return loss


# ------------------------------------------------------------
# Combined Loss
# ------------------------------------------------------------

class CombinedLoss(nn.Module):
    """
    CrossEntropy + λ * SupCon
    
    λ schedule: you can anneal λ upward over epochs so the model
    first learns a reasonable classifier, then refines the geometry.
    """
    def __init__(self, num_classes=3, temperature=0.1, lam=0.3,
                 class_weights=None, pair_weights=None):
        super().__init__()
        self.lam = lam
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.supcon = SupervisedContrastiveLoss(temperature=temperature,
                                                pair_weights=pair_weights)

    def forward(self, logits, projections, labels):
        ce_loss = self.ce(logits, labels)
        con_loss = self.supcon(projections, labels)
        return ce_loss + self.lam * con_loss, ce_loss, con_loss