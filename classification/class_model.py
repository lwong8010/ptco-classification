import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
# 1. Basic Residual Block for 1D Spectra
# ------------------------------------------------------------

class ResBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
        )

        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(x + self.block(x))


# ------------------------------------------------------------
# 2. Multi-Head Self-Attention for 1D Signals
# ------------------------------------------------------------

class SelfAttention1D(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, x):
        """
        x: (batch, channels, length)
        MultiheadAttention expects (batch, length, channels)
        """
        x_t = x.transpose(1, 2)  # (B, L, C)
        out, _ = self.attn(x_t, x_t, x_t)
        return out.transpose(1, 2)  # back to (B, C, L)


# ------------------------------------------------------------
# 3. One Branch: CNN → Residual Blocks → Attention → MLP
# ------------------------------------------------------------

class SpectraBranch(nn.Module):
    def __init__(
        self,
        input_length,
        cnn_channels=64,
        mlp_hidden=128,
        dropout=0.15
    ):
        super().__init__()

        # -----------------------------------
        # Initial feature extraction layer
        # -----------------------------------
        self.initial_cnn = nn.Sequential(
            nn.Conv1d(1, cnn_channels, kernel_size=9, padding=4),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
        )

        # -----------------------------------
        # Residual blocks (deep local feature extraction)
        # -----------------------------------
        self.resblocks = nn.Sequential(
            ResBlock1D(cnn_channels, kernel_size=5),
            ResBlock1D(cnn_channels, kernel_size=15),
        )

        # -----------------------------------
        # Global attention layer
        # -----------------------------------
        self.attention = SelfAttention1D(cnn_channels, num_heads=4)

        # -----------------------------------
        # Pooling
        # -----------------------------------
        self.pool = nn.AdaptiveAvgPool1d(32)  # reduce sequence length

        # -----------------------------------
        # Compute flattened size dynamically
        # -----------------------------------
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_length)
            z = self.initial_cnn(dummy)
            z = self.resblocks(z)
            z = self.attention(z)
            z = self.pool(z)
            flat_dim = z.shape[1] * z.shape[2]

        self.flat_dim = flat_dim

        # -----------------------------------
        # Branch MLP
        # -----------------------------------
        self.mlp = nn.Sequential(
            nn.Linear(self.flat_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.initial_cnn(x)
        x = self.resblocks(x)
        x = self.attention(x)
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        return self.mlp(x)


# ------------------------------------------------------------
# 4. Full Two-Branch Model
# ------------------------------------------------------------

class TwoBranchSpectraClassifier(nn.Module):
    def __init__(
        self,
        xanes_length,
        exafs_length,
        num_classes=3,
        branch_channels=64,
        branch_hidden=128,
        head_hidden=128,
        dropout=0.2,
    ):
        super().__init__()

        # Branch 1 = XANES
        self.xanes_branch = SpectraBranch(
            input_length=xanes_length,
            cnn_channels=branch_channels,
            mlp_hidden=branch_hidden,
        )

        # Branch 2 = EXAFS
        self.exafs_branch = SpectraBranch(
            input_length=exafs_length,
            cnn_channels=branch_channels,
            mlp_hidden=branch_hidden,
        )

        # Combined classifier head
        self.head = nn.Sequential(
            nn.Linear(branch_hidden * 2, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, num_classes),
        )
        
    def encode(self, xanes, exafs):
        z1 = self.xanes_branch(xanes)
        z2 = self.exafs_branch(exafs)
        return torch.cat([z1, z2], dim=-1)

    def forward(self, xanes, exafs):
        z1 = self.xanes_branch(xanes)
        z2 = self.exafs_branch(exafs)
        z = torch.cat([z1, z2], dim=-1)
        return self.head(z)

# ------------------------------------------------------------
# 5. Full Two-Branch Model with Gated Fusion
# ------------------------------------------------------------

class ProjectionHead(nn.Module):
    """Small MLP projection head for contrastive loss. Discarded at inference."""
    def __init__(self, in_dim, hidden_dim=128, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, z):
        return F.normalize(self.net(z), dim=-1)  # L2 normalize for cosine similarity

class TwoBranchSpectraClassifierGF(nn.Module):
    def __init__(
        self, 
        xanes_length, 
        exafs_length, 
        num_classes=3,
        branch_channels=64, 
        branch_hidden=128,
        head_hidden=128, 
        dropout=0.2,
        proj_hidden=128,   # projection head hidden dim
        proj_out=64        # contrastive embedding dim
    ):
        super().__init__()

        self.xanes_branch = SpectraBranch(xanes_length, branch_channels, branch_hidden)
        self.exafs_branch = SpectraBranch(exafs_length, branch_channels, branch_hidden)

        self.gate = nn.Sequential(
            nn.Linear(branch_hidden * 2, branch_hidden),
            nn.Sigmoid()
        )

        self.head = nn.Sequential(
            nn.Linear(branch_hidden, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, num_classes),
        )
        
        self.projector = ProjectionHead(branch_hidden, proj_hidden, proj_out)

    def encode(self, xanes, exafs):
        z1 = self.xanes_branch(xanes)
        z2 = self.exafs_branch(exafs)

        g = self.gate(torch.cat([z1, z2], dim=-1))
        z = g * z1 + (1 - g) * z2
        return z

    def forward(self, xanes, exafs, return_projection=False):
        z = self.encode(xanes, exafs)
        logits = self.head(z)
        if return_projection:
            proj = self.projector(z)    # only computed when needed (i.e. during training)
            return logits, proj
        return logits

# ------------------------------------------------------------
# 6. One-Branch Model (trains on either XANES or EXAFS)
# ------------------------------------------------------------

class OneBranchSpectraClassifier(nn.Module):
    def __init__(
        self,
        input_length,
        num_classes=3,
        branch_channels=64,
        branch_hidden=128,
        head_hidden=128,
        dropout=0.2,
    ):
        super().__init__()

        # Single spectral branch (XANES OR EXAFS)
        self.branch = SpectraBranch(
            input_length=input_length,
            cnn_channels=branch_channels,
            mlp_hidden=branch_hidden,
        )

        # Classifier head
        self.head = nn.Sequential(
            nn.Linear(branch_hidden, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, num_classes),
        )

    def encode(self, spectra):
        """
        Returns latent embedding of the spectrum.
        """
        return self.branch(spectra)

    def forward(self, spectra):
        """
        spectra: (B, L) or (B, 1, L) depending on SpectraBranch
        """
        z = self.encode(spectra)
        return self.head(z)


# ------------------------------------------------------------
# 7. Two-Branch Baseline Model (ablation target: no SpectraBranch)
# ------------------------------------------------------------

class SimpleSpectraBranch(nn.Module):
    """
    Single linear projection of the flattened raw spectrum. No hidden layers,
    no nonlinearity. Minimal baseline for ablation against SpectraBranch.
    """
    def __init__(
        self,
        input_length,
        mlp_hidden=128,
    ):
        super().__init__()
        # input_length is accepted for interface parity with SpectraBranch but
        # unused: the projection is lazy so it adapts to the actual flattened
        # length, mirroring SpectraBranch's length-agnostic adaptive pooling.
        self.proj = nn.LazyLinear(mlp_hidden)

    def forward(self, x):
        """
        x: (B, 1, L) or (B, L)
        """
        x = x.reshape(x.size(0), -1)
        return self.proj(x)


class TwoBranchSpectraClassifierBaseline(nn.Module):
    """
    Minimal two-branch baseline for ablation against TwoBranchSpectraClassifier:
    each branch is a single linear projection of the flattened raw spectrum
    (no hidden layers, no nonlinearity), concatenated and passed through a
    single linear classifier head. Accepts the same kwargs as
    TwoBranchSpectraClassifier (branch_channels, head_hidden, dropout are
    unused) so it's a drop-in swap via model_type.
    """
    def __init__(
        self,
        xanes_length,
        exafs_length,
        num_classes=3,
        branch_channels=64,
        branch_hidden=128,
        head_hidden=128,
        dropout=0.2,
    ):
        super().__init__()

        self.xanes_branch = SimpleSpectraBranch(input_length=xanes_length, mlp_hidden=branch_hidden)
        self.exafs_branch = SimpleSpectraBranch(input_length=exafs_length, mlp_hidden=branch_hidden)

        self.head = nn.Linear(branch_hidden * 2, num_classes)

    def encode(self, xanes, exafs):
        z1 = self.xanes_branch(xanes)
        z2 = self.exafs_branch(exafs)
        return torch.cat([z1, z2], dim=-1)

    def forward(self, xanes, exafs):
        z = self.encode(xanes, exafs)
        return self.head(z)
