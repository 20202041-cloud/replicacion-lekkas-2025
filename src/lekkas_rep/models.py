"""Modelos de deep learning: CBAM-ResNet-18 3D (flujo B) y fusión deep-radiómica (flujo C).

- CBAM (Convolutional Block Attention Module; Woo et al., 2018): atención por canal ("qué") y espacial ("dónde").
- ResNet-18 3D: bloques residuales básicos [2, 2, 2, 2] con convoluciones 3D.
- Fusión: proyección de la firma radiómica + embedding de la CNN, compuerta squeeze-and-excitation
  (Hu et al., 2018) sobre el vector concatenado y clasificador final.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention3D(nn.Module):
    def __init__(self, ch: int, reduction: int = 16):
        super().__init__()
        hidden = max(ch // reduction, 4)
        self.mlp = nn.Sequential(nn.Conv3d(ch, hidden, 1, bias=False), nn.ReLU(inplace=True),
                                 nn.Conv3d(hidden, ch, 1, bias=False))

    def forward(self, x):
        a = self.mlp(F.adaptive_avg_pool3d(x, 1)) + self.mlp(F.adaptive_max_pool3d(x, 1))
        return x * torch.sigmoid(a)


class SpatialAttention3D(nn.Module):
    def __init__(self, kernel: int = 7):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel, padding=kernel // 2, bias=False)

    def forward(self, x):
        s = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], dim=1)
        return x * torch.sigmoid(self.conv(s))


class CBAM3D(nn.Module):
    def __init__(self, ch: int, reduction: int = 16):
        super().__init__()
        self.ca = ChannelAttention3D(ch, reduction)
        self.sa = SpatialAttention3D()

    def forward(self, x):
        return self.sa(self.ca(x))


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(cout)
        self.conv2 = nn.Conv3d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm3d(cout)
        self.cbam = CBAM3D(cout)
        self.down = None
        if stride != 1 or cin != cout:
            self.down = nn.Sequential(nn.Conv3d(cin, cout, 1, stride, bias=False), nn.BatchNorm3d(cout))

    def forward(self, x):
        idt = x if self.down is None else self.down(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.cbam(self.bn2(self.conv2(out)))
        return F.relu(out + idt, inplace=True)


class CBAMResNet18_3D(nn.Module):
    """Devuelve logits (2 clases) y el embedding de la capa de pooling global."""

    def __init__(self, in_ch: int = 1, base: int = 64, n_classes: int = 2):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv3d(in_ch, base, 7, 2, 3, bias=False), nn.BatchNorm3d(base),
                                  nn.ReLU(inplace=True), nn.MaxPool3d(3, 2, 1))
        chans = [base, base * 2, base * 4, base * 8]
        layers, cin = [], base
        for i, c in enumerate(chans):
            stride = 1 if i == 0 else 2
            layers.append(nn.Sequential(BasicBlock3D(cin, c, stride), BasicBlock3D(c, c, 1)))
            cin = c
        self.layer1, self.layer2, self.layer3, self.layer4 = layers
        self.embed_dim = chans[-1]
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(self.embed_dim, n_classes))

    def features(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)  # mapa de activación usado por Grad-CAM

    def embed(self, x):
        return torch.flatten(F.adaptive_avg_pool3d(self.features(x), 1), 1)

    def forward(self, x):
        e = self.embed(x)
        return self.head(e), e


class SEGate(nn.Module):
    """Compuerta squeeze-and-excitation sobre un vector de rasgos (pondera cada componente en [0, 1])."""

    def __init__(self, dim: int, reduction: int = 16):
        super().__init__()
        hidden = max(dim // reduction, 8)
        self.fc = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, dim), nn.Sigmoid())

    def forward(self, z):
        w = self.fc(z)
        return z * w, w


class DeepRadiomicFusion(nn.Module):
    """Flujo C: une el embedding de la CNN con la firma radiómica mediante una compuerta SE.

    mode = "frozen":      la CNN queda congelada (solo se entrenan adaptador, compuerta y clasificador).
    mode = "end_to_end":  la CNN se ajusta junto con el adaptador (con una tasa de aprendizaje menor).
    """

    def __init__(self, backbone: CBAMResNet18_3D, n_radiomic: int, proj_dim: int = 64,
                 se_reduction: int = 16, dropout: float = 0.3, mode: str = "frozen"):
        super().__init__()
        self.backbone = backbone
        self.mode = mode
        self.adaptor = nn.Sequential(nn.Linear(n_radiomic, proj_dim), nn.BatchNorm1d(proj_dim), nn.ReLU(inplace=True))
        dim = backbone.embed_dim + proj_dim
        self.gate = SEGate(dim, se_reduction)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, 2))
        if mode == "frozen":
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x, r):
        if self.mode == "frozen":
            self.backbone.eval()
            with torch.no_grad():
                e = self.backbone.embed(x)
        else:
            e = self.backbone.embed(x)
        z = torch.cat([e, self.adaptor(r)], dim=1)
        zg, w = self.gate(z)
        return self.classifier(zg), w

    def gate_share(self, w: torch.Tensor) -> torch.Tensor:
        """Peso medio que la compuerta asigna a la parte radiómica (interpretabilidad)."""
        return w[:, self.backbone.embed_dim:].mean(1) / w.mean(1).clamp_min(1e-8)
