"""
Hybrid 3D CNN + QNN segmentation model (Section 7-10).

Two components, built and tested SEPARATELY before being combined:
  - Encoder3D / Decoder3D: standard 3D U-Net-style encoder-decoder with
    skip connections. No PennyLane dependency -- test this in isolation
    first (see test_stage1_encoder_decoder.py).
  - QNNBottleneck: the 4-qubit quantum channel-gating module (Section 9).
    Requires PennyLane. Test this in isolation second
    (see test_stage2_qnn.py), before wiring it into the full model.

HybridSegModel combines both, with use_qnn=False available as a fallback
(identity bottleneck) so the encoder-decoder can still be exercised even
before PennyLane is installed/working.
"""

import torch
import torch.nn as nn

try:
    import pennylane as qml
    _PENNYLANE_AVAILABLE = True
except ImportError:
    _PENNYLANE_AVAILABLE = False


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.InstanceNorm3d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.InstanceNorm3d(out_ch),
        nn.ReLU(inplace=True),
    )


class Encoder3D(nn.Module):
    """4-stage encoder. Input (B,1,128,128,128) -> bottleneck (B,128,16,16,16),
    with skip features saved at each resolution for the decoder."""

    def __init__(self, in_channels=1, base_channels=16):
        super().__init__()
        c = base_channels
        self.stage1 = conv_block(in_channels, c)       # 128^3, c
        self.down1 = nn.Conv3d(c, c, kernel_size=2, stride=2)      # -> 64^3

        self.stage2 = conv_block(c, c * 2)              # 64^3, 2c
        self.down2 = nn.Conv3d(c * 2, c * 2, kernel_size=2, stride=2)  # -> 32^3

        self.stage3 = conv_block(c * 2, c * 4)           # 32^3, 4c
        self.down3 = nn.Conv3d(c * 4, c * 4, kernel_size=2, stride=2)  # -> 16^3

        self.bottleneck = conv_block(c * 4, c * 8)       # 16^3, 8c

    def forward(self, x):
        s1 = self.stage1(x)
        x = self.down1(s1)

        s2 = self.stage2(x)
        x = self.down2(s2)

        s3 = self.stage3(x)
        x = self.down3(s3)

        bottleneck = self.bottleneck(x)  # (B, 8c, 16,16,16)
        return bottleneck, [s1, s2, s3]


class Decoder3D(nn.Module):
    """Mirrors Encoder3D, fusing skip connections at each resolution."""

    def __init__(self, base_channels=16, out_channels=1):
        super().__init__()
        c = base_channels
        self.up3 = nn.ConvTranspose3d(c * 8, c * 4, kernel_size=2, stride=2)  # -> 32^3
        self.dec3 = conv_block(c * 4 + c * 4, c * 4)  # fuse with s3

        self.up2 = nn.ConvTranspose3d(c * 4, c * 2, kernel_size=2, stride=2)  # -> 64^3
        self.dec2 = conv_block(c * 2 + c * 2, c * 2)  # fuse with s2

        self.up1 = nn.ConvTranspose3d(c * 2, c, kernel_size=2, stride=2)  # -> 128^3
        self.dec1 = conv_block(c + c, c)  # fuse with s1

        self.out_conv = nn.Conv3d(c, out_channels, kernel_size=1)  # segmentation logits

    def forward(self, bottleneck, skips):
        s1, s2, s3 = skips

        x = self.up3(bottleneck)
        x = self.dec3(torch.cat([x, s3], dim=1))

        x = self.up2(x)
        x = self.dec2(torch.cat([x, s2], dim=1))

        x = self.up1(x)
        x = self.dec1(torch.cat([x, s1], dim=1))

        return self.out_conv(x)  # (B, 1, 128,128,128) raw logits


class QNNBottleneck(nn.Module):
    """
    Global channel-wise gate on the pooled bottleneck (Section 7/9):
        AdaptiveAvgPool3d(1) -> Linear(C->4) -> 4-qubit QNN -> Linear(4->C)
        -> Sigmoid -> multiply onto the original bottleneck feature map.

    This is a Squeeze-and-Excitation-style multiplicative gate, NOT a
    per-voxel operator -- per-voxel QNN evaluation was explicitly rejected
    as computationally infeasible (Section 7).
    """

    def __init__(self, channels, n_qubits=4, n_layers=2):
        super().__init__()
        if not _PENNYLANE_AVAILABLE:
            raise ImportError(
                "PennyLane is not installed. Install it with `pip install pennylane` "
                "before using QNNBottleneck. (Encoder3D/Decoder3D do not require it --"
                " test those first with use_qnn=False.)"
            )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.pre = nn.Linear(channels, n_qubits)
        self.post = nn.Linear(n_qubits, channels)
        self.n_qubits = n_qubits

        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="adjoint")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

        weight_shapes = {"weights": (n_layers, n_qubits)}
        self.qnode = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, bottleneck):
        b, c = bottleneck.shape[0], bottleneck.shape[1]
        pooled = self.pool(bottleneck).view(b, c)          # (B, C)
        q_in = self.pre(pooled)                             # (B, 4)

        # Force float32 around the QNN even under global AMP (Section 9) --
        # quantum rotation gradients are numerically sensitive to precision loss.
        autocast_device = "cuda" if q_in.is_cuda else "cpu"
        with torch.amp.autocast(device_type=autocast_device, enabled=False):
            q_in = q_in.to(torch.float32)
            q_out = self.qnode(q_in)                        # (B, 4)

        gate = torch.sigmoid(self.post(q_out))               # (B, C) in (0,1)
        gate = gate.view(b, c, 1, 1, 1)
        return bottleneck * gate


class IdentityBottleneck(nn.Module):
    """Stage-1 stand-in for QNNBottleneck: a plain linear channel mix with
    no quantum component, so the encoder-decoder can be validated on its
    own (Section: two-stage testing plan)."""

    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 2, channels),
            nn.Sigmoid(),
        )

    def forward(self, bottleneck):
        b, c = bottleneck.shape[0], bottleneck.shape[1]
        pooled = self.pool(bottleneck).view(b, c)
        gate = self.fc(pooled).view(b, c, 1, 1, 1)
        return bottleneck * gate


class HybridSegModel(nn.Module):
    def __init__(self, in_channels=1, base_channels=16, out_channels=1,
                 use_qnn=True, n_qubits=4, n_layers=2):
        super().__init__()
        self.encoder = Encoder3D(in_channels, base_channels)
        bottleneck_channels = base_channels * 8
        self.bottleneck_module = (
            QNNBottleneck(bottleneck_channels, n_qubits, n_layers)
            if use_qnn else IdentityBottleneck(bottleneck_channels)
        )
        self.decoder = Decoder3D(base_channels, out_channels)

    def forward(self, x):
        bottleneck, skips = self.encoder(x)
        bottleneck = self.bottleneck_module(bottleneck)
        return self.decoder(bottleneck, skips)