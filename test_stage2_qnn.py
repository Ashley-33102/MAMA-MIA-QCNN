"""
STAGE 2 pre-flight check: the QNN module in ISOLATION, no CNN involved.
Only run this after Stage 1 (test_stage1_encoder_decoder.py) has already
passed -- this isolates PennyLane/TorchLayer integration issues from any
encoder/decoder issues.

Checks, in order:
  1. PennyLane is installed and importable.
  2. TorchLayer accepts a batched input of shape (batch, n_qubits) directly
     (no manual per-sample looping).
  3. A full forward + backward pass produces real, non-NaN gradients on
     both the classical pre/post linear layers AND the quantum circuit
     weights.
  4. Wall-clock time for one step, so you know up front whether quantum
     evaluation will be a training bottleneck.

Usage:
    python test_stage2_qnn.py
"""

import time

import torch

from model import QNNBottleneck, _PENNYLANE_AVAILABLE


def main():
    if not _PENNYLANE_AVAILABLE:
        print("FAIL: PennyLane is not installed. Run: pip install pennylane")
        return

    import pennylane as qml
    print(f"PennyLane version: {qml.__version__}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print("(Note: PennyLane's default.qubit simulator itself runs on CPU regardless "
          "of `device` above -- only the classical Linear/Sigmoid layers around it move to GPU.)")

    torch.manual_seed(42)
    channels = 128  # matches base_channels*8 in the full model (16*8)
    qnn = QNNBottleneck(channels=channels, n_qubits=4, n_layers=2).to(device)

    batch_size = 2
    # Fake pooled bottleneck feature map: (B, C, D, H, W) with small spatial
    # dims, since QNNBottleneck pools to (B,C) internally anyway.
    fake_bottleneck = torch.randn(batch_size, channels, 16, 16, 16, device=device)

    print(f"\nInput shape : {tuple(fake_bottleneck.shape)}")
    t0 = time.time()
    out = qnn(fake_bottleneck)
    t_forward = time.time() - t0
    print(f"Output shape: {tuple(out.shape)}")
    assert out.shape == fake_bottleneck.shape, "QNNBottleneck must preserve shape!"

    loss = out.mean()
    t0 = time.time()
    loss.backward()
    t_backward = time.time() - t0

    print(f"\nForward time : {t_forward*1000:.1f} ms")
    print(f"Backward time: {t_backward*1000:.1f} ms")

    print("\nGradient check:")
    any_nan = False
    any_missing = False
    for name, p in qnn.named_parameters():
        if p.grad is None:
            print(f"  {name:>20}: NO GRADIENT (this parameter is not being trained!)")
            any_missing = True
        else:
            has_nan = torch.isnan(p.grad).any().item()
            any_nan = any_nan or has_nan
            print(f"  {name:>20}: grad norm = {p.grad.norm().item():.6f}"
                  f"{'  <-- NaN!' if has_nan else ''}")

    print()
    if any_nan:
        print("FAIL: NaN gradients in the QNN module.")
    elif any_missing:
        print("FAIL: some QNN parameters have no gradient -- check they're actually "
              "wired into the forward pass.")
    else:
        print("OK: QNN forward + backward pass succeeded, all parameters have real gradients.")

    est_per_epoch = (t_forward + t_backward) * (68 / batch_size)  # 68 = train set size
    print(f"\nRough estimate: ~{est_per_epoch:.2f}s of QNN compute per training epoch "
          f"(68 training cases / batch size {batch_size}) -- if this is more than a "
          f"couple seconds, batching/throughput is worth a closer look before the full run.")


if __name__ == "__main__":
    main()