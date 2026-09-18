"""
STAGE 1 pre-flight check: encoder-decoder + skip connections ONLY.
No PennyLane/QNN involved -- this isolates "is my U-Net correct" from
"does the quantum layer work" (see the two-stage testing plan).

Run this first. It should complete in a few seconds and print real,
non-NaN gradient norms for every parameter group.

Usage:
    python test_stage1_encoder_decoder.py
"""

import torch

from model import HybridSegModel


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    torch.manual_seed(42)
    model = HybridSegModel(
        in_channels=1, base_channels=16, out_channels=1, use_qnn=False
    ).to(device)

    batch_size = 2
    x = torch.randn(batch_size, 1, 128, 128, 128, device=device)
    y = (torch.rand(batch_size, 1, 128, 128, 128, device=device) > 0.98).float()  # sparse fake mask

    print(f"Input shape : {tuple(x.shape)}")
    logits = model(x)
    print(f"Output shape: {tuple(logits.shape)}")
    assert logits.shape == x.shape, "Output shape must match input spatial shape!"

    # Simple BCE-with-logits smoke test loss (swap for DiceCELoss in the real
    # training script -- this file only checks the encoder/decoder plumbing).
    loss_fn = torch.nn.BCEWithLogitsLoss()
    loss = loss_fn(logits, y)
    print(f"Loss value  : {loss.item():.6f}")

    loss.backward()

    print("\nGradient check (per top-level module):")
    any_nan = False
    for name, module in [("encoder", model.encoder),
                          ("bottleneck_module", model.bottleneck_module),
                          ("decoder", model.decoder)]:
        total_norm = 0.0
        n_params = 0
        for p in module.parameters():
            if p.grad is not None:
                n_params += 1
                total_norm += p.grad.norm().item() ** 2
                if torch.isnan(p.grad).any():
                    any_nan = True
        total_norm = total_norm ** 0.5
        print(f"  {name:>18}: {n_params} params with grad, total grad norm = {total_norm:.6f}")

    if any_nan:
        print("\nFAIL: NaN gradients detected.")
    else:
        print("\nOK: forward + backward pass succeeded, no NaN gradients.")
        if device == "cuda":
            print(f"Peak VRAM used: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()