"""
STAGE 3 pre-flight check: the FULL combined model (encoder-decoder + QNN
together), now that both have already passed in isolation (Stage 1, Stage 2).

Usage:
    python test_stage3_combined.py
"""

import torch

from model import HybridSegModel


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cpu":
        print("WARNING: not running on GPU -- fix CUDA before a real training run.")

    torch.manual_seed(42)
    model = HybridSegModel(
        in_channels=1, base_channels=16, out_channels=1, use_qnn=True
    ).to(device)

    batch_size = 2
    x = torch.randn(batch_size, 1, 128, 128, 128, device=device)
    y = (torch.rand(batch_size, 1, 128, 128, 128, device=device) > 0.98).float()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    print(f"Input shape : {tuple(x.shape)}")
    logits = model(x)
    print(f"Output shape: {tuple(logits.shape)}")
    assert logits.shape == x.shape

    loss_fn = torch.nn.BCEWithLogitsLoss()
    loss = loss_fn(logits, y)
    print(f"Loss value  : {loss.item():.6f}")
    loss.backward()

    print("\nGradient check (per top-level module):")
    any_nan = False
    for name, module in [("encoder", model.encoder),
                          ("bottleneck_module (QNN)", model.bottleneck_module),
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
        print(f"  {name:>26}: {n_params} params with grad, total grad norm = {total_norm:.6f}")

    if any_nan:
        print("\nFAIL: NaN gradients detected in the combined model.")
    else:
        print("\nOK: full combined model forward + backward succeeded, no NaN gradients.")
        if device == "cuda":
            print(f"Peak VRAM used: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()