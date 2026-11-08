"""Z-Image DiT on PyTorch (diffusers) — the reference and the speed baseline for bench-zimage-vai.

1. fp32 forward on fixed inputs -> data/zimage/ref-<H>x<W>.safetensors
     latent [16, h, w], cap_feats [Nc0, 2560], t [1], out [16, h, w]
   plus the bf16 output of the same inputs (out_bf16), so the vAi error can be
   read against the error bf16 itself makes.
2. bf16 timing: one DiT forward (one denoising step), CUDA-synchronized.

Caption features come from data/zimage/demo-input.safetensors when it exists
(examples/zimage/tools.py prepare), otherwise — or with --cap-tokens N —
seeded Gaussian ones (77 tokens unless N is given).

--stages also stores the fp32 activations after each stage (stage.<name>),
which bench-zimage-vai --stages compares one by one to localize a mismatch.
"""
import argparse
import os
import time

import torch
from safetensors.torch import load_file, save_file

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "zimage")


def forward(model, latent, t, cap, dtype):
    x = [latent.to("cuda", dtype).unsqueeze(1)]          # [16, 1, h, w]
    ts = torch.tensor([t], device="cuda", dtype=dtype)
    with torch.no_grad():
        out = model(x, ts, [cap.to("cuda", dtype)], return_dict=False)[0]
    return out[0].squeeze(1)                               # [16, h, w]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(DATA, "Z-Image-Turbo", "transformer"))
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--t", type=float, default=0.5)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--compile", action="store_true", help="also time torch.compile")
    ap.add_argument("--stages", action="store_true", help="store per-stage fp32 activations")
    ap.add_argument("--cap-tokens", type=int, default=0, help="seeded Gaussian caption of this length")
    args = ap.parse_args()

    from diffusers import ZImageTransformer2DModel

    assert args.height % 16 == 0 and args.width % 16 == 0, "image sides must be multiples of 16"
    g = torch.Generator("cpu").manual_seed(0)
    h, w = args.height // 8, args.width // 8
    latent = torch.randn((16, h, w), generator=g)
    demo = os.path.join(DATA, "demo-input.safetensors")
    if args.cap_tokens == 0 and os.path.exists(demo):
        cap = load_file(demo)["cap_feats"]
    else:
        cap = torch.randn((args.cap_tokens or 77, 2560), generator=g)

    # 1. fp32 reference
    model = ZImageTransformer2DModel.from_pretrained(args.ckpt, torch_dtype=torch.float32).to("cuda").eval()
    stages = {}
    if args.stages:
        def keep(name):
            def hook(_, __, out):
                stages[f"stage.{name}"] = out.detach().float().reshape(-1, out.shape[-1]).cpu().contiguous()
            return hook
        model.t_embedder.register_forward_hook(keep("temb"))
        model.all_x_embedder["2-1"].register_forward_hook(keep("x_embed"))
        for i, b in enumerate(model.noise_refiner):
            b.register_forward_hook(keep(f"noise_refiner.{i}"))
        for i, b in enumerate(model.context_refiner):
            b.register_forward_hook(keep(f"context_refiner.{i}"))
        for i in sorted({0, 1, 2, 5, 10, 20, len(model.layers) - 1}):
            model.layers[i].register_forward_hook(keep(f"layers.{i}"))
    ref = forward(model, latent, args.t, cap, torch.float32).float().cpu()
    del model
    torch.cuda.empty_cache()

    # 2. bf16 baseline
    model = ZImageTransformer2DModel.from_pretrained(args.ckpt, torch_dtype=torch.bfloat16).to("cuda").eval()
    out16 = forward(model, latent, args.t, cap, torch.bfloat16).float().cpu()

    path = os.path.join(DATA, f"ref-{args.height}x{args.width}.safetensors")
    save_file({"latent": latent.contiguous(), "cap_feats": cap.contiguous(),
               "t": torch.tensor([args.t]), "out": ref.contiguous(), "out_bf16": out16.contiguous(),
               **stages}, path)
    rel = ((out16 - ref).norm() / ref.norm()).item()
    print(f"reference -> {path}")
    print(f"bf16 vs fp32: rel L2 {rel:.3e}, max abs {(out16 - ref).abs().max().item():.3e}")

    def bench(fn, label):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        times.sort()
        print(f"{label}: median {times[len(times) // 2] * 1e3:.1f} ms, min {times[0] * 1e3:.1f} ms "
              f"({args.iters} iters, {args.height}x{args.width}, {len(cap)} caption tokens)")

    bench(lambda: forward(model, latent, args.t, cap, torch.bfloat16), "torch bf16 eager")
    if args.compile:
        model = torch.compile(model)
        bench(lambda: forward(model, latent, args.t, cap, torch.bfloat16), "torch bf16 compile")


if __name__ == "__main__":
    main()
