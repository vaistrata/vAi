"""Z-Image VAE decoder on PyTorch (diffusers) — the reference for bench-zimage-vae-vai.

fp32 decode of one latent -> data/zimage/ref-vae-<H>x<W>.safetensors
    latent [16, h, w]  the DiT's latent, before / scaling + shift
    image  [3, H, W]   the decoder output (about [-1, 1])
plus, with --stages, the fp32 activations after conv_in, the mid block and
each up block (stage.<name>), which bench-zimage-vae-vai --stages compares
one by one. Then the decode is timed.

The latent is data/zimage/demo-latent.bin when it exists (what
examples/zimage/demo wrote), otherwise a seeded Gaussian one. Runs on CUDA
when there is a GPU, else on the CPU (a 512x512 decode takes seconds).
"""
import argparse
import os
import time

import numpy as np
import torch
from safetensors.torch import save_file

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "zimage")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", default=os.path.join(DATA, "Z-Image-Turbo", "vae"))
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--latent", default=os.path.join(DATA, "demo-latent.bin"))
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--stages", action="store_true", help="store per-stage fp32 activations")
    args = ap.parse_args()

    from diffusers import AutoencoderKL

    device = "cuda" if torch.cuda.is_available() else "cpu"
    h, w = args.height // 8, args.width // 8
    if os.path.exists(args.latent):
        latent = torch.from_numpy(np.fromfile(args.latent, dtype=np.float32).reshape(16, h, w))
    else:
        latent = torch.randn((16, h, w), generator=torch.Generator("cpu").manual_seed(0))

    vae = AutoencoderKL.from_pretrained(args.vae, torch_dtype=torch.float32).to(device).eval()
    z = (latent / vae.config.scaling_factor + vae.config.shift_factor).unsqueeze(0).to(device)

    stages = {}
    if args.stages:
        dec = vae.decoder
        taps = [("conv_in", dec.conv_in), ("mid", dec.mid_block)]
        taps += [(f"up.{i}", b) for i, b in enumerate(dec.up_blocks)]
        for name, mod in taps:
            mod.register_forward_hook(lambda m, i, o, name=name: stages.__setitem__(
                "stage." + name, o[0].float().cpu().contiguous()))

    with torch.no_grad():
        image = vae.decode(z).sample[0].float().cpu()

    out = os.path.join(DATA, f"ref-vae-{args.height}x{args.width}.safetensors")
    save_file({"latent": latent.contiguous(), "image": image.contiguous(), **stages}, out)
    print("wrote", out)

    times = []
    with torch.no_grad():
        for _ in range(args.iters):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            vae.decode(z)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    times.sort()
    print(f"torch fp32 ({device}): median {times[len(times) // 2] * 1e3:.1f} ms, "
          f"min {times[0] * 1e3:.1f} ms ({args.iters} iters)")


if __name__ == "__main__":
    main()
