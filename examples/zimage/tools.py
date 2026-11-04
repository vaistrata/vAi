"""Host-side halves of demo-zimage: the parts of the Z-Image pipeline vAi does not run.

  prepare   prompt -> data/zimage/demo-input.safetensors
              cap_feats [Nc0, 2560]  Qwen3 caption encoder hidden states (the
                                     pipeline's second-to-last layer, padding dropped)
              noise     [16, H/8, W/8] the initial latent
  decode    data/zimage/demo-latent.bin -> data/zimage/demo.png  (VAE)

Both read Tongyi-MAI/Z-Image-Turbo through diffusers (text_encoder/, tokenizer/, vae/):
the full snapshot at data/zimage/Z-Image-Turbo when present (the DiT loader reads
its transformer/ there), the HuggingFace repo otherwise.
"""
import argparse
import os

import numpy as np
import torch
from safetensors.torch import save_file

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data", "zimage")
LOCAL = os.path.join(DATA, "Z-Image-Turbo")
REPO = LOCAL if os.path.exists(os.path.join(LOCAL, "model_index.json")) else "Tongyi-MAI/Z-Image-Turbo"


def prepare(args):
    assert args.height % 16 == 0 and args.width % 16 == 0, "image sides must be multiples of 16"
    from diffusers import ZImagePipeline

    pipe = ZImagePipeline.from_pretrained(REPO, transformer=None, vae=None, torch_dtype=torch.bfloat16)
    pipe.text_encoder.to("cuda")
    embeds, _ = pipe.encode_prompt(args.prompt, device="cuda", do_classifier_free_guidance=False)
    cap = embeds[0].float().cpu().contiguous()

    g = torch.Generator("cpu").manual_seed(args.seed)
    noise = torch.randn((16, args.height // 8, args.width // 8), generator=g, dtype=torch.float32)

    os.makedirs(DATA, exist_ok=True)
    save_file({"cap_feats": cap, "noise": noise}, os.path.join(DATA, "demo-input.safetensors"))
    print(f"cap_feats {tuple(cap.shape)}, noise {tuple(noise.shape)}")


def decode(args):
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(REPO, subfolder="vae", torch_dtype=torch.float32).to("cuda")
    h, w = args.height // 8, args.width // 8
    lat = np.fromfile(os.path.join(DATA, "demo-latent.bin"), dtype=np.float32).reshape(1, 16, h, w)
    lat = torch.from_numpy(lat).to("cuda")
    lat = lat / vae.config.scaling_factor + vae.config.shift_factor
    with torch.no_grad():
        img = vae.decode(lat).sample[0]
    img = ((img.clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()

    from PIL import Image
    Image.fromarray(img).save(os.path.join(DATA, "demo.png"))
    print("wrote data/zimage/demo.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("prompt")
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    d = sub.add_parser("decode")
    d.add_argument("--height", type=int, default=1024)
    d.add_argument("--width", type=int, default=1024)
    args = ap.parse_args()
    {"prepare": prepare, "decode": decode}[args.cmd](args)
