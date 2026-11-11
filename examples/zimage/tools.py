"""Host-side tools for demo-zimage and the Android app (examples/zimage/android).

  prepare   prompt -> data/zimage/demo-input.safetensors
              cap_feats [Nc0, 2560]  Qwen3 caption encoder hidden states (the
                                     pipeline's second-to-last layer, padding dropped)
              noise     [16, H/8, W/8] the initial latent
            and the prompt itself -> data/zimage/prompt.txt  (shown by the Android app)
  decode    data/zimage/demo-latent.bin -> data/zimage/demo.png  (VAE)
  text-encoder  text_encoder/ -> data/zimage/Z-Image-Turbo-text-encoder-q4.safetensors,
            the caption encoder's prepared file, byte for byte what vAi writes
            (models/zimage/tongyi-text-encoder-weights.h) but without a GPU or
            the 8 GB checkpoint on the device that runs it

prepare and decode read Tongyi-MAI/Z-Image-Turbo through diffusers (text_encoder/, tokenizer/, vae/):
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
    with open(os.path.join(DATA, "prompt.txt"), "w") as f:   # shown by the Android app
        f.write(args.prompt + "\n")
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


def half_bits(f):
    """fp32 -> fp16 bits as vAi's floatToHalf: round to nearest even, subnormals to zero."""
    b = np.ascontiguousarray(f, dtype=np.float32).view(np.uint32)
    sign = (b >> 16) & 0x8000
    exp_raw = (b >> 23) & 0xFF
    mant = b & 0x7FFFFF
    exp = exp_raw.astype(np.int64) - 112
    rest = mant & 0x1FFF
    mant10 = (mant >> 13) + ((rest > 0x1000) | ((rest == 0x1000) & ((mant >> 13) & 1).astype(bool)))
    normal = (exp * 0x400 + mant10).astype(np.int64)
    out = np.where(normal >= 0x7C00, 0x7C00, normal)
    out = np.where(exp <= 0, 0, out)
    out = np.where(exp_raw == 0xFF, 0x7C00 | np.where(mant != 0, 0x200, 0), out)
    return (sign | out.astype(np.uint32)).astype(np.uint16)


def q4_bytes(w):
    """fp32 [..., K] -> vAi's Q4 bytes (Dtype::Q4): every 8 4-bit values of a group of
    32 in one word, all words first, then one fp16 scale per group. As vAi's
    quantizeQ4: the group's largest magnitude maps to q = 0, the scale rounded first."""
    x = np.ascontiguousarray(w, dtype=np.float32).reshape(-1, 32)
    a = np.abs(x)
    i = a.argmax(1)
    peak = np.where(a.max(1) > 0, x[np.arange(len(x)), i], np.float32(0)).astype(np.float32)
    scale = half_bits(peak / np.float32(-8.0))
    d = scale.view(np.float16).astype(np.float32)
    inv = np.where(d != 0, np.float32(1.0) / np.where(d != 0, d, 1), 0).astype(np.float32)
    q = np.clip(np.floor(x.astype(np.float64) * inv.astype(np.float64)[:, None] + 8.5), 0, 15).astype(np.uint32)
    q = q.reshape(-1, 8)
    words = np.zeros(len(q), dtype=np.uint32)
    for k in range(8):
        words |= q[:, k] << np.uint32(4 * k)
    return words.tobytes() + scale.tobytes()


def text_encoder(args):
    """The tensors of tongyi::forEachWeight(TextEncoder&) in its order, then the
    embedding table in fp16, written as vAi's saveSafetensors() does."""
    import json
    from safetensors import safe_open

    folder = os.path.join(LOCAL, "text_encoder")
    with open(os.path.join(folder, "model.safetensors.index.json")) as f:
        where = json.load(f)["weight_map"]
    files = {}

    def get(key):
        shard = where[key]
        if shard not in files:
            files[shard] = safe_open(os.path.join(folder, shard), "pt")
        return files[shard].get_tensor(key).float().numpy()

    with open(os.path.join(folder, "config.json")) as f:
        cfg = json.load(f)
    layers = args.layers or cfg["num_hidden_layers"] - 1
    D, F, dh = cfg["hidden_size"], cfg["intermediate_size"], cfg["head_dim"]
    hq, hkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]

    # (name, dtype, shape, make)
    plan = []
    for i in range(layers):
        p = f"model.layers.{i}."
        f32 = lambda k: (lambda: get(k).astype(np.float32).tobytes())
        q4 = lambda k: (lambda: q4_bytes(get(k)))
        plan += [
            (p + "input_layernorm.weight", "F32", [D], f32(p + "input_layernorm.weight")),
            (p + "self_attn.q_proj.weight", "Q4", [hq * dh, D], q4(p + "self_attn.q_proj.weight")),
            (p + "self_attn.k_proj.weight", "Q4", [hkv * dh, D], q4(p + "self_attn.k_proj.weight")),
            (p + "self_attn.v_proj.weight", "Q4", [hkv * dh, D], q4(p + "self_attn.v_proj.weight")),
            (p + "self_attn.q_norm.weight", "F32", [dh], f32(p + "self_attn.q_norm.weight")),
            (p + "self_attn.k_norm.weight", "F32", [dh], f32(p + "self_attn.k_norm.weight")),
            (p + "self_attn.o_proj.weight", "Q4", [D, hq * dh], q4(p + "self_attn.o_proj.weight")),
            (p + "post_attention_layernorm.weight", "F32", [D], f32(p + "post_attention_layernorm.weight")),
            (p + "mlp.gate_up_proj.weight", "Q4", [2 * F, D],
             (lambda p=p: q4_bytes(np.concatenate([get(p + "mlp.gate_proj.weight"), get(p + "mlp.up_proj.weight")])))),
            (p + "mlp.down_proj.weight", "Q4", [D, F], q4(p + "mlp.down_proj.weight")),
        ]
    plan.append(("model.embed_tokens.weight", "F16", [cfg["vocab_size"], D],
                 lambda: half_bits(get("model.embed_tokens.weight")).tobytes()))

    def nbytes(dtype, shape):
        n = int(np.prod(shape))
        return n // 2 + n // 32 * 2 if dtype == "Q4" else n * {"F32": 4, "F16": 2}[dtype]

    header, offset = [], 0
    for name, dtype, shape, _ in plan:
        n = nbytes(dtype, shape)
        header.append(f'"{name}":{{"dtype":"{dtype}","shape":[{",".join(map(str, shape))}],'
                      f'"data_offsets":[{offset},{offset + n}]}}')
        offset += n
    header = "{" + ",".join(header) + "}"
    header += " " * ((8 - len(header) % 8) % 8)

    out = args.out or os.path.join(DATA, "Z-Image-Turbo-text-encoder-q4.safetensors")
    with open(out + ".partial", "wb") as f:
        f.write(len(header).to_bytes(8, "little") + header.encode())
        for name, dtype, shape, make in plan:
            data = make()
            assert len(data) == nbytes(dtype, shape), name
            f.write(data)
    os.replace(out + ".partial", out)
    print(f"wrote {out} ({layers} layers, {offset / 2**30:.2f} GiB)")


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
    t = sub.add_parser("text-encoder")
    t.add_argument("--layers", type=int, default=0, help="the first N layers only (tests); 0: all it runs")
    t.add_argument("--out", default="")
    args = ap.parse_args()
    {"prepare": prepare, "decode": decode, "text-encoder": text_encoder}[args.cmd](args)
