"""Z-Image caption encoder (Qwen3) in fp32 — the reference for bench-zimage-text-encoder-vai.

The pipeline's encode_prompt (diffusers ZImagePipeline):
    chat template  <|im_start|>user\\n{prompt}<|im_end|>\\n<|im_start|>assistant\\n
    cap_feats      hidden_states[-2] of Qwen3ForCausalLM over those tokens: the
                   output of layer 34 of 36, before the final norm, padding dropped
written to data/zimage/ref-text-encoder.safetensors:
    prompt     [n] fp32    the prompt's UTF-8 bytes
    ids        [T] fp32    the template's token ids (exact below 2^24)
    cap_feats  [T, 2560]   fp32
    stage.<i>  [T, 2560]   the output of layer i, for a few i (--stages)

The forward is spelled out here in fp32, reading one layer's weights at a
time, so it runs in the memory of a laptop (the checkpoint is bf16, 8 GB; fp32
would need 16). --check-hf also runs transformers' Qwen3 in bf16 on the same
tokens and prints how far it is from this one.
"""
import argparse
import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "zimage")
SNAPSHOT = os.path.join(DATA, "Z-Image-Turbo")


class Weights:
    """The checkpoint's tensors by name, opened shard by shard as asked, as fp32."""

    def __init__(self, folder):
        self.folder = folder
        with open(os.path.join(folder, "model.safetensors.index.json")) as f:
            self.where = json.load(f)["weight_map"]
        self.files = {}

    def __getitem__(self, key):
        shard = self.where[key]
        if shard not in self.files:
            self.files[shard] = safe_open(os.path.join(self.folder, shard), "pt")
        return self.files[shard].get_tensor(key).float()


def rms_norm(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rope(x, theta):
    # x [H, T, Dh]; rotate_half, as transformers' Qwen3.
    dh, t = x.shape[-1], x.shape[-2]
    inv = 1.0 / (theta ** (torch.arange(0, dh, 2, dtype=torch.float64) / dh))
    ang = torch.arange(t, dtype=torch.float64)[:, None] * inv[None, :]
    cos = torch.cat([ang.cos(), ang.cos()], -1).float()
    sin = torch.cat([ang.sin(), ang.sin()], -1).float()
    x1, x2 = x[..., : dh // 2], x[..., dh // 2:]
    return x * cos + torch.cat([-x2, x1], -1) * sin


def encode(w, cfg, ids, layers, stages):
    eps, dh = cfg["rms_norm_eps"], cfg["head_dim"]
    hq, hkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    t = len(ids)
    x = w["model.embed_tokens.weight"][torch.tensor(ids)]
    mask = torch.full((t, t), float("-inf")).triu(1)
    out = {}
    for i in range(layers):
        p = f"model.layers.{i}."
        h = rms_norm(x, w[p + "input_layernorm.weight"], eps)
        q = (h @ w[p + "self_attn.q_proj.weight"].T).view(t, hq, dh)
        k = (h @ w[p + "self_attn.k_proj.weight"].T).view(t, hkv, dh)
        v = (h @ w[p + "self_attn.v_proj.weight"].T).view(t, hkv, dh)
        q = rope(rms_norm(q, w[p + "self_attn.q_norm.weight"], eps).transpose(0, 1), cfg["rope_theta"])
        k = rope(rms_norm(k, w[p + "self_attn.k_norm.weight"], eps).transpose(0, 1), cfg["rope_theta"])
        v = v.transpose(0, 1)
        k = k.repeat_interleave(hq // hkv, 0)
        v = v.repeat_interleave(hq // hkv, 0)
        a = torch.softmax(q @ k.transpose(1, 2) / dh ** 0.5 + mask, -1) @ v   # [H, T, Dh]
        x = x + a.transpose(0, 1).reshape(t, hq * dh) @ w[p + "self_attn.o_proj.weight"].T
        h = rms_norm(x, w[p + "post_attention_layernorm.weight"], eps)
        g = torch.nn.functional.silu(h @ w[p + "mlp.gate_proj.weight"].T) * (h @ w[p + "mlp.up_proj.weight"].T)
        x = x + g @ w[p + "mlp.down_proj.weight"].T
        if i in stages:
            out[f"stage.{i}"] = x.clone()
    return x, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="?", default="A red fox sitting in a snowy birch forest at dawn")
    ap.add_argument("--snapshot", default=SNAPSHOT)
    ap.add_argument("--stages", action="store_true", help="store a few layers' outputs")
    ap.add_argument("--check-hf", action="store_true", help="compare with transformers' bf16 Qwen3")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(os.path.join(args.snapshot, "tokenizer"))
    text = tok.apply_chat_template([{"role": "user", "content": args.prompt}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=True)
    ids = tok(text, truncation=True, max_length=512).input_ids
    print(repr(text))
    print(f"{len(ids)} tokens:", ids)

    folder = os.path.join(args.snapshot, "text_encoder")
    with open(os.path.join(folder, "config.json")) as f:
        cfg = json.load(f)
    layers = cfg["num_hidden_layers"] - 1          # hidden_states[-2]
    stages = {0, 1, layers // 2, layers - 1} if args.stages else set()

    with torch.no_grad():
        feats, out = encode(Weights(folder), cfg, ids, layers, stages)
    # fp32 throughout: vAi's safetensors reader takes float dtypes only.
    out.update({"prompt": torch.tensor(list(args.prompt.encode()), dtype=torch.float32),
                "ids": torch.tensor(ids, dtype=torch.float32), "cap_feats": feats.contiguous()})
    path = os.path.join(DATA, "ref-text-encoder.safetensors")
    save_file({k: v.contiguous() for k, v in out.items()}, path, metadata={"prompt": args.prompt})
    print("wrote", path, tuple(feats.shape))

    if args.check_hf:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(folder, torch_dtype=torch.bfloat16).eval()
        with torch.no_grad():
            hf = model(input_ids=torch.tensor([ids]), output_hidden_states=True).hidden_states[-2][0].float()
        rel = ((hf - feats).norm() / feats.norm()).item()
        cos = torch.nn.functional.cosine_similarity(hf.flatten(), feats.flatten(), 0).item()
        print(f"transformers bf16 vs this fp32: rel L2 {rel:.3e}, cosine {cos:.6f}")


if __name__ == "__main__":
    main()
