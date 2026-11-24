#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import struct
import time
import numpy as np

E2M1_LUT = np.array([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
], dtype=np.float32)

def make_e4m3_lut():
    lut = np.zeros(256, dtype=np.float32)
    for b in range(256):
        s = -1.0 if (b & 0x80) else 1.0
        e = (b >> 3) & 0xF
        m = b & 0x7
        if e == 0:
            val = (m / 8.0) * (2.0 ** -6)
        elif e == 15 and m == 7:
            val = 0.0
        else:
            val = (1.0 + m / 8.0) * (2.0 ** (e - 7))
        lut[b] = s * val
    return lut

E4M3_LUT = make_e4m3_lut()

def make_ue8m0_lut():
    lut = np.zeros(256, dtype=np.float32)
    for b in range(256):
        lut[b] = 0.0 if b == 255 else (2.0 ** (b - 127))
    return lut

UE8M0_LUT = make_ue8m0_lut()

def dequant_nvfp4_expert(w_u8, sc_u8, amax_f32, denom_f32):
    R, half_K = w_u8.shape
    K = half_K * 2
    lo = E2M1_LUT[w_u8 & 0xF]
    hi = E2M1_LUT[(w_u8 >> 4) & 0xF]
    w = np.empty((R, K), dtype=np.float32)
    w[:, 0::2] = lo
    w[:, 1::2] = hi
    sc = E4M3_LUT[sc_u8] * (amax_f32 / denom_f32)
    w *= np.repeat(sc, 16, axis=1)
    return w

def dequant_mxfp8_expert(w_u8, sc_u8):
    w = E4M3_LUT[w_u8]
    sc = UE8M0_LUT[sc_u8]
    w *= np.repeat(sc, 32, axis=1)
    return w

def quantize_q2_g64_mse(w_f32):
    R, K = w_f32.shape
    assert K % 64 == 0
    G = K // 64
    wg = w_f32.reshape(R, G, 64)
    w_min = wg.min(axis=2, keepdims=True)
    w_max = wg.max(axis=2, keepdims=True)
    w_mid = 0.5 * (w_min + w_max)
    half_r = np.maximum(0.5 * (w_max - w_min), 1e-8)

    best_err = np.full((R, G), np.inf, dtype=np.float32)
    best_lo = np.zeros((R, G), dtype=np.float32)
    best_sc = np.zeros((R, G), dtype=np.float32)
    best_q = np.zeros((R, G, 64), dtype=np.uint8)

    for shrink in (1.0, 0.88, 0.76, 0.64):
        lo = w_mid - shrink * half_r
        hi = w_mid + shrink * half_r
        sc = np.maximum((hi - lo) / 3.0, 1e-8)
        q = np.clip(np.rint((wg - lo) / sc), 0.0, 3.0).astype(np.uint8)
        rec = q.astype(np.float32) * sc + lo
        err = np.sum((wg - rec) ** 2, axis=2)
        mask = err < best_err
        best_err = np.where(mask, err, best_err)
        best_lo = np.where(mask, lo[:, :, 0], best_lo)
        best_sc = np.where(mask, sc[:, :, 0], best_sc)
        best_q = np.where(mask[:, :, None], q, best_q)

    q_flat = best_q.reshape(R, K // 4, 4)
    packed = (
        q_flat[:, :, 0]
        | (q_flat[:, :, 1] << 2)
        | (q_flat[:, :, 2] << 4)
        | (q_flat[:, :, 3] << 6)
    ).astype(np.uint8)
    return packed, best_sc.astype(np.float16), best_lo.astype(np.float16)

def read_safetensors_header(path):
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hlen).decode("utf-8"))
    return hdr, 8 + hlen

def main():
    ap = argparse.ArgumentParser(description="Quantize Inkling-2-Mini MoE experts to native warpgrid Q2_G64 safetensors")
    ap.add_argument("--src", required=True, help="Source Inkling-2-Mini model directory")
    ap.add_argument("--dst", required=True, help="Output Q2 model directory")
    ap.add_argument("--keep-nvfp4-layers", default="0,1", help="Comma-separated layer IDs to keep at NVFP4")
    args = ap.parse_args()

    keep_layers = set(int(x) for x in args.keep_nvfp4_layers.split(",") if x.strip())
    os.makedirs(args.dst, exist_ok=True)
    for fn in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        sp = os.path.join(args.src, fn)
        if os.path.exists(sp):
            shutil.copy2(sp, os.path.join(args.dst, fn))

    src_st = os.path.join(args.src, "model-00001-of-00001.safetensors")
    if not os.path.exists(src_st):
        src_st = os.path.join(args.src, "model.safetensors")
    if not os.path.exists(src_st):
        raise FileNotFoundError(
            f"Expected single-file safetensors (model-00001-of-00001.safetensors or model.safetensors) in {args.src}; "
            "sharded checkpoints are not supported."
        )
    dst_st = os.path.join(args.dst, "model-00001-of-00001.safetensors")
    hdr, data_off = read_safetensors_header(src_st)

    skip_names = set()
    q2_layers = []
    for i in range(24):
        p = f"model.llm.layers.{i}.mlp.experts."
        if i in keep_layers:
            continue
        if p + "w13_weight" in hdr:
            q2_layers.append(i)
            for suf in (
                "w13_weight", "w13_weight_scale", "w13_weight_amax", "w13_weight_outer_scale_denominator",
                "w2_weight", "w2_weight_scale", "w2_weight_amax", "w2_weight_outer_scale_denominator",
            ):
                if p + suf in hdr:
                    skip_names.add(p + suf)

    t0 = time.time()
    out_meta = {}
    cur_off = 0

    def add_entry(name, dtype, shape, nbytes):
        nonlocal cur_off
        out_meta[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [cur_off, cur_off + nbytes]}
        cur_off += nbytes

    copy_list = []
    for k, v in hdr.items():
        if k == "__metadata__" or k in skip_names:
            continue
        b0, b1 = v["data_offsets"]
        nbytes = b1 - b0
        add_entry(k, v["dtype"], v["shape"], nbytes)
        copy_list.append((k, data_off + b0, nbytes))

    q2_specs = []
    for i in q2_layers:
        p = f"model.llm.layers.{i}.mlp.experts."
        w13_info = hdr[p + "w13_weight"]
        w2_info = hdr[p + "w2_weight"]
        is_mxfp8 = (w13_info["dtype"] == "F8_E4M3")
        E, R13, K13_raw = w13_info["shape"]
        K13 = K13_raw if is_mxfp8 else K13_raw * 2
        _, R2, K2_raw = w2_info["shape"]
        K2 = K2_raw if is_mxfp8 else K2_raw * 2

        add_entry(p + "w13_q2_weight", "U8", [E, R13, K13 // 4], E * R13 * (K13 // 4))
        add_entry(p + "w13_q2_scale", "F16", [E, R13, K13 // 64], E * R13 * (K13 // 64) * 2)
        add_entry(p + "w13_q2_min", "F16", [E, R13, K13 // 64], E * R13 * (K13 // 64) * 2)
        add_entry(p + "w2_q2_weight", "U8", [E, R2, K2 // 4], E * R2 * (K2 // 4))
        add_entry(p + "w2_q2_scale", "F16", [E, R2, K2 // 64], E * R2 * (K2 // 64) * 2)
        add_entry(p + "w2_q2_min", "F16", [E, R2, K2 // 64], E * R2 * (K2 // 64) * 2)
        q2_specs.append((i, p, is_mxfp8, E, R13, K13, R2, K2))

    hdr_bytes = json.dumps(out_meta, separators=(",", ":")).encode("utf-8")
    pad = (8 - (len(hdr_bytes) % 8)) % 8
    hdr_bytes += b" " * pad

    print(f"Writing {dst_st} ({(cur_off + len(hdr_bytes) + 8) / (1024**3):.2f} GiB), quantizing {len(q2_layers)} layers to Q2_G64...")
    with open(src_st, "rb") as fin, open(dst_st, "wb") as fout:
        fout.write(struct.pack("<Q", len(hdr_bytes)))
        fout.write(hdr_bytes)

        buf_sz = 16 * 1024 * 1024
        for _, off, nbytes in copy_list:
            fin.seek(off)
            rem = nbytes
            while rem > 0:
                chunk = fin.read(min(buf_sz, rem))
                fout.write(chunk)
                rem -= len(chunk)

        def load_np(name, dtype):
            info = hdr[name]
            b0, b1 = info["data_offsets"]
            fin.seek(data_off + b0)
            return np.frombuffer(fin.read(b1 - b0), dtype=dtype).reshape(info["shape"])

        for i, p, is_mxfp8, E, R13, K13, R2, K2 in q2_specs:
            tl = time.time()
            for mat, R, K in (("w13", R13, K13), ("w2", R2, K2)):
                w_raw = load_np(p + f"{mat}_weight", np.uint8)
                sc_raw = load_np(p + f"{mat}_weight_scale", np.uint8)
                if not is_mxfp8:
                    amax = load_np(p + f"{mat}_weight_amax", np.float32)
                    denom = float(load_np(p + f"{mat}_weight_outer_scale_denominator", np.float32).reshape(-1)[0])
                q_all = np.empty((E, R, K // 4), dtype=np.uint8)
                s_all = np.empty((E, R, K // 64), dtype=np.float16)
                m_all = np.empty((E, R, K // 64), dtype=np.float16)
                for e in range(E):
                    if is_mxfp8:
                        wf = dequant_mxfp8_expert(w_raw[e], sc_raw[e])
                    else:
                        wf = dequant_nvfp4_expert(w_raw[e], sc_raw[e], float(amax[e]), denom)
                    q_all[e], s_all[e], m_all[e] = quantize_q2_g64_mse(wf)
                fout.write(q_all.tobytes())
                fout.write(s_all.tobytes())
                fout.write(m_all.tobytes())
            print(f"  layer {i:2d} ({'mxfp8' if is_mxfp8 else 'nvfp4'} -> q2_g64) done in {time.time() - tl:.2f}s", flush=True)

    print(f"Finished in {time.time() - t0:.1f}s -> {dst_st}")

if __name__ == "__main__":
    main()
