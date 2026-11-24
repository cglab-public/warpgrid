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

E2M1_POS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
E2M1_MID = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=np.float32)

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
E4M3_POS = E4M3_LUT[:127]

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

def encode_e4m3_pos(vals):
    v = np.clip(vals, 0.0, 448.0)
    idx = np.searchsorted(E4M3_POS, v)
    idx = np.clip(idx, 0, 126)
    prev = np.maximum(idx - 1, 0)
    choose_prev = np.abs(v - E4M3_POS[prev]) < np.abs(E4M3_POS[idx] - v)
    return np.where(choose_prev, prev, idx).astype(np.uint8)

def quantize_nvfp4_expert(w_f32):
    R, K = w_f32.shape
    assert K % 16 == 0
    denom = np.float32(6.0 * 448.0)
    amax = np.float32(max(float(np.max(np.abs(w_f32))), 1e-8))
    gscale = amax / denom
    wb = w_f32.reshape(R, K // 16, 16)
    blk_max = np.max(np.abs(wb), axis=2)
    sc_f = blk_max / (6.0 * gscale)
    sc_u8 = encode_e4m3_pos(sc_f)
    eff_sc = np.maximum(E4M3_LUT[sc_u8] * gscale, 1e-12)
    scaled = wb / eff_sc[:, :, None]
    sign = (scaled < 0).astype(np.uint8) << 3
    mag = np.abs(scaled)
    code = np.searchsorted(E2M1_MID, mag).astype(np.uint8)
    nib = (sign | code).reshape(R, K // 2, 2)
    packed = (nib[:, :, 0] | (nib[:, :, 1] << 4)).astype(np.uint8)
    return packed, sc_u8, amax

def quantize_q2_g64_mse(w_f32, imp_f32=None):
    R, K = w_f32.shape
    assert K % 64 == 0
    G = K // 64
    wg = w_f32.reshape(R, G, 64)
    if imp_f32 is not None:
        iw = np.maximum(imp_f32.reshape(1, G, 64), 1e-8).astype(np.float32)
    else:
        iw = np.ones((1, G, 64), dtype=np.float32)

    w_min = wg.min(axis=2, keepdims=True)
    w_max = wg.max(axis=2, keepdims=True)
    w_mid = 0.5 * (w_min + w_max)
    half_r = np.maximum(0.5 * (w_max - w_min), 1e-8)

    best_err = np.full((R, G), np.inf, dtype=np.float32)
    best_lo = np.zeros((R, G), dtype=np.float32)
    best_sc = np.zeros((R, G), dtype=np.float32)
    best_q = np.zeros((R, G, 64), dtype=np.uint8)

    for shrink in (1.0, 0.90, 0.80, 0.70, 0.60):
        lo = w_mid - shrink * half_r
        hi = w_mid + shrink * half_r
        sc = np.maximum((hi - lo) / 3.0, 1e-8)
        q = np.clip(np.rint((wg - lo) / sc), 0.0, 3.0).astype(np.uint8)
        rec = q.astype(np.float32) * sc + lo
        err = np.sum(iw * ((wg - rec) ** 2), axis=2)
        mask = err < best_err
        best_err = np.where(mask, err, best_err)
        best_lo = np.where(mask, lo[:, :, 0], best_lo)
        best_sc = np.where(mask, sc[:, :, 0], best_sc)
        best_q = np.where(mask[:, :, None], q, best_q)

    sum_w = np.sum(iw, axis=2)
    sum_wx = np.sum(iw * wg, axis=2)
    q_cur = best_q
    lo_cur = best_lo
    sc_cur = best_sc
    for _ in range(3):
        qf = q_cur.astype(np.float32)
        sum_wq = np.sum(iw * qf, axis=2)
        sum_wq2 = np.sum(iw * qf * qf, axis=2)
        sum_wqx = np.sum(iw * qf * wg, axis=2)
        det = sum_w * sum_wq2 - sum_wq * sum_wq
        valid = det > 1e-10
        new_sc = np.where(valid, (sum_w * sum_wqx - sum_wq * sum_wx) / np.where(valid, det, 1.0), sc_cur)
        new_sc = np.maximum(new_sc, 1e-8)
        new_lo = np.where(valid, (sum_wx - new_sc * sum_wq) / sum_w, lo_cur)
        new_q = np.clip(np.rint((wg - new_lo[:, :, None]) / new_sc[:, :, None]), 0.0, 3.0).astype(np.uint8)
        rec = new_q.astype(np.float32) * new_sc[:, :, None] + new_lo[:, :, None]
        err = np.sum(iw * ((wg - rec) ** 2), axis=2)
        imp_mask = err < best_err
        best_err = np.where(imp_mask, err, best_err)
        best_lo = np.where(imp_mask, new_lo, best_lo)
        best_sc = np.where(imp_mask, new_sc, best_sc)
        best_q = np.where(imp_mask[:, :, None], new_q, best_q)
        lo_cur = np.where(imp_mask, new_lo, lo_cur)
        sc_cur = np.where(imp_mask, new_sc, sc_cur)
        q_cur = np.where(imp_mask[:, :, None], new_q, q_cur)

    q_flat = best_q.reshape(R, K // 4, 4)
    packed = (
        q_flat[:, :, 0]
        | (q_flat[:, :, 1] << 2)
        | (q_flat[:, :, 2] << 4)
        | (q_flat[:, :, 3] << 6)
    ).astype(np.uint8)
    return packed, best_sc.astype(np.float16), best_lo.astype(np.float16)

def quantize_q3_g64_mse(w_f32, imp_f32=None):
    R, K = w_f32.shape
    assert K % 64 == 0
    G = K // 64
    wg = w_f32.reshape(R, G, 64)
    if imp_f32 is not None:
        iw = np.maximum(imp_f32.reshape(1, G, 64), 1e-8).astype(np.float32)
    else:
        iw = np.ones((1, G, 64), dtype=np.float32)

    w_min = wg.min(axis=2, keepdims=True)
    w_max = wg.max(axis=2, keepdims=True)
    w_mid = 0.5 * (w_min + w_max)
    half_r = np.maximum(0.5 * (w_max - w_min), 1e-8)

    best_err = np.full((R, G), np.inf, dtype=np.float32)
    best_lo = np.zeros((R, G), dtype=np.float32)
    best_sc = np.zeros((R, G), dtype=np.float32)
    best_q = np.zeros((R, G, 64), dtype=np.uint8)

    for shrink in (1.0, 0.92, 0.84, 0.76):
        lo = w_mid - shrink * half_r
        hi = w_mid + shrink * half_r
        sc = np.maximum((hi - lo) / 7.0, 1e-8)
        q = np.clip(np.rint((wg - lo) / sc), 0.0, 7.0).astype(np.uint8)
        rec = q.astype(np.float32) * sc + lo
        err = np.sum(iw * ((wg - rec) ** 2), axis=2)
        mask = err < best_err
        best_err = np.where(mask, err, best_err)
        best_lo = np.where(mask, lo[:, :, 0], best_lo)
        best_sc = np.where(mask, sc[:, :, 0], best_sc)
        best_q = np.where(mask[:, :, None], q, best_q)

    sum_w = np.sum(iw, axis=2)
    sum_wx = np.sum(iw * wg, axis=2)
    q_cur = best_q
    lo_cur = best_lo
    sc_cur = best_sc
    for _ in range(3):
        qf = q_cur.astype(np.float32)
        sum_wq = np.sum(iw * qf, axis=2)
        sum_wq2 = np.sum(iw * qf * qf, axis=2)
        sum_wqx = np.sum(iw * qf * wg, axis=2)
        det = sum_w * sum_wq2 - sum_wq * sum_wq
        valid = det > 1e-10
        new_sc = np.where(valid, (sum_w * sum_wqx - sum_wq * sum_wx) / np.where(valid, det, 1.0), sc_cur)
        new_sc = np.maximum(new_sc, 1e-8)
        new_lo = np.where(valid, (sum_wx - new_sc * sum_wq) / sum_w, lo_cur)
        new_q = np.clip(np.rint((wg - new_lo[:, :, None]) / new_sc[:, :, None]), 0.0, 7.0).astype(np.uint8)
        rec = new_q.astype(np.float32) * new_sc[:, :, None] + new_lo[:, :, None]
        err = np.sum(iw * ((wg - rec) ** 2), axis=2)
        imp_mask = err < best_err
        best_err = np.where(imp_mask, err, best_err)
        best_lo = np.where(imp_mask, new_lo, best_lo)
        best_sc = np.where(imp_mask, new_sc, best_sc)
        best_q = np.where(imp_mask[:, :, None], new_q, best_q)
        lo_cur = np.where(imp_mask, new_lo, lo_cur)
        sc_cur = np.where(imp_mask, new_sc, sc_cur)
        q_cur = np.where(imp_mask[:, :, None], new_q, q_cur)

    q_flat = best_q.reshape(R, K)
    lo2 = (q_flat & 3).reshape(R, K // 4, 4)
    packed_lo = (lo2[:, :, 0] | (lo2[:, :, 1] << 2) | (lo2[:, :, 2] << 4) | (lo2[:, :, 3] << 6)).astype(np.uint8)
    hi1 = ((q_flat >> 2) & 1).reshape(R, K // 8, 8)
    packed_hi = (
        hi1[:, :, 0] | (hi1[:, :, 1] << 1) | (hi1[:, :, 2] << 2) | (hi1[:, :, 3] << 3)
        | (hi1[:, :, 4] << 4) | (hi1[:, :, 5] << 5) | (hi1[:, :, 6] << 6) | (hi1[:, :, 7] << 7)
    ).astype(np.uint8)
    return packed_lo, packed_hi, best_sc.astype(np.float16), best_lo.astype(np.float16)

def read_safetensors_header(path):
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hlen).decode("utf-8"))
    return hdr, 8 + hlen

def main():
    ap = argparse.ArgumentParser(description="Quantize Inkling-2-Mini MoE experts to native warpgrid Q2_G64 / mixed NVFP4 safetensors")
    ap.add_argument("--src", required=True, help="Source Inkling-2-Mini model directory")
    ap.add_argument("--dst", required=True, help="Output Q2 model directory")
    ap.add_argument("--keep-nvfp4-layers", default="0,1", help="Comma-separated layer IDs to keep or convert to NVFP4")
    ap.add_argument("--q3-layers", default="", help="Comma-separated layer IDs to quantize to 3.5-bpw Q3_G64")
    ap.add_argument("--imatrix", default="", help="Optional [24, 2304] float32 activation importance matrix binary")
    ap.add_argument("--imatrix-w2", default="", help="Optional [24, 128, 1152] float32 per-expert w2 importance matrix binary")
    ap.add_argument("--drop-vision", action="store_true", help="Drop model.vision.* tensors to save ~1.1 GiB on disk")
    args = ap.parse_args()

    nvfp4_layers = set(int(x) for x in args.keep_nvfp4_layers.split(",") if x.strip())
    q3_layers_set = set(int(x) for x in args.q3_layers.split(",") if x.strip())
    imatrix = None
    if args.imatrix:
        imatrix = np.fromfile(args.imatrix, dtype=np.float32).reshape(24, 2304)
        print(f"Loaded imatrix {args.imatrix}: shape={imatrix.shape}, mean={imatrix.mean():.4f}")
    imatrix_w2 = None
    if args.imatrix_w2:
        imatrix_w2 = np.fromfile(args.imatrix_w2, dtype=np.float32).reshape(24, 128, 1152)
        print(f"Loaded imatrix_w2 {args.imatrix_w2}: shape={imatrix_w2.shape}, mean={imatrix_w2.mean():.6f}")

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
    q3_layers = []
    conv_nvfp4_layers = []
    for i in range(24):
        p = f"model.llm.layers.{i}.mlp.experts."
        if p + "w13_weight" not in hdr:
            continue
        is_mxfp8 = (hdr[p + "w13_weight"]["dtype"] == "F8_E4M3")
        if i in nvfp4_layers and not is_mxfp8:
            continue
        if i in nvfp4_layers and is_mxfp8:
            conv_nvfp4_layers.append(i)
        elif i in q3_layers_set:
            q3_layers.append(i)
        else:
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
        if args.drop_vision and k.startswith("model.vision."):
            continue
        b0, b1 = v["data_offsets"]
        nbytes = b1 - b0
        add_entry(k, v["dtype"], v["shape"], nbytes)
        copy_list.append((k, data_off + b0, nbytes))

    nvfp4_specs = []
    for i in conv_nvfp4_layers:
        p = f"model.llm.layers.{i}.mlp.experts."
        E, R13, K13 = hdr[p + "w13_weight"]["shape"]
        _, R2, K2 = hdr[p + "w2_weight"]["shape"]
        add_entry(p + "w13_weight", "U8", [E, R13, K13 // 2], E * R13 * (K13 // 2))
        add_entry(p + "w13_weight_scale", "F8_E4M3", [E, R13, K13 // 16], E * R13 * (K13 // 16))
        add_entry(p + "w13_weight_amax", "F32", [E], E * 4)
        add_entry(p + "w13_weight_outer_scale_denominator", "F32", [1], 4)
        add_entry(p + "w2_weight", "U8", [E, R2, K2 // 2], E * R2 * (K2 // 2))
        add_entry(p + "w2_weight_scale", "F8_E4M3", [E, R2, K2 // 16], E * R2 * (K2 // 16))
        add_entry(p + "w2_weight_amax", "F32", [E], E * 4)
        add_entry(p + "w2_weight_outer_scale_denominator", "F32", [1], 4)
        nvfp4_specs.append((i, p, E, R13, K13, R2, K2))

    q3_specs = []
    for i in q3_layers:
        p = f"model.llm.layers.{i}.mlp.experts."
        w13_info = hdr[p + "w13_weight"]
        w2_info = hdr[p + "w2_weight"]
        is_mxfp8 = (w13_info["dtype"] == "F8_E4M3")
        E, R13, K13_raw = w13_info["shape"]
        K13 = K13_raw if is_mxfp8 else K13_raw * 2
        _, R2, K2_raw = w2_info["shape"]
        K2 = K2_raw if is_mxfp8 else K2_raw * 2

        add_entry(p + "w13_q3_lo", "U8", [E, R13, K13 // 4], E * R13 * (K13 // 4))
        add_entry(p + "w13_q3_hi", "U8", [E, R13, K13 // 8], E * R13 * (K13 // 8))
        add_entry(p + "w13_q3_scale", "F16", [E, R13, K13 // 64], E * R13 * (K13 // 64) * 2)
        add_entry(p + "w13_q3_min", "F16", [E, R13, K13 // 64], E * R13 * (K13 // 64) * 2)
        add_entry(p + "w2_q3_lo", "U8", [E, R2, K2 // 4], E * R2 * (K2 // 4))
        add_entry(p + "w2_q3_hi", "U8", [E, R2, K2 // 8], E * R2 * (K2 // 8))
        add_entry(p + "w2_q3_scale", "F16", [E, R2, K2 // 64], E * R2 * (K2 // 64) * 2)
        add_entry(p + "w2_q3_min", "F16", [E, R2, K2 // 64], E * R2 * (K2 // 64) * 2)
        q3_specs.append((i, p, is_mxfp8, E, R13, K13, R2, K2))

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

    print(
        f"Writing {dst_st} ({(cur_off + len(hdr_bytes) + 8) / (1024**3):.2f} GiB): "
        f"{len(nvfp4_specs)} mxfp8->nvfp4, {len(q3_specs)} -> q3_g64, {len(q2_specs)} -> q2_g64 "
        f"(imatrix={'on' if imatrix is not None else 'off'})...",
        flush=True,
    )
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

        denom_arr = np.array([6.0 * 448.0], dtype=np.float32)
        for i, p, E, R13, K13, R2, K2 in nvfp4_specs:
            tl = time.time()
            for mat, R, K in (("w13", R13, K13), ("w2", R2, K2)):
                w_raw = load_np(p + f"{mat}_weight", np.uint8)
                sc_raw = load_np(p + f"{mat}_weight_scale", np.uint8)
                q_all = np.empty((E, R, K // 2), dtype=np.uint8)
                s_all = np.empty((E, R, K // 16), dtype=np.uint8)
                a_all = np.empty((E,), dtype=np.float32)
                for e in range(E):
                    wf = dequant_mxfp8_expert(w_raw[e], sc_raw[e])
                    q_all[e], s_all[e], a_all[e] = quantize_nvfp4_expert(wf)
                fout.write(q_all.tobytes())
                fout.write(s_all.tobytes())
                fout.write(a_all.tobytes())
                fout.write(denom_arr.tobytes())
            print(f"  layer {i:2d} (mxfp8 -> nvfp4) done in {time.time() - tl:.2f}s", flush=True)

        for i, p, is_mxfp8, E, R13, K13, R2, K2 in q3_specs:
            tl = time.time()
            for mat, R, K in (("w13", R13, K13), ("w2", R2, K2)):
                w_raw = load_np(p + f"{mat}_weight", np.uint8)
                sc_raw = load_np(p + f"{mat}_weight_scale", np.uint8)
                if not is_mxfp8:
                    amax = load_np(p + f"{mat}_weight_amax", np.float32)
                    denom = float(load_np(p + f"{mat}_weight_outer_scale_denominator", np.float32).reshape(-1)[0])
                imp = imatrix[i] if (imatrix is not None and mat == "w13") else None
                lo_all = np.empty((E, R, K // 4), dtype=np.uint8)
                hi_all = np.empty((E, R, K // 8), dtype=np.uint8)
                s_all = np.empty((E, R, K // 64), dtype=np.float16)
                m_all = np.empty((E, R, K // 64), dtype=np.float16)
                for e in range(E):
                    if is_mxfp8:
                        wf = dequant_mxfp8_expert(w_raw[e], sc_raw[e])
                    else:
                        wf = dequant_nvfp4_expert(w_raw[e], sc_raw[e], float(amax[e]), denom)
                    imp_e = imp if mat == "w13" else (imatrix_w2[i, e] if imatrix_w2 is not None else None)
                    lo_all[e], hi_all[e], s_all[e], m_all[e] = quantize_q3_g64_mse(wf, imp_e)
                fout.write(lo_all.tobytes())
                fout.write(hi_all.tobytes())
                fout.write(s_all.tobytes())
                fout.write(m_all.tobytes())
            print(f"  layer {i:2d} ({'mxfp8' if is_mxfp8 else 'nvfp4'} -> q3_g64) done in {time.time() - tl:.2f}s", flush=True)

        for i, p, is_mxfp8, E, R13, K13, R2, K2 in q2_specs:
            tl = time.time()
            for mat, R, K in (("w13", R13, K13), ("w2", R2, K2)):
                w_raw = load_np(p + f"{mat}_weight", np.uint8)
                sc_raw = load_np(p + f"{mat}_weight_scale", np.uint8)
                if not is_mxfp8:
                    amax = load_np(p + f"{mat}_weight_amax", np.float32)
                    denom = float(load_np(p + f"{mat}_weight_outer_scale_denominator", np.float32).reshape(-1)[0])
                imp = imatrix[i] if (imatrix is not None and mat == "w13") else None
                q_all = np.empty((E, R, K // 4), dtype=np.uint8)
                s_all = np.empty((E, R, K // 64), dtype=np.float16)
                m_all = np.empty((E, R, K // 64), dtype=np.float16)
                for e in range(E):
                    if is_mxfp8:
                        wf = dequant_mxfp8_expert(w_raw[e], sc_raw[e])
                    else:
                        wf = dequant_nvfp4_expert(w_raw[e], sc_raw[e], float(amax[e]), denom)
                    imp_e = imp if mat == "w13" else (imatrix_w2[i, e] if imatrix_w2 is not None else None)
                    q_all[e], s_all[e], m_all[e] = quantize_q2_g64_mse(wf, imp_e)
                fout.write(q_all.tobytes())
                fout.write(s_all.tobytes())
                fout.write(m_all.tobytes())
            print(f"  layer {i:2d} ({'mxfp8' if is_mxfp8 else 'nvfp4'} -> q2_g64) done in {time.time() - tl:.2f}s", flush=True)

    print(f"Finished in {time.time() - t0:.1f}s -> {dst_st}")

if __name__ == "__main__":
    main()
