#![allow(unsafe_op_in_unsafe_fn)]
//! Custom CPU SDPA (scaled dot-product attention) for the decode hot
//! path. Designed for Llama-3.2-1B inference on x86_64 CPU.
//!
//! ## Why this exists
//!
//! Candle's default attention path for CPU does
//! 1. `Tensor::cat` to grow the KV cache (allocates+copies the whole
//!    history every token);
//! 2. `repeat_kv` to broadcast K/V from `n_kv_head` to `n_head`
//!    (4× memory blowup on Llama-3.2-1B which uses GQA);
//! 3. `q.matmul(k.t())` and `att.matmul(v.contiguous())` (each forces
//!    its operands contiguous in fresh allocations).
//!
//! Profiling on 2026-06-10 showed those copies account for ~10 ms/token
//! out of 30 ms total — bigger than all matmuls combined. This kernel
//! sidesteps every copy:
//! - reads the strided KV view directly via `storage_and_layout()`;
//! - indexes the broadcast (`kv_head = q_head / (n_head / n_kv_head)`)
//!   instead of materializing it;
//! - writes a fresh `(1, n_head, 1, head_dim)` contiguous output.
//!
//! Decode-only (seq_len = 1). Prefill stays on the existing matmul
//! path — its arithmetic intensity is high enough that the copies
//! amortize.

use candle_core::{Result, Storage, Tensor, WithDType};
#[cfg(test)]
use candle_core::Device;

/// Compute `softmax(Q·Kᵀ / √d) · V` with GQA: each Q head reads from
/// the KV head at index `q_head / (n_head / n_kv_head)`.
///
/// Shapes (all f32, CPU):
/// - `q`:  `(1, n_head, 1, head_dim)`, may or may not be contiguous;
/// - `k`:  `(1, n_kv_head, current_len, head_dim)`, strided (typically
///   a `narrow` view into a `KvCache` buffer);
/// - `v`:  same shape and stride pattern as `k`;
///
/// Returns a fresh contiguous `(1, 1, n_head * head_dim)` tensor —
/// i.e. heads already concatenated. This skips the
/// `transpose(1, 2).reshape(...)` copy that the caller would otherwise
/// need to merge heads. The memory layout for `(n_head, head_dim)` is
/// already the concat-heads layout we want.
///
/// No mask is applied — decode attends to every position including
/// itself, which matches causal attention semantics for a single-token
/// step.
pub fn sdpa_gqa_decode(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    current_len: usize,
) -> Result<Tensor> {
    debug_assert_eq!(q.dims(), &[1, n_head, 1, head_dim]);
    debug_assert_eq!(k.dims(), &[1, n_kv_head, current_len, head_dim]);
    debug_assert_eq!(v.dims(), &[1, n_kv_head, current_len, head_dim]);
    debug_assert_eq!(n_head % n_kv_head, 0);
    let kv_repeat = n_head / n_kv_head;
    let inv_sqrt_d = 1.0_f32 / (head_dim as f32).sqrt();

    let (qs, ql) = q.storage_and_layout();
    let (ks, kl) = k.storage_and_layout();
    let (vs, vl) = v.storage_and_layout();
    let qd = cpu_f32(&qs)?;
    let kd = cpu_f32(&ks)?;
    let vd = cpu_f32(&vs)?;

    let q_off = ql.start_offset();
    let q_str = ql.stride();
    let k_off = kl.start_offset();
    let k_str = kl.stride();
    let v_off = vl.start_offset();
    let v_str = vl.stride();

    let q_head_s = q_str[1];
    let k_head_s = k_str[1];
    let k_time_s = k_str[2];
    let v_head_s = v_str[1];
    let v_time_s = v_str[2];

    // Assume innermost dim is unit-strided (asserted in debug builds).
    // True for contiguous q and for KvCache buffer narrows on dim 2.
    debug_assert_eq!(q_str[3], 1);
    debug_assert_eq!(k_str[3], 1);
    debug_assert_eq!(v_str[3], 1);

    let mut out = vec![0.0_f32; n_head * head_dim];
    let mut scores = vec![0.0_f32; current_len];

    #[cfg(target_arch = "x86_64")]
    {
        if head_dim % 16 == 0 && head_dim <= 128 && std::is_x86_feature_detected!("avx512f") {
            unsafe {
                sdpa_inner_avx512(
                    qd, q_off, q_head_s,
                    kd, k_off, k_head_s, k_time_s,
                    vd, v_off, v_head_s, v_time_s,
                    n_head, kv_repeat, head_dim, current_len, inv_sqrt_d,
                    &mut out, &mut scores,
                );
            }
            drop(qs); drop(ks); drop(vs);
            return Tensor::from_vec(out, (1, 1, n_head * head_dim), q.device());
        }
    }

    sdpa_inner_scalar(
        qd, q_off, q_head_s,
        kd, k_off, k_head_s, k_time_s,
        vd, v_off, v_head_s, v_time_s,
        n_head, kv_repeat, head_dim, current_len, inv_sqrt_d,
        &mut out, &mut scores,
    );

    drop(qs); drop(ks); drop(vs);
    Tensor::from_vec(out, (1, 1, n_head * head_dim), q.device())
}

#[inline]
fn sdpa_inner_scalar(
    qd: &[f32], q_off: usize, q_head_s: usize,
    kd: &[f32], k_off: usize, k_head_s: usize, k_time_s: usize,
    vd: &[f32], v_off: usize, v_head_s: usize, v_time_s: usize,
    n_head: usize, kv_repeat: usize, head_dim: usize, current_len: usize,
    inv_sqrt_d: f32,
    out: &mut [f32], scores: &mut [f32],
) {
    for h in 0..n_head {
        let kv_h = h / kv_repeat;
        let q_base = q_off + h * q_head_s;
        let k_base = k_off + kv_h * k_head_s;
        let v_base = v_off + kv_h * v_head_s;

        let mut max_s = f32::NEG_INFINITY;
        for t in 0..current_len {
            let k_t = k_base + t * k_time_s;
            let mut s = 0.0_f32;
            for d in 0..head_dim {
                s += qd[q_base + d] * kd[k_t + d];
            }
            s *= inv_sqrt_d;
            scores[t] = s;
            if s > max_s { max_s = s; }
        }

        let mut sum_exp = 0.0_f32;
        for s in scores[..current_len].iter_mut() {
            *s = (*s - max_s).exp();
            sum_exp += *s;
        }
        let inv_sum = 1.0_f32 / sum_exp;

        let out_base = h * head_dim;
        for d in 0..head_dim { out[out_base + d] = 0.0; }
        for t in 0..current_len {
            let w = scores[t] * inv_sum;
            let v_t = v_base + t * v_time_s;
            for d in 0..head_dim {
                out[out_base + d] += w * vd[v_t + d];
            }
        }
    }
}

/// AVX-512 fast path. Hoists Q into 4 ZMM registers (head_dim=64 case)
/// and keeps the output accumulator in registers across the V-walk —
/// each weighted V[t] add is a single FMA on each register.
///
/// Numerical caveat: dot product order differs from scalar (16 lanes
/// accumulate independently, then horizontal reduce). Results match
/// scalar within ~1e-4 on Llama-scale inputs — tied with the existing
/// scalar-vs-reference tolerance.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn sdpa_inner_avx512(
    qd: &[f32], q_off: usize, q_head_s: usize,
    kd: &[f32], k_off: usize, k_head_s: usize, k_time_s: usize,
    vd: &[f32], v_off: usize, v_head_s: usize, v_time_s: usize,
    n_head: usize, kv_repeat: usize, head_dim: usize, current_len: usize,
    inv_sqrt_d: f32,
    out: &mut [f32], scores: &mut [f32],
) {
    use std::arch::x86_64::*;
    const LANES: usize = 16;
    debug_assert_eq!(head_dim % LANES, 0);
    let n_vec = head_dim / LANES;
    // Cap at 8 to keep the register array bounded; head_dim=128
    // (Llama-2 7B etc.) → 8 vecs. Dispatch routes head_dim > 128 to the
    // scalar path, which handles any width.
    assert!(n_vec <= 8, "head_dim {} > 128 must go to the scalar path", head_dim);

    let qp = qd.as_ptr();
    let kp = kd.as_ptr();
    let vp = vd.as_ptr();
    let op = out.as_mut_ptr();

    for h in 0..n_head {
        let kv_h = h / kv_repeat;
        let q_base = q_off + h * q_head_s;
        let k_base = k_off + kv_h * k_head_s;
        let v_base = v_off + kv_h * v_head_s;

        // Hoist Q[h] into ZMM registers.
        let mut qreg = [_mm512_setzero_ps(); 8];
        for i in 0..n_vec {
            qreg[i] = _mm512_loadu_ps(qp.add(q_base + i * LANES));
        }

        // Scores walk: dot(Q[h], K[kv_h, t]) → scalar.
        let mut max_s = f32::NEG_INFINITY;
        for t in 0..current_len {
            let k_t = k_base + t * k_time_s;
            let mut acc = _mm512_setzero_ps();
            for i in 0..n_vec {
                let kv = _mm512_loadu_ps(kp.add(k_t + i * LANES));
                acc = _mm512_fmadd_ps(qreg[i], kv, acc);
            }
            let s = _mm512_reduce_add_ps(acc) * inv_sqrt_d;
            *scores.get_unchecked_mut(t) = s;
            if s > max_s { max_s = s; }
        }

        // Softmax (scalar — current_len calls to exp, small fraction of work).
        let mut sum_exp = 0.0_f32;
        for t in 0..current_len {
            let v = (*scores.get_unchecked(t) - max_s).exp();
            *scores.get_unchecked_mut(t) = v;
            sum_exp += v;
        }
        let inv_sum = 1.0_f32 / sum_exp;

        // Weighted V sum — output accumulator stays in registers.
        let mut oreg = [_mm512_setzero_ps(); 8];
        for t in 0..current_len {
            let w = _mm512_set1_ps(*scores.get_unchecked(t) * inv_sum);
            let v_t = v_base + t * v_time_s;
            for i in 0..n_vec {
                let vv = _mm512_loadu_ps(vp.add(v_t + i * LANES));
                oreg[i] = _mm512_fmadd_ps(w, vv, oreg[i]);
            }
        }
        let out_base = h * head_dim;
        for i in 0..n_vec {
            _mm512_storeu_ps(op.add(out_base + i * LANES), oreg[i]);
        }
    }
}

fn cpu_f32<'a>(s: &'a Storage) -> Result<&'a [f32]> {
    match s {
        Storage::Cpu(cs) => f32::cpu_storage_as_slice(cs),
        _ => Err(candle_core::Error::Msg(
            "sdpa_gqa_decode only supports CPU storage".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Equivalence test: our SDPA vs a "reference" implementation that
    /// goes through Tensor ops the same way Candle's MLP/attn does.
    /// Pass if max abs error < 1e-4 across all output elements.
    #[test]
    fn matches_reference_attention() {
        let dev = Device::Cpu;
        let n_head = 8;
        let n_kv_head = 2;
        let head_dim = 16;
        let current_len = 5;
        let inv_sqrt_d = 1.0_f32 / (head_dim as f32).sqrt();

        // Deterministic input data.
        let mk = |seed: u64, shape: &[usize]| -> Tensor {
            let n: usize = shape.iter().product();
            let mut x = vec![0.0_f32; n];
            for i in 0..n {
                let v = ((i as u64).wrapping_mul(seed).wrapping_add(1234)) & 0xff;
                x[i] = (v as f32 / 128.0) - 1.0;
            }
            Tensor::from_vec(x, shape.to_vec(), &dev).unwrap()
        };
        let q = mk(7, &[1, n_head, 1, head_dim]);
        let k_full = mk(11, &[1, n_kv_head, current_len, head_dim]);
        let v_full = mk(13, &[1, n_kv_head, current_len, head_dim]);

        let our = sdpa_gqa_decode(
            &q, &k_full, &v_full, n_head, n_kv_head, head_dim, current_len,
        ).unwrap();
        let our_vec = our.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        // Reference: expand K, V across query heads, then standard SDPA.
        let kv_repeat = n_head / n_kv_head;
        let k_rep = candle_transformers::utils::repeat_kv(k_full.clone(), kv_repeat).unwrap();
        let v_rep = candle_transformers::utils::repeat_kv(v_full.clone(), kv_repeat).unwrap();
        let att = q.matmul(&k_rep.t().unwrap()).unwrap();
        let att = (att * inv_sqrt_d as f64).unwrap();
        let att = candle_nn::ops::softmax_last_dim(&att).unwrap();
        let ref_out = att.matmul(&v_rep.contiguous().unwrap()).unwrap();
        let ref_vec = ref_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        assert_eq!(our_vec.len(), ref_vec.len());
        let mut max_err = 0.0_f32;
        for (a, b) in our_vec.iter().zip(ref_vec.iter()) {
            let e = (a - b).abs();
            if e > max_err { max_err = e; }
        }
        assert!(max_err < 1e-4, "max error {} too high", max_err);
    }

    /// Llama-3.2-1B shapes: n_head=32, n_kv_head=8, head_dim=64.
    /// Exercises AVX-512 path with 4 ZMM vectors per dot product.
    #[test]
    fn matches_reference_llama_shapes() {
        let dev = Device::Cpu;
        let n_head = 32;
        let n_kv_head = 8;
        let head_dim = 64;
        let current_len = 17;
        let inv_sqrt_d = 1.0_f32 / (head_dim as f32).sqrt();

        let mk = |seed: u64, shape: &[usize]| -> Tensor {
            let n: usize = shape.iter().product();
            let mut x = vec![0.0_f32; n];
            for i in 0..n {
                let v = ((i as u64).wrapping_mul(seed).wrapping_add(1234)) & 0xff;
                x[i] = (v as f32 / 128.0) - 1.0;
            }
            Tensor::from_vec(x, shape.to_vec(), &dev).unwrap()
        };
        let q = mk(7, &[1, n_head, 1, head_dim]);
        let k = mk(11, &[1, n_kv_head, current_len, head_dim]);
        let v = mk(13, &[1, n_kv_head, current_len, head_dim]);

        let our = sdpa_gqa_decode(
            &q, &k, &v, n_head, n_kv_head, head_dim, current_len,
        ).unwrap();
        let our_vec = our.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let kv_repeat = n_head / n_kv_head;
        let k_rep = candle_transformers::utils::repeat_kv(k.clone(), kv_repeat).unwrap();
        let v_rep = candle_transformers::utils::repeat_kv(v.clone(), kv_repeat).unwrap();
        let att = q.matmul(&k_rep.t().unwrap()).unwrap();
        let att = (att * inv_sqrt_d as f64).unwrap();
        let att = candle_nn::ops::softmax_last_dim(&att).unwrap();
        let ref_out = att.matmul(&v_rep.contiguous().unwrap()).unwrap();
        let ref_vec = ref_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        assert_eq!(our_vec.len(), ref_vec.len());
        let mut max_err = 0.0_f32;
        for (a, b) in our_vec.iter().zip(ref_vec.iter()) {
            let e = (a - b).abs();
            if e > max_err { max_err = e; }
        }
        assert!(max_err < 1e-4,
            "Llama-shape max error {} too high", max_err);
    }

    /// head_dim=256 (Gemma-class) is 16-aligned but exceeds the AVX-512
    /// kernel's 8-register cap — dispatch must route it to the scalar
    /// path instead of panicking.
    #[test]
    fn matches_reference_head_dim_over_128() {
        let dev = Device::Cpu;
        let n_head = 4;
        let n_kv_head = 2;
        let head_dim = 256;
        let current_len = 9;
        let inv_sqrt_d = 1.0_f32 / (head_dim as f32).sqrt();

        let mk = |seed: u64, shape: &[usize]| -> Tensor {
            let n: usize = shape.iter().product();
            let mut x = vec![0.0_f32; n];
            for i in 0..n {
                let v = ((i as u64).wrapping_mul(seed).wrapping_add(1234)) & 0xff;
                x[i] = (v as f32 / 128.0) - 1.0;
            }
            Tensor::from_vec(x, shape.to_vec(), &dev).unwrap()
        };
        let q = mk(7, &[1, n_head, 1, head_dim]);
        let k = mk(11, &[1, n_kv_head, current_len, head_dim]);
        let v = mk(13, &[1, n_kv_head, current_len, head_dim]);

        let our = sdpa_gqa_decode(
            &q, &k, &v, n_head, n_kv_head, head_dim, current_len,
        ).unwrap();
        let our_vec = our.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let kv_repeat = n_head / n_kv_head;
        let k_rep = candle_transformers::utils::repeat_kv(k.clone(), kv_repeat).unwrap();
        let v_rep = candle_transformers::utils::repeat_kv(v.clone(), kv_repeat).unwrap();
        let att = q.matmul(&k_rep.t().unwrap()).unwrap();
        let att = (att * inv_sqrt_d as f64).unwrap();
        let att = candle_nn::ops::softmax_last_dim(&att).unwrap();
        let ref_out = att.matmul(&v_rep.contiguous().unwrap()).unwrap();
        let ref_vec = ref_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        assert_eq!(our_vec.len(), ref_vec.len());
        let mut max_err = 0.0_f32;
        for (a, b) in our_vec.iter().zip(ref_vec.iter()) {
            let e = (a - b).abs();
            if e > max_err { max_err = e; }
        }
        assert!(max_err < 1e-4,
            "head_dim=256 max error {} too high", max_err);
    }

    /// Walks a strided narrow view (simulating what KvCache returns).
    #[test]
    fn works_on_strided_kv_narrow() {
        let dev = Device::Cpu;
        let n_head = 4;
        let n_kv_head = 2;
        let head_dim = 8;
        let max_len = 16;
        let current_len = 3;

        let mk = |seed: u64, shape: &[usize]| -> Tensor {
            let n: usize = shape.iter().product();
            let mut x = vec![0.0_f32; n];
            for i in 0..n {
                let v = ((i as u64).wrapping_mul(seed).wrapping_add(99)) & 0xff;
                x[i] = (v as f32 / 128.0) - 1.0;
            }
            Tensor::from_vec(x, shape.to_vec(), &dev).unwrap()
        };
        let q = mk(2, &[1, n_head, 1, head_dim]);
        // Allocate a buffer-sized K/V, then narrow it as KvCache would.
        let k_buf = mk(3, &[1, n_kv_head, max_len, head_dim]);
        let v_buf = mk(5, &[1, n_kv_head, max_len, head_dim]);
        let k_view = k_buf.narrow(2, 0, current_len).unwrap();
        let v_view = v_buf.narrow(2, 0, current_len).unwrap();
        assert!(!k_view.is_contiguous(), "narrow on dim 2 must be strided");

        let our = sdpa_gqa_decode(
            &q, &k_view, &v_view, n_head, n_kv_head, head_dim, current_len,
        ).unwrap();

        // Reference: same op on a contiguous copy of the view.
        let k_c = k_view.contiguous().unwrap();
        let v_c = v_view.contiguous().unwrap();
        let ref_out = sdpa_gqa_decode(
            &q, &k_c, &v_c, n_head, n_kv_head, head_dim, current_len,
        ).unwrap();

        let a = our.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = ref_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let mut max_err = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            let e = (x - y).abs();
            if e > max_err { max_err = e; }
        }
        assert!(max_err < 1e-5,
            "strided vs contiguous mismatch, max err {}", max_err);
    }
}
