# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
#
# Memory-attribution probe for first-prefill / startup OOM transients.
# Validated recipe + usage: docs/gfx906/DEVLOG-muse-glimmer.md (round 4)
# and the personal skill gfx906-mem-attribution.

"""
Attribution probe for the first-prefill OOM transient (Muse-Glimmer-30B).

Question: who holds the >10.6 GiB/GPU transient that OOMed the first
4096-token prefill chunk under TP=2 serving (boot J 17:40, boot K
18:41; 254 MiB last straw, free: 0, inductor gptq_gemm frame)?
Candidates: our gfx906 FA backend (Q8/gather/prefill buffers), inductor
piecewise-compiled buffers, Exllama gptq_gemm workspace, or model shape.

Arms (one run each, env ARM):
  custom  default attention (all-CUSTOM gfx906 FA), default compilation
  rocm    attention_config pins ROCM_ATTN (Triton FA) — no gfx906 FA
  eager   enforce_eager (no inductor at all), all-CUSTOM FA

Tensor parallelism (env TP, default 1):
  TP=1  in-process model runner -> primary measurement:
        peak transient = max_memory_allocated delta around the first
        prefill, plus a raw memory-snapshot dump for owner analysis.
        Headroom with 0.5 GiB KV cap: ~4.5 GiB
        (31.98 - 25.6 weights - 1.1 non-torch - 0.5 KV - ~0.3 graphs).
  TP=2  matches serving exactly (6 GiB cap, [6,12,18,24] captures,
        10.6 GiB headroom). Workers are separate processes, so the
        result is survive/OOM + last-straw size only (no in-process
        peak).

Design notes:
- The transient scales ~linearly with the batched-token chunk and is
  ~TP-invariant (input-side M*K activations are not split by TP), so a
  TP=1 peak at bt4096 has the same shape as the TP=2 OOM.
- The prompt is PP tokens (default 4097 -> first chunk = 4096, the
  boot OOM site), run cold (no warmup prefill) to match the serving
  sequence.
- GDN (linear-attention) layers use the mamba backend selector, which
  the attention_config pin does not touch — the rocm arm only swaps
  the full/sliding attention layers.

Run (server must be down, GPUs free):
  source ~/env-rocm-7.14-gfx906.sh
  cd /local/git/vllm-gfx906-mobydick
  env HIP_VISIBLE_DEVICES=0,1 FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
      HF_HUB_OFFLINE=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
      ARM=custom TP=1 .venv/bin/python /local/tmp/muse/probe_oom_attribution.py
"""

import json
import os
import sys
import time

SNAP = ("/local/cache/huggingface/hub/"
        "models--cyankiwi--Muse-Glimmer-30B-AWQ-INT4/snapshots/"
        "cba01edf73e0f0f4f013615cc01281ea04e79f85")
ARM = os.environ.get("ARM", "custom")
TP = int(os.environ.get("TP", "1"))
PP = int(os.environ.get("PP", "4097"))
KV_BYTES = int(os.environ.get(
    "KV_BYTES", "6442450944" if TP == 2 else "536870912"))
MAX_LEN = int(os.environ.get("MAX_LEN", "131072" if TP == 2 else "0"))
TAG = f"{ARM}_tp{TP}_pp{PP}"

# --- in-process (thread) compile pool -------------------------------------
# This torch writes async_compile.wait() into every generated wrapper and
# the process pools fail on this box: the default FORK pool is HSA-unsafe
# after the parent inits HSA, and SPAWN children fail to init HSA on
# wedge-accumulated boots (parent HSA works, child doesn't). Compiling in
# parent-process threads sidesteps both and keeps the compile transients
# inside the measured process.
if os.environ.get("PROBE_THREAD_COMPILE", "1") == "1":
    from concurrent.futures import ThreadPoolExecutor
    import torch._inductor.async_compile as _ac

    def _thread_pool():
        if not hasattr(_ac, "_probe_thread_pool"):
            _ac._probe_thread_pool = ThreadPoolExecutor(max_workers=2)
            _ac._pool_set.add(_ac._probe_thread_pool)
        return _ac._probe_thread_pool

    _thread_pool.cache_clear = lambda: None
    _ac.AsyncCompile.process_pool = staticmethod(_thread_pool)
    print("PROBE: thread compile pool active", flush=True)
# --------------------------------------------------------------------------


def _dump_snapdiff(s0, s1, n):
    """Print blocks present in snapshot s1 but not s0 (by address),
    grouped by state, largest first."""
    MiB = 2**20
    b0 = {}
    for s in s0:
        for b in s["blocks"]:
            b0[b["address"]] = (b["state"], b["size"])
    new = []
    for s in s1:
        for b in s["blocks"]:
            if b["address"] not in b0:
                new.append((b["size"], b["state"], b["address"]))
    new.sort(reverse=True)
    tot = sum(x[0] for x in new if x[1] == "active_allocated")
    print(f"PROBE-SNAPDIFF fwd{n}: new blocks={len(new)} "
          f"allocated={tot / MiB:.1f} MiB", flush=True)
    for sz, st, addr in new[:12]:
        print(f"  {sz / MiB:9.1f} MiB  {st}  0x{addr:x}", flush=True)


def main():
    import torch
    from transformers import AutoTokenizer

    # --- bad-fork / driver diagnostics: instrument inductor autotune exec ---
    import os as _os
    import torch._inductor.codegen.wrapper as _w
    _orig_autotune = _w.PythonWrapperCodegen.generate_and_run_autotune_block

    def _diag_autotune(self):
        try:
            bad_fork = torch.cuda._is_in_bad_fork()
            n_dev = torch._C._cuda_getDeviceCount()
        except Exception as e:  # noqa: BLE001
            bad_fork, n_dev = f"err:{e}", "err"
        print(f"PROBE-DIAG: autotune-exec pid={_os.getpid()} ppid={_os.getppid()} "
              f"is_available={torch.cuda.is_available()} "
              f"bad_fork={bad_fork} devcount={n_dev} "
              f"HSA_VISIBLE={_os.environ.get('HSA_VISIBLE_DEVICES')} "
              f"HIP_VISIBLE={_os.environ.get('HIP_VISIBLE_DEVICES')}", flush=True)
        import triton as _tr
        try:
            _bk = sorted(getattr(_tr.backends, "backends", {}).keys())
            _drv = repr(getattr(_tr.runtime.driver, "driver", None))[:80]
        except Exception as e:  # noqa: BLE001
            _bk, _drv = f"err:{e}", "err"
        print(f"PROBE-DIAG: triton backends={_bk} driver={_drv} "
              f"triton_file={getattr(_tr, '__file__', '?')}", flush=True)
        try:
            return _orig_autotune(self)
        except Exception:
            import traceback as _tb2
            print("PROBE-DIAG: autotune-exec FAILED:", flush=True)
            _tb2.print_exc()
            # inner triton state at failure
            try:
                import torch._inductor.runtime.triton_helpers as _th
                import torch as _tc
                for nm, be in _tr.backends.backends.items():
                    print(f"PROBE-DIAG: backend {nm}: active="
                          f"{_th._is_backend_active(nm, be)}", flush=True)
            except Exception as e2:  # noqa: BLE001
                print(f"PROBE-DIAG: backend-state err {e2}", flush=True)
            raise

    _w.PythonWrapperCodegen.generate_and_run_autotune_block = _diag_autotune

    # Trace every fork() in this process (bad-fork is set if one happens
    # after the first CUDA init).
    import traceback as _tb
    _orig_fork = _os.fork
    _cuda_init_seen = [False]

    def _traced_fork():
        print(f"PROBE-DIAG: fork() pid={_os.getpid()} cuda_init_seen={_cuda_init_seen[0]}", flush=True)
        _tb.print_stack(limit=12)
        import sys as _sys
        _sys.stdout.flush()
        return _orig_fork()

    _os.fork = _traced_fork
    _orig_isavail = torch.cuda.is_available

    def _tracked_isavail():
        r = _orig_isavail()
        if r:
            _cuda_init_seen[0] = True
        return r

    torch.cuda.is_available = _tracked_isavail
    # --------------------------------------------------------------------------

    from vllm import LLM, SamplingParams

    extra = {
        "dtype": "float16",
        "max_model_len": MAX_LEN if MAX_LEN else PP + 256,
        "max_num_seqs": 4,
        "kv_cache_memory_bytes": KV_BYTES,
        "max_num_batched_tokens": 4096,
        "seed": 0,
        "tensor_parallel_size": TP,
        "compilation_config": {
            "cudagraph_capture_sizes": [6, 12, 18, 24] if TP == 2
            else [1, 2],
        },
    }
    if ARM == "rocm":
        extra["attention_config"] = {"backend": "ROCM_ATTN"}
    if ARM == "eager":
        extra["enforce_eager"] = True

    print(f"PROBE: arm={ARM} tp={TP} pp={PP} kv={KV_BYTES} "
          f"max_len={extra['max_model_len']}", flush=True)
    t0 = time.time()
    llm = LLM(model=SNAP, **extra)
    print(f"PROBE: init {time.time() - t0:.0f}s", flush=True)

    if os.environ.get("PROBE_PER_LAYER", "0") == "1":
        # Per-attention-call allocated-bytes trajectory through the
        # prefill. Flat = allocator reuses the per-call buffers (the
        # transient holder is upstream); monotone growth = the attention
        # path (or what it feeds) accumulates.
        def _hook(cls_name, cls):
            orig = cls.forward
            st = {"n": 0}

            def wrapped(self, layer, query, key, value, kv_cache,
                        attn_metadata, output=None, **kw):
                n = st["n"]
                st["n"] += 1
                b = torch.cuda.memory_allocated()
                out = orig(self, layer, query, key, value, kv_cache,
                           attn_metadata, output=output, **kw)
                a = torch.cuda.memory_allocated()
                if n < 8 or n % 8 == 0 or a - b > 256 * 2**20:
                    print(f"PROBE-PL {cls_name} call={n} "
                          f"before={b / 2**30:.3f} "
                          f"after={a / 2**30:.3f} "
                          f"delta={(a - b) / 2**20:.1f} MiB "
                          f"out={out.data_ptr() if out is not None else 'None'}",
                          flush=True)
                return out

            cls.forward = wrapped
            print(f"PROBE: per-layer hook on {cls_name}", flush=True)

        try:
            from vllm.gfx906_fa.gfx906_fa_backend import Gfx906FAImpl
            _hook("gfa", Gfx906FAImpl)
        except Exception as e:  # noqa: BLE001
            print(f"PROBE: gfa hook failed: {e!r}", flush=True)
        try:
            from vllm.v1.attention.backends.rocm_attn import (
                RocmAttentionBackend)
            _hook("rocm", RocmAttentionBackend)
        except Exception as e:  # noqa: BLE001
            print(f"PROBE: rocm hook failed: {e!r}", flush=True)

        if ARM == "custom":
            # Isolate which step of the custom FA call nets +256 MiB:
            # the C++ binding vs the Python wrapper around it.
            import vllm._gfx906_fa_C as _ext
            _bnd = {"n": 0}
            _orig_fwd = _ext.forward

            def _fwd_wrapped(*a, **kw):
                n = _bnd["n"]
                _bnd["n"] += 1
                b = torch.cuda.memory_allocated()
                r = torch.cuda.memory_reserved()
                out = _orig_fwd(*a, **kw)
                a2 = torch.cuda.memory_allocated()
                r2 = torch.cuda.memory_reserved()
                if n < 6 or n % 8 == 7 or a2 - b > 128 * 2**20:
                    print(f"PROBE-BND call={n} in={a[0].shape} "
                          f"delta={(a2 - b) / 2**20:.1f} MiB "
                          f"res_d={(r2 - r) / 2**20:.1f} MiB "
                          f"out_ptr={out.data_ptr() if out is not None else '-'}",
                          flush=True)
                return out

            _ext.forward = _fwd_wrapped
            # The backend module imported forward_paged by name? Patch
            # both namespaces.
            import vllm.gfx906_fa.gfx906_fa_paged as _fp
            import vllm.gfx906_fa.gfx906_fa_backend as _be
            _pg = {"n": 0}
            _orig_pg = _fp.forward_paged

            def _pg_wrapped(*a, **kw):
                n = _pg["n"]
                _pg["n"] += 1
                b = torch.cuda.memory_allocated()
                out = _orig_pg(*a, **kw)
                a2 = torch.cuda.memory_allocated()
                if n < 6 or n % 8 == 7 or a2 - b > 128 * 2**20:
                    print(f"PROBE-PGD call={n} delta={(a2 - b) / 2**20:.1f} MiB",
                          flush=True)
                return out

            _fp.forward_paged = _pg_wrapped
            if hasattr(_be, "forward_paged"):
                _be.forward_paged = _pg_wrapped
            print("PROBE: binding+wrapper hooks on", flush=True)

            # Bisect the remaining PL-PGD gap: the _ensure_* calls and a
            # segment-snapshot diff around the first few forwards.
            from vllm.gfx906_fa.gfx906_fa_backend import Gfx906FAImpl
            _ens = {"n": 0}
            _ofb = Gfx906FAImpl._ensure_forward_buffers
            _ogb = Gfx906FAImpl._ensure_gather_buffers

            def _ofb_w(self, **kw):
                n = _ens["n"]
                _ens["n"] += 1
                b = torch.cuda.memory_allocated()
                r = _ofb(self, **kw)
                a2 = torch.cuda.memory_allocated()
                if a2 - b > 8 * 2**20 or n < 3:
                    print(f"PROBE-ENS fwd_bufs n={n} "
                          f"delta={(a2 - b) / 2**20:.1f} MiB", flush=True)
                return r

            def _ogb_w(self, **kw):
                b = torch.cuda.memory_allocated()
                r = _ogb(**kw)  # classmethod: cls already bound
                a2 = torch.cuda.memory_allocated()
                print(f"PROBE-ENS gather_bufs "
                      f"delta={(a2 - b) / 2**20:.1f} MiB", flush=True)
                return r

            Gfx906FAImpl._ensure_forward_buffers = _ofb_w
            Gfx906FAImpl._ensure_gather_buffers = _ogb_w

            if _ens is not None and os.environ.get(
                    "PROBE_SNAPDIFF", "1") == "1":
                _snapdiff = {"done": 0}
                _orig_pl_hook = None  # (PL hook already installed above;
                # we piggyback via a second wrap of _ofb_w? no -- wrap the
                # class forward again, innermost.)
                _inner = Gfx906FAImpl.forward

                def _inner2(self, layer, query, key, value, kv_cache,
                            attn_metadata, output=None, **kw):
                    n = _snapdiff["done"]
                    if n < 3:
                        _snapdiff["done"] += 1
                        torch.cuda.memory._record_memory_history(
                            max_entries=100_000)
                        torch.cuda.synchronize()
                        s0 = torch.cuda.memory_snapshot()
                    out = _inner(self, layer, query, key, value, kv_cache,
                                 attn_metadata, output=output, **kw)
                    if n < 3:
                        torch.cuda.synchronize()
                        s1 = torch.cuda.memory_snapshot()
                        torch.cuda.memory._record_memory_history(
                            enabled=False)
                        _dump_snapdiff(s0, s1, n)
                    return out

                Gfx906FAImpl.forward = _inner2
                print("PROBE: snapdiff hooks on", flush=True)

    tok = AutoTokenizer.from_pretrained(SNAP)
    filler = "The quick brown fox jumps over the lazy dog. "
    p, t = f"[probe {TAG}] ", tok.encode(f"[probe {TAG}] ")
    while len(t) < PP:
        p += filler
        t = tok.encode(p)
    p = tok.decode(t[:PP])
    print(f"PROBE: prompt tokens={len(tok.encode(p))}", flush=True)

    inproc = TP == 1
    if inproc:
        torch.cuda.synchronize()
        base_alloc = torch.cuda.memory_allocated()
        free_before, total = torch.cuda.mem_get_info()
        torch.cuda.reset_peak_memory_stats()
        print(f"PROBE: pre-generate: allocated={base_alloc / 2**30:.3f} "
              f"GiB, free={free_before / 2**30:.2f} GiB / "
              f"total={total / 2**30:.2f} GiB", flush=True)

    if inproc and os.environ.get("PROBE_MEM_HISTORY", "1") == "1":
        # Record allocation history BEFORE the prefill so the post-hoc
        # snapshot carries per-block allocation stacks (the transient
        # blocks are all allocated during the prefill).
        torch.cuda.memory._record_memory_history(max_entries=10_000_000)
        print("PROBE: memory history recording started", flush=True)

    t0 = time.time()
    survived = True
    oom_msg = ""
    if inproc and os.environ.get("PROBE_KINETO", "0") == "1":
        from torch.profiler import ProfilerActivity, profile as tprofile
        with tprofile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                with_stack=False,
        ) as prof:
            try:
                llm.generate([p], SamplingParams(temperature=0.0,
                                                 max_tokens=16),
                             use_tqdm=False)
                print(f"PROBE: SURVIVED pp={PP} in {time.time() - t0:.1f}s",
                      flush=True)
            except Exception as e:  # noqa: BLE001
                survived = False
                oom_msg = str(e)
                print(f"PROBE: FAILED after {time.time() - t0:.1f}s: "
                      f"{type(e).__name__}: {oom_msg[:400]}", flush=True)
        trace = f"/local/tmp/muse/kineto_{TAG}.json"
        prof.export_chrome_trace(trace)
        print(f"PROBE: kineto trace -> {trace}", flush=True)
    else:
        try:
            llm.generate([p], SamplingParams(temperature=0.0, max_tokens=16),
                         use_tqdm=False)
            print(f"PROBE: SURVIVED pp={PP} in {time.time() - t0:.1f}s",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            survived = False
            oom_msg = str(e)
            print(f"PROBE: FAILED after {time.time() - t0:.1f}s: "
                  f"{type(e).__name__}: {oom_msg[:400]}", flush=True)

    if inproc:
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        free_after, _ = torch.cuda.mem_get_info()
        print(f"PROBE: peak_transient={(peak - base_alloc) / 2**30:.3f} "
              f"GiB (allocated {base_alloc / 2**30:.3f} -> peak "
              f"{peak / 2**30:.3f}); free now {free_after / 2**30:.2f} "
              f"GiB; headroom was {free_before / 2**30:.2f} GiB; "
              f"survived={survived}", flush=True)
        try:
            torch.cuda.synchronize()
            snap = torch.cuda.memory_snapshot()
            out = f"/local/tmp/muse/oom_snap_{TAG}.json"
            with open(out, "w") as f:
                json.dump(snap, f)
            print(f"PROBE: snapshot -> {out} "
                  f"(top-level {type(snap).__name__})", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"PROBE: snapshot failed: {e!r}", flush=True)

    print(f"PROBE: RESULT arm={ARM} tp={TP} survived={survived}",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
