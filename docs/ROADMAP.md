# Roadmap

Ordered by how much it buys per hour of work. None of these change the public
CLI or the `.vzm` format except where noted.

## Inference speed

Fucina already supplies the big three: a KV-cached decode path, a worker-team
matmul, and arch-specific CPU kernels. What is left here is choosing and
measuring the providers:

1. **Benchmark BLAS providers** — Fucina auto-detects OpenBLAS/MKL/BLIS/NVPL
   and falls back to its Zig kernels. Building with `-Dblas=...` and recording
   per-shape numbers would tell us which provider to recommend for training vs
   serving.
2. **GPU GEMM offload** — Fucina can build with `-Dgpu=metal` (macOS) or
   `-Dgpu=cuda` (Linux/NVIDIA) for large GEMMs. The model is tiny today, so the
   work is mostly wiring the build flag and verifying parity.
3. **Quantized weights** — Fucina has q8_0/q4_K block-quantized RHS kernels. A
   second dtype in `.vzm` (`dtype = 1`) plus the loader would shrink the
   checkpoint and speed up the memory-bound decode shapes.
4. **Larger models** — the current 345k-parameter model fits in L2; the next
   meaningful speed test is a model whose weights stream from RAM.

## Model quality

5. **BPE tokenizer** — 4k merges trained on the corpus, stored in the checkpoint
   (the `vocab` field becomes a merge table + version bump to `.vzm` v2). At
   equal parameter count this is worth far more than char-level.
6. **Multi-file corpus loader** — walk a directory, concatenate with document
   separators, shuffle document order per epoch.
7. **Checkpoint resume** — Fucina's AdamW can `saveState`/`loadState`; persist
   that next to the weights so a 12k-step run can be interrupted without a cold
   start.
8. **Dropout + faster LR schedules** — trivially useful once the model grows
   past ~10M parameters where overfitting starts to bite.

## Engineering

9. **Concurrent server** — a worker thread per connection with a session pool
   instead of the current serial accept loop, plus a mutex per session.
10. **Graceful shutdown + `SIGINT`** — needed to run under a supervisor.
11. **HTTP keep-alive and `Content-Length` on the streaming route** — lets
    reverse proxies (nginx, Cloudflare) buffer predictably.
12. **`llm export --format gguf`** — interop with the wider tooling ecosystem.
