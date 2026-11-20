# Roadmap

Ordered by how much it buys per hour of work. None of these change the public
CLI or the `.vzm` format except where noted.

## Inference speed

1. **KV cache** — currently every generated token re-runs the full context
   (O(T²) attention work per token). Caching K/V per layer per position makes
   generation linear in context length and is the single biggest win: expect
   5–20× at 200+ tokens of context. Touches `Session.forward` (skip the qkv
   projection for already-seen positions) and the attention loop indexing.
2. **Row-block threads** — `matmul` splits cleanly along `m`; a small pool over
   `std.Thread` with no synchronisation (each thread owns its rows) gives close
   to linear scaling on more cores.
3. **Int8 / Q8_0 weights** — the blob is already a flat f32 array with a tensor
   registry, so a second dtype in `.vzm` (`dtype = 1`) plus a dequantising GEMM
   is a contained change. Meaningful size and speed win on ARM NEON.
4. **AVX2/NEON-specific kernels** — replace `@Vector(8, f32)` blocks with
   arch-dispatched intrinsics (`@import("builtin").cpu`) once profiling says the
   generic kernel is the ceiling.

## Model quality

5. **BPE tokenizer** — 4k merges trained on the corpus, stored in the checkpoint
   (the `vocab` field becomes a merge table + version bump to `.vzm` v2). At
   equal parameter count this is worth far more than char-level.
6. **Multi-file corpus loader** — walk a directory, concatenate with document
   separators, shuffle document order per epoch.
7. **Checkpoint resume** — serialise AdamW moments (`opt.m`, `opt.v`, `t`) next
   to the weights so a 12k-step run can be interrupted without a cold start.
8. **Dropout + faster LR schedules** — trivially useful once the model grows
   past ~10M parameters where overfitting starts to bite.

## Engineering

9. **Concurrent server** — a worker thread per connection with a session pool
   instead of the current serial accept loop, plus a mutex per session.
10. **Graceful shutdown + `SIGINT`** — needed to run under a supervisor.
11. **HTTP keep-alive and `Content-Length` on the streaming route** — lets
    reverse proxies (nginx, Cloudflare) buffer predictably.
12. **`llm export --format gguf`** — interop with the wider tooling ecosystem.
