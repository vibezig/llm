# vibezig/llm

A **fullstack LLM engine written in Zig**: tensor library, transformer training
from scratch, inference, and an HTTP server with a web chat UI — all in one
5 MB static binary with **zero dependencies**.

No PyTorch. No Python. No GPU. The model is trained and served by the same Zig
code that defines it.

```
$ llm train --steps 12000        # learn Shakespeare from characters up
$ llm infer --prompt "ROMEO:"    # sample text, streaming
$ llm serve                      # http://localhost:8080 — chat UI + JSON API
```

Real output from a 12,000-step checkpoint (`--temp 0.8 --top-k 40`):

```
ROMEO:
Ah, all the come of your son,
As are is my such comes repent.

ESCALUS:
Hill I father, they dareless the crown:
I am do tell me such for the gold friend:
'Tis't I'll say to her hath dead wash of this advise,
So to Julie
```

It is a 345k-parameter model trained for 12,000 steps on a 1 MB corpus — it
learned the *shape* of Shakespeare (names, line breaks, verse rhythm, real words),
not the meaning of it.

---

## What's in here

| layer | file | what it does |
| --- | --- | --- |
| tensor math | `src/tensor.zig` | f32 buffers, 4×8 blocked GEMM, layernorm, GELU, softmax, cross-entropy |
| checkpoint format | `src/model.zig` | config + tensor registry + weights in one flat blob (`.vzm`) |
| tokenizer | `src/tokenizer.zig` | byte-level char tokenizer, vocab travels with the model |
| transformer | `src/gpt.zig` | forward, hand-written backward, AdamW, sampling |
| CLI | `src/main.zig` | `gen` / `train` / `infer` / `info` / `serve` |
| server | `src/server.zig` | HTTP/1.1 API + embedded UI (SSE token stream) |
| web UI | `src/web/index.html` | single-file chat client, compiled into the binary |

## Quickstart

Requires **Zig 0.17.1** (nothing else).

```bash
git clone https://github.com/vibezig/llm && cd llm

zig build --release=fast        # build zig-out/bin/llm
zig build test                  # 24 tests, incl. finite-difference gradient checks

./zig-out/bin/llm gen           # create random tensors -> models/tiny.vzm
./zig-out/bin/llm train         # train on data/input.txt (TinyShakespeare)
./zig-out/bin/llm infer --prompt "ROMEO:" --tokens 300
./zig-out/bin/llm serve         # then open http://localhost:8080
```

`zig build` alone gives you a debug binary; `--release=fast` is what you want for
training.

**Models are build outputs, not repo contents.** `models/` is git-ignored: `llm gen`
writes a fresh random tensor model, `llm train` replaces it with a trained one.
Nothing in the repo depends on a checkpoint being present.

## CLI

```
llm gen   [--out models/tiny.vzm] [--layers 3] [--heads 4] [--embd 96]
          [--block 64] [--data data/input.txt] [--seed 1337]

llm train [--data data/input.txt] [--out models/tiny.vzm] [--model in.vzm]
          [--steps 12000] [--batch 8] [--lr 0.003] [--wd 0.01]
          [--eval-every 500] [--sample-every 1000] [--seed 1337]

llm infer [--model models/tiny.vzm] [--prompt "ROMEO:"] [--tokens 200]
          [--temp 0.8] [--top-k 40] [--seed 0]

llm serve [--model models/tiny.vzm] [--port 8080]
llm info  [--model models/tiny.vzm]
```

`gen` writes a fresh, randomly-initialised tensor model — the same file format
`train` and `infer` use, so a checkpoint is just "tensors someone trained".
`info` prints the config and the tensor registry:

```
model: models/tiny.vzm
  params   345504
  layers   3
  heads    4 (head dim 24)
  embd     96
  block    64
  vocab    65
  tensors  34
  tensors:
    wte                [65, 96]
    wpe                [64, 96]
    h0.ln1.g           [96, 1]
    h0.attn.wq         [96, 96]
    ...
```

## The model

A GPT-2-shaped decoder with pre-LN blocks and a tied output head:

```
tokens ──► wte[idx] + wpe[pos]
             │
             ▼  ×n_layer
    ┌─────────────────────────────┐
    │ x = x + attn(ln1(x))        │   causal multi-head self-attention
    │ x = x + mlp(ln2(x))         │   GELU, 4× hidden expansion
    └─────────────────────────────┘
             │
             ▼
        ln_f ──► logits = x @ wteᵀ ──► softmax ──► next byte
```

Default `tiny` config: 3 layers, 4 heads, 96 dims, 64-token context, 65-byte
vocab → **345,504 parameters** (1.4 MB). Bigger shapes are one flag away:

```bash
./zig-out/bin/llm gen  --layers 6 --heads 6 --embd 192 --block 128 --out models/medium.vzm
./zig-out/bin/llm train --model models/medium.vzm --steps 20000 --batch 16
```

### Backprop is written out by hand

There is no autograd tape. `Session.fwdBwd` keeps an explicit activation cache
(per-layer norm statistics, q/k/v, attention probabilities, GELU inputs) and
chains the analytic gradients through it, including:

* the softmax Jacobian inside scaled dot-product attention, with the causal mask,
* layernorm backward with the mean/variance correction terms,
* weight-tied gradients (the embedding matrix receives both the lookup and the
  output-projection contribution),
* residual additions folded into a single `dx`.

That is easy to get subtly wrong, so the test suite checks every gradient bucket
(embedding, attention, MLP, layernorm) against central finite differences:

```
test "gradients match finite differences" ... OK
```

Training loop: AdamW (β₁=0.9, β₂=0.99), decoupled weight decay 0.01, global
gradient-norm clipping at 1.0, 5% linear warmup then cosine decay to 10% of peak
LR, and a 95/5 train/validation split of the corpus.

## Checkpoint format (`.vzm`)

One flat `f32` blob plus a named registry, so parameters, gradients and optimizer
state all share offsets — `grad[i]` always belongs to `param[i]`.

| field | type | notes |
| --- | --- | --- |
| magic | 4 bytes | `"VZLM"` |
| version | u32 | `1` |
| n_layer, n_head, n_embd, block_size, vocab_size | 5 × u32 | model shape |
| vocab_len, vocab | u32 + bytes | tokenizer is part of the model |
| n_tensors | u32 | registry length |
| per tensor: name_len, name, rows, cols, dtype | u32, bytes, u32, u32, u32 | dtype `0` = f32 |
| blob | `total_floats × f32` | row-major, little-endian |

Everything is written by `model.save()` and validated on load (bad magic,
truncated file, unknown dtype and shape mismatches are all rejected — see the
`deserialize rejects a corrupted magic` test).

## HTTP API

`llm serve` starts a hand-rolled HTTP/1.1 server on `std.Io.net` — no framework,
no middleware, ~300 lines including JSON escaping.

| route | description |
| --- | --- |
| `GET /` | the chat UI (embedded in the binary) |
| `GET /api/health` | `{"ok":true}` |
| `GET /api/model` | config + parameter count |
| `GET /api/generate?prompt=&tokens=&temp=&top_k=&seed=` | blocking JSON completion |
| `POST /api/generate` | same, JSON body |
| `GET /api/stream?...` | server-sent events, one byte per event |

```bash
# blocking
curl -s 'http://localhost:8080/api/generate?prompt=ROMEO:&tokens=120&temp=0.8'

# JSON body
curl -s -X POST http://localhost:8080/api/generate \
  -d '{"prompt":"JULIET:","tokens":64,"temperature":0.6,"top_k":20}'

# streaming (one SSE event per generated byte)
curl -N 'http://localhost:8080/api/stream?prompt=ROMEO:&tokens=64'
data: {"event":"start"}
data: {"b":82}
data: {"b":79}
...
data: {"event":"done"}
```

Streaming sends **bytes**, not JS strings: the browser feeds them through a
streaming `TextDecoder`, so multi-byte UTF-8 survives being split across tokens.

## Performance

Measured on a 2 vCPU ARM (Graviton) box, single-threaded, `--release=fast`, no
SIMD intrinsics beyond what Zig/LLVM vectorises:

| workload | throughput |
| --- | --- |
| training (345k params, batch 8 × seq 64) | 1,906 tok/s (~1.1 GFLOP/step) |
| inference, short context | ~650 tok/s |
| inference, 200 tokens of context | ~160 tok/s (full recompute, no KV cache) |

That checkpoint is 12,000 steps = 6.1M tokens ≈ 5.5 epochs in **54 minutes**
(same box): train loss 4.20 → 1.35, held-out val loss **1.78**.

The GEMM kernels are 4×8 register-blocked: four rows of `A` share one vector of
`B`, which cuts `B` traffic 4× and keeps the FMA loop in registers. That alone
was worth ~1.7× over the naive axpy loop. `matmulTnAcc` (used for the six weight
gradients per layer) uses the same blocking against activation rows.

## Limits (on purpose)

This is a readable reference implementation, not a production inference stack:

* **char-level tokenizer** — one token per byte; no BPE yet.
* **no KV cache** — generation re-runs the whole context each step, O(T²) per token.
* **single-threaded** — no thread pool over row blocks yet.
* **f32 only** — no quantisation, no bf16/f16.
* **one request at a time** — the server serialises generations behind the model.
* **CPU only** — no CUDA/Metal backend.

Each of those is a contained piece of work; see `docs/ROADMAP.md`.

## Tests

```bash
zig build test
```

24 tests: GEMM vs naive reference, layernorm statistics and backward vs finite
differences, softmax/CE gradients, `.vzm` round-trip and corruption rejection,
full-model gradient checks, backward determinism (scratch-buffer leaks), "one
AdamW step reduces loss", sampling length/range, query decoding and JSON
escaping.

## License

MIT.
