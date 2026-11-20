# vibezig/llm

A **fullstack LLM engine written in Zig**: transformer training from scratch,
inference, and an HTTP server with a web chat UI — all in one static binary.

The tensor math, autograd and AdamW come from [Fucina](https://github.com/matteo-grella/fucina),
a CPU-first tensor/autograd runtime fetched by the Zig build. No PyTorch. No
Python. No GPU. The model is trained and served by the same Zig code that
defines it.

```
$ llm train --steps 12000        # learn Shakespeare from BPE tokens up
$ llm infer --prompt "ROMEO:"    # sample text, streaming
$ llm serve                      # http://localhost:8080 — chat UI + JSON API
```

The tokenizer is a **byte-level BPE trained on the corpus itself** (`--vocab`,
default 1,024 tokens), not a downloaded vocabulary. The transformer is a
modern dense block — RMSNorm, RoPE, GQA and SwiGLU — at **403k parameters**,
and the tokenizer blob travels inside the `.vzm` checkpoint.

---

## What's in here

| layer | file | what it does |
| --- | --- | --- |
| tensor math | `fucina` 0.5.1 | tagged f32 tensors, CPU SIMD kernels, eager autograd, AdamW |
| checkpoint format | `src/model.zig` | config + tensor registry + weights in one flat blob (`.vzm`) |
| tokenizer | `src/tokenizer.zig`, `src/bpe/` | corpus-trained byte-level BPE (vendored bpe.zig) |
| transformer | `src/gpt.zig` | RMSNorm + RoPE + GQA + SwiGLU; KV-cached sampling |
| CLI | `src/main.zig` | `gen` / `train` / `infer` / `info` / `serve` |
| server | `src/server.zig` | HTTP/1.1 API + embedded UI (SSE token stream) |
| web UI | `src/web/index.html` | single-file chat client, compiled into the binary |

## Quickstart

Requires **Zig 0.16.0**. The first build fetches Fucina 0.5.1 (MIT) from
`build.zig.zon`; there are no system libraries to install.

```bash
git clone https://github.com/vibezig/llm && cd llm

zig build --release=fast        # build zig-out/bin/llm
zig build test                  # 27 tests, incl. finite-difference gradient checks

./zig-out/bin/llm gen           # train a 1,024-token BPE, create models/tiny.vzm
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
llm gen   [--out models/tiny.vzm] [--layers 3] [--heads 4] [--kv-heads 2]
          [--embd 96] [--block 64] [--data data/input.txt] [--vocab 1024]
          [--seed 1337]

llm train [--data data/input.txt] [--out models/tiny.vzm] [--model in.vzm]
          [--vocab 1024] [--heads 4] [--kv-heads 2] [--steps 12000]
          [--batch 8] [--lr 0.003] [--wd 0.01]
          [--eval-every 500] [--sample-every 1000] [--seed 1337]

llm infer [--model models/tiny.vzm] [--prompt "ROMEO:"] [--tokens 200]
          [--temp 0.8] [--top-k 40] [--seed 0]

llm serve [--model models/tiny.vzm] [--port 8080]
llm info  [--model models/tiny.vzm]
```

`gen` trains the BPE vocab, then writes a fresh, randomly-initialised tensor
model — the same file format `train` and `infer` use, so a checkpoint is just
"tokenizer + tensors someone trained". `info` prints the config and registry:

```
model: models/tiny.vzm
  params   403104
  layers   3
  heads    4 (kv 2, head dim 24)
  embd     96
  block    64
  vocab    1024
  tokenizer bpe
  tensors  29
  tensors:
    wte                [1024, 96]
    h0.ln1.g           [96, 1]
    h0.attn.wq         [96, 96]
    h0.attn.wk         [96, 48]
    ...
```

## The model

A modern pre-norm decoder with a tied output head:

```
tokens ──► wte[idx]
             │
             ▼  ×n_layer
    ┌──────────────────────────────────────┐
    │ x = x + GQA(RoPE(RMSNorm(x)))        │   grouped-query attention
    │ x = x + SwiGLU(RMSNorm(x))           │   gate · SiLU(up) · down
    └──────────────────────────────────────┘
             │
             ▼
        RMSNorm ──► logits = x @ wteᵀ ──► softmax ──► next token
```

RMSNorm has a weight and no bias; positions come from RoPE (no `wpe`);
attention uses 4 query heads over 2 KV heads (GQA); the FFN is SwiGLU with a
`8/3 · d` hidden size.

Default `tiny` config: 3 layers, 4 heads / 2 KV heads, 96 dims, 64-token
context, a corpus-trained 1,024-token BPE → **403,104 parameters** (1.6 MB).
Bigger shapes are one flag away:

```bash
./zig-out/bin/llm gen  --layers 6 --heads 6 --kv-heads 2 --embd 192 --block 128 --out models/medium.vzm
./zig-out/bin/llm train --model models/medium.vzm --steps 20000 --batch 16
```

### Tokenizer

`src/bpe/` is a Zig 0.16 port of
[alvarobartt/bpe.zig](https://github.com/alvarobartt/bpe.zig)
(MIT OR Apache-2.0), with the POSIX-regex pre-tokenizer replaced by a pure-Zig
implementation. `src/bpe/train.zig` adds a corpus-trained BPE: `llm gen` starts
from the 256 byte tokens, greedily merges the most frequent pairs until
`--vocab` (default 1,024), and stores the vocab + merge pairs in the `.vzm` v3
blob. No downloaded tokenizer, no GPT-2 vocabulary.

The checkpoint format is v3 (modern block + tokenizer kind); older v1/v2
checkpoints are not loadable and must be retrained.

### Autograd, not a hand-written tape

`Session` states the forward pass with Fucina's tagged tensors and lets
`loss.backward()` walk the recorded graph; AdamW updates the parameter leaves.
Every parameter is a Fucina `variable` that borrows its slice of the `.vzm`
blob **in place**, so the optimizer writes straight into the checkpoint blob
and `model.save()` needs no copy-back.

The wiring is guarded by central finite differences on seven gradient buckets
(embeddings, GQA attention + RoPE, SwiGLU, RMSNorm), and cached decoding is
checked token-for-token against a full forward pass:

```
test "gradients match finite differences" ... OK
test "cached decoding matches a full forward pass (greedy)" ... OK
```

Training loop: AdamW (β₁=0.9, β₂=0.99), decoupled weight decay 0.01, global
gradient-norm clipping at 1.0, 5% linear warmup then cosine decay to 10% of peak
LR, and a 95/5 train/validation split of the corpus.

## Checkpoint format (`.vzm`)

One flat `f32` blob plus a named registry. Fucina variables borrow their slices
in place, so the optimizer updates the checkpoint blob directly; gradients and
Adam moments live in Fucina's per-parameter state.

| field | type | notes |
| --- | --- | --- |
| magic | 4 bytes | `"VZLM"` |
| version | u32 | `3` (modern dense block) |
| tokenizer_kind | u32 | `1` = char, `2` = corpus-trained BPE |
| n_layer, n_head, n_kv_head, n_embd, block_size, vocab_size | 6 × u32 | model shape |
| vocab_len, vocab | u32 + bytes | tokenizer payload (char table or BPE blob) |
| n_tensors | u32 | registry length |
| per tensor: name_len, name, rows, cols, dtype | u32, bytes, u32, u32, u32 | dtype `0` = f32 |
| blob | `total_floats × f32` | row-major, little-endian |

Everything is written by `model.save()` and validated on load (bad magic,
truncated file, unknown dtype and shape mismatches are all rejected — see the
`deserialize rejects a corrupted magic` test).

## HTTP API (OpenAI compatible)

`llm serve` starts a hand-rolled HTTP/1.1 server on `std.Io.net` — no framework,
no middleware. The routes speak the OpenAI schema, so any OpenAI client works by
pointing its base URL here (the API key is accepted and ignored):

```bash
export OPENAI_BASE_URL=http://localhost:8080/v1
export OPENAI_API_KEY=local
```

| route | description |
| --- | --- |
| `GET /` | the chat UI (embedded in the binary) |
| `GET /health` | `{"ok":true}` |
| `GET /v1/models` | model list |
| `GET /v1/models/vibezig-llm` | model details + `meta` (params, layers, ...) |
| `POST /v1/chat/completions` | chat completion; `"stream": true` for SSE |
| `POST /v1/completions` | legacy text completion |

```bash
curl -s localhost:8080/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"ROMEO:"}],"max_tokens":64}'
```

Bun one-liner against the same endpoint (top-level `await` works):

```bash
bun -e '
const j = await fetch("http://localhost:8080/v1/chat/completions", {
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify({
    model: "vibezig-llm",
    messages: [{ role: "user", content: "ROMEO:" }],
    max_tokens: 64,
  }),
}).then((r) => r.json());
console.log(j.choices[0].message.content);
'
```

Or with the official package (`bun add openai`):

```bash
bun -e 'import OpenAI from "openai";
const c = new OpenAI();  // reads OPENAI_BASE_URL / OPENAI_API_KEY
const r = await c.chat.completions.create({ model: "vibezig-llm",
  messages: [{ role: "user", content: "ROMEO:" }], max_tokens: 64 });
console.log(r.choices[0].message.content);'
```

## Performance

Measured on an Intel i5-13400 (16 threads), `--release=fast`, Fucina CPU
kernels:

| workload | throughput |
| --- | --- |
| BPE training on 1 MB (vocab 1,024) | **~0.5 s** |
| encode / decode (1.1 MB corpus) | ~19 MB/s / ~46 MB/s |
| training (403k params, batch 8 × seq 64) | ~19,000–27,000 tok/s |
| inference, KV cache, 200-token context | ~6,900 tok/s |

A 500-step run takes the modern model from loss 6.94 (random) to 4.02 — ahead of
the old GPT-2 block's 4.30 at the same step count — and produces readable
fragments. Throughput depends on the host CPU and the BLAS provider Fucina
auto-detects (`-Dblas=openblas` / `mkl` / `blis` are used when installed).

## Limits (on purpose)

This is a readable reference implementation, not a production inference stack:

* **v3 checkpoints only** — the modern block changes the tensor registry; old
  v1/v2 GPT-2 checkpoints must be retrained.
* **encoder is O(L²) per piece** — it finds the next lowest-rank merge by
  rescanning the piece. Fine for words and prompts, but long whitespace or
  punctuation runs are quadratic; a rank-ordered linked list is the next win.
* **f32 only** — no quantisation, no bf16/f16.
* **one request at a time** — the server serialises generations behind the model.
* **CPU only** — Fucina's Metal/CUDA GEMM offload is not enabled here.
* **no MoE yet** — the dense block is the baseline; routing is a later step.
* **pinned toolchain** — Fucina 0.5.1 requires Zig 0.16.0; a Zig upgrade waits
  for upstream.

Each of those is a contained piece of work; see `docs/ROADMAP.md`.

## Tests

```bash
zig build test
```

27 tests: full-model gradients vs finite differences, backward determinism,
"one AdamW step reduces loss", KV-cached decoding vs a full forward pass,
sampling length/range, BPE training + blob round-trip, optimized-vs-reference
trainer parity, `.vzm` v3 round-trips and corruption rejection, plus the CLI
and server helpers.

## License

MIT. See `LICENSE`.

## Third-party

* [Fucina](https://github.com/matteo-grella/fucina) 0.5.1 is fetched by the Zig
  build and statically linked. MIT, copyright (c) 2026 Matteo Grella and the
  Fucina contributors.
* [`src/bpe/`](src/bpe/) is a 0.16 port of
  [alvarobartt/bpe.zig](https://github.com/alvarobartt/bpe.zig), MIT OR
  Apache-2.0 (license files live beside the sources). `train.zig` is the
  corpus-trained extension written for this project.
