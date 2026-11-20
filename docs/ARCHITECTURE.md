# Architecture notes

Why the code looks the way it does.

## 1. One flat parameter blob, borrowed by Fucina

`Model` owns `blob: []f32` plus a registry of `(name, offset, rows, cols)`.
Offsets are assigned in a canonical order (`buildRegistry`), so:

* the `.vzm` checkpoint is still a header plus one memcpy of the blob;
* the transformer hot path never does name lookups: `model.layout` resolves every
  offset once at load time;
* `Session` wraps each slice as a Fucina **variable borrowing that slice in
  place** (`ctx.fromBorrowedSlice`), so the optimizer writes straight into the
  checkpoint blob and `model.save()` needs no copy-back.

Gradients and Adam moments are Fucina-owned per-parameter state, so they no
longer share the blob's layout. Adding a tensor means adding it to
`buildRegistry`, to `LayerOffsets`, and to `Session.registerParams`; the
registry test asserts offsets stay contiguous.

The checkpoint also carries the tokenizer bytes (`vocab_len`/`vocab`): a raw
byte->id table for v1 char models, or the compact BPE vocab+merge blob for v2.
`Model` parses it into a `Tokenizer` at load time, so a checkpoint is
self-contained. Both versions remain readable.

`Session` is heap-allocated because Fucina pins an `ExecContext` after `init`
(the embedded allocator points at itself) and the optimizer retains pointers to
the parameter fields. Moving a `Session` value would invalidate both.

## 2. Kernels and autograd (Fucina)

All math is [Fucina](https://github.com/matteo-grella/fucina) 0.5.1: tagged f32
tensors, CPU SIMD kernels, eager autograd, AdamW. There are no local tensor
kernels in this repository.

The model uses:

* `.plain` matmul for `[rows, in] @ [in, out]` weights (qkv, MLP, output
  projection), matching the `.vzm` row-major `[in, out]` layout;
* `.trans_b` for the tied head: `[rows, embed] @ [vocab, embed]ᵀ`;
* fused `layerNorm`, `gelu` and `crossEntropy` ops, each with its own VJP.

Fucina selects a CBLAS provider at build time (`-Dblas=openblas|mkl|blis|nvpl`)
and otherwise uses its portable Zig vector kernels. The dependency and its
version are pinned in `build.zig.zon`.

## 3. Forward pass

Tokens come from the model's tokenizer: a byte-level BPE trained on the corpus
(`src/bpe/`, using bpe.zig's encoder and a pure-Zig trainer), or the legacy char
path for v1 checkpoints. The trainer is incremental — symbols are a doubly
linked list, each adjacent pair has an intrusive position list, and a max-heap
selects the next merge — so a merge touches only that pair's occurrences
(1 MB / vocab 1,024 trains in ~0.5 s). Tokens and positions are gathered into a
`[batch*seq, model]` stream (row `r = b*T + t`). Per layer:

```
x  := LN1(x)
q  := x @ Wq ; k := x @ Wk ; v := x @ Wv
O  := causalAttention(q, k, v)        # per batch row, all heads at once
x  := x + O @ Wo
y  := LN2(x)
h  := GELU(y @ Wfc)
x  := x + h @ Wproj
```

Attention is the only place the batch axis is explicit: q/k/v are projected for
all rows in one matmul, then each batch row is narrowed, split to
`[seq, head, d]`, passed through Fucina's causal `groupedAttention`, and
concatenated back along `.seq`. That keeps one weight read per projection while
respecting per-sequence causality.

The final `LN_f` and `logits = x @ wteᵀ` close the pass. The embedding matrix is
tied: it is both the lookup table and the output projection, and Fucina
accumulates both gradient contributions into the same variable.

## 4. Backward pass

`loss.backward(ctx)` walks Fucina's recorded graph; every parameter leaf
accumulates its gradient, `clipGradNorm` applies global clipping, `AdamW.step`
updates the borrowed blob, and `zeroGrad` clears the leaves for the next step.

This repository no longer implements VJPs. It guards the model **wiring** with:

* **numerical agreement** — perturb individual parameters of embedding /
  attention / MLP / layernorm tensors and compare the analytic gradient with
  central finite differences;
* **determinism** — repeated forward/backward passes must reproduce the loss and
  grad norm exactly;
* **progress** — 20 AdamW steps must reduce the loss;
* **decode parity** — KV-cached generation must match a full forward pass token
  for token.

## 5. Sampling

`generate` prefills the prompt one token at a time into a per-layer host K/V
cache, then each new token costs a single-row forward over borrowed zero-copy
cache views: O(T) attention per token instead of O(T²). When the context fills,
the newest half is kept and replayed. Temperature scaling and top-k sampling are
unchanged. Because byte-level BPE tokens can split a UTF-8 sequence, the CLI and
server decode each token to bytes and the server feeds those bytes through the
existing streaming UTF-8 repair before emitting SSE.

## 6. HTTP layer

`std.http.Server` over a `std.Io.net` stream, one connection at a time, with
`connection: close` on every response. Requests are parsed by hand: the target is
split on `?`, query values are percent-decoded into a fixed buffer inside
`GenParams` (so `prompt` never dangles), and JSON responses are built with a
byte-level escaper instead of a serialiser.

The streaming route uses `respondStreaming` (chunked transfer encoding) and emits
one SSE event per generated byte. Bytes rather than strings, because a byte-level
tokenizer can split a UTF-8 sequence across tokens and the browser decodes with a
streaming `TextDecoder`.
