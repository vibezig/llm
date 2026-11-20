# Architecture notes

Why the code looks the way it does, and the maths behind the backward pass.

## 1. One flat parameter blob

`Model` owns `blob: []f32` plus a registry of `(name, offset, rows, cols)`.
Offsets are assigned in a canonical order (`buildRegistry`), so:

* gradients (`Session.grads`) and Adam moments (`OptState.m/v`) are allocated
  with the same length and use the **same indices** — no per-tensor plumbing;
* saving a checkpoint is a header plus one memcpy of the blob;
* the transformer hot path never does name lookups: `model.layout` resolves every
  offset once at load time.

Adding a tensor means adding it to `buildRegistry` and to `LayerOffsets`; the
registry test asserts offsets stay contiguous.

## 2. Kernels

Everything is row-major f32. The two GEMM shapes that matter:

* `matmul(m,n,k,a,b,c)`: `c = a @ b` where `b` is input-major `[k,n]`. Used for
  qkv projections, the MLP and the output projection.
* `matmulNt(m,n,k,a,b,c)`: `c = a @ bᵀ` with `b` output-major `[n,k]`. Used for
  the logits (`x @ wteᵀ`), for `dX = dY @ Wᵀ`, and for `dQ`/`dK`.
* `matmulTnAcc(m,n,k,a,b,c)`: `c += aᵀ @ b` with `a` token-major `[k,m]`. Used for
  every weight gradient (`dW = activationsᵀ @ dY`).

`matmul` and `matmulTnAcc` are 4×8 register-blocked: four independent rows share
one `@Vector(8, f32)` load of `B`, so arithmetic intensity per `B` byte goes up
4× and the FMA loop stays in registers. The tail rows/columns fall back to a
plain axpy loop, which keeps the code correct for any shape (tests compare
against the naive reference for `m,n,k` that are not multiples of 4/8).

## 3. Forward pass

Rows are `(batch × time)` flattened: row `r = b*T + t`, so all per-token work is a
2-D matmul. Attention needs to see individual heads, and each head occupies a
contiguous `head_dim` slice inside the row, so the per-head loops read
`q[r*C + h*hd ..][0..hd]` without a transpose.

Per layer:

```
x  := LN1(x)            (save x, mean, rstd)
q  := x @ Wq ; k := x @ Wk ; v := x @ Wv            (save q,k,v)
for each (batch, head):
    S[t,s] = (q_t · k_s) / sqrt(hd)   for s <= t      (causal)
    P      = softmax(S)                                (save P)
    O_t    = Σ_s P[t,s] · v_s                          (save O)
x  := x + O @ Wo
y  := LN2(x)            (save x, mean, rstd)
h  := GELU(y @ Wfc)     (save pre-GELU h and post-GELU h)
x  := x + h @ Wproj
```

The final `LN_f` and `logits = x @ wteᵀ` close the pass. The embedding matrix is
tied: it is both the lookup table and the output projection.

## 4. Backward pass

Given `dl = ∂L/∂logits` from softmax cross-entropy (already scaled by `1/rows`):

```
dlnf   = dl @ wte
dwte  += dlᵀ @ lnf                       (output side of the tie)
dx     = LayerNormBwd(dlnf, …)            # gradient into the last block
```

then for each layer, from the top:

```
# MLP branch
dh2    = dx @ Wprojᵀ
dWproj+= hᵀ @ dx
dh1    = GELU'(h_pre) ⊙ dh2
dWfc  += yᵀ @ dh1
dln2   = dh1 @ Wfcᵀ
dx     = dx + LayerNormBwd(dln2, …)        # residual: gradients add

# attention branch
dO     = dx @ Woᵀ
dWo   += Oᵀ @ dx
for each (batch, head):
    for t:  dV_s += P[t,s] · dO_t        for s <= t
            dP_s  = dO_t · v_s
    row_sum = Σ_s P[t,s] · dP_s
    dS_s    = (dP_s − row_sum) · P[t,s]   # softmax Jacobian, diagonal form
    dQ_t   += dS_s · k_s · scale
    dK_s   += dS_s · q_t · scale
dWq += xᵀ @ dQ ; dWk += xᵀ @ dK ; dWv += xᵀ @ dV
dln1   = dQ @ Wqᵀ + dK @ Wkᵀ + dV @ Wvᵀ
dx     = dx + LayerNormBwd(dln1, …)
```

Finally `dx` is scattered back into `dwte` (row per token id) and `dwpe` (row per
position). The tied matrix therefore accumulates two contributions, which is why
`dwte` is written with `+=` in both places.

Layer norm backward uses the standard closed form:

```
xhat   = (x − μ) · rstd
dg    += dy ⊙ xhat
db    += dy
dx    += rstd · (g⊙dy − mean(g⊙dy) − xhat · mean(g⊙dy ⊙ xhat))
```

Two invariants keep this honest:

* **determinism** — every scratch buffer is zeroed before an accumulating kernel;
  a test runs five identical backward passes and demands bit-identical gradients
  (this caught a real bug: `dh2` used to accumulate across steps and the gradient
  norm grew ~10⁸ within a dozen steps);
* **numerical agreement** — a second test perturbs individual parameters of
  embedding / attention / MLP / layernorm tensors and compares the analytic
  gradient with central finite differences.

## 5. Sampling

`generate` runs the model on the current context, applies temperature, keeps the
top-k logits (sorted by score; k ≤ vocab so an O(V log V) sort is nothing), and
samples from the renormalised distribution. The context slides once it hits
`block_size`, dropping the oldest token — no KV cache yet, so each step is a full
forward pass (see `docs/ROADMAP.md`).

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
