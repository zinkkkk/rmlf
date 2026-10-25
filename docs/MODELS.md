# rmlf model types — what they are, why they exist, how they convert

This crate produces a few different "model" and "decomposition" objects. This doc
explains each one, why it exists, and how to move between them.

- **Current state** (the tangle this doc is fixing): `models_current.png`
- **Target state** (where the API is going): `models_target.png`

## Provenance — papers vs. our extras

Some objects are the **papers' mathematics**; others are **utilities we added**.
Keeping this straight is the key to understanding why they differ. (Paper 1 =
*…curse of dimensionality*, arXiv:2405.00495; Paper 2 = *Variable decoupling…*,
arXiv:2605.07246.)

| Object | From the papers? |
|--------|------------------|
| recursive null space (`recurse`) | **Paper 1, Thm 5.3** — the n-D null space as a cascade of 1-D solves |
| `DecoupledModel` (`C_num`/`C_den`) | **Paper 1 Thm 5.4 + Paper 2 Thm 2.1/Cor 2.1** — `Bary` weights as a Hadamard product of per-variable factors, by inspection |
| `DecoupledFn` / `decoupled_functions` (the Φ_l) | **Both papers** (P1 §6 Eq. 23; P2 §4 Eq. 17) — the **decoupled single-variable functions** of the rational-KST result; the headline output |
| `Barycentric` (dense model) | the assembled interpolant (Eq. 6 / Eq. 10) |
| `fibre` / `BarycentricRational1D` (model **slice**) | **our extra** — a slicing utility; the papers never restrict the model to a line |
| `CompressedBarycentric` (TT) | **our extra** — Paper 1 stores the small 1-D Loewners (Thm 5.6); TT cores are our implementation of the storage/eval saving |
| `ChannelDecomposition` (TT channels) | **our extra** — a TT-rank coupling inspector |

So: `DecoupledModel` and `DecoupledFn` are the **paper's decomposition**; `fibre`,
TT compression, and channels are **our conveniences** layered on top. The cleanup
preserves every one — it only adds traits/conversions and clarifies naming.

---

## The 3 core models — same function, three representations

All three represent the **same** `n`-variable rational `g(x) = N(x)/D(x)` in
barycentric form. They differ only in *how the coefficients are stored*, which is
what makes one or another convenient.

| Type | Stores | Built by | Use it when |
|------|--------|----------|-------------|
| **`Barycentric`** | the dense coefficient tensor `c` (+ nodes `λ`, sampled values `w`) | `alg1` / `alg2` / `alg3` | the default, general-purpose model; the **hub** all conversions go through |
| **`DecoupledModel`** | two separate Kronecker tensors `C_num`, `C_den` | `decouple_opt` / `decouple_adaptive` | you want the **KST structure** — the per-variable decomposition exposed (numerator and denominator handled separately) |
| **`CompressedBarycentric`** | the coefficient tensors in **tensor-train (TT)** form | `CompressedBarycentric::from_model(&Barycentric, tol)` | high `n`: TT makes **storage and evaluation linear in `n`** instead of `∏ kₗ` |

### Converting between them — every pair, one idiomatic method each

Every core model implements three discoverable conversion methods (on the
[`Model`] trait), so you never have to remember a free function or annotate a type:

```rust
let comp = bary.to_compressed();    // Barycentric  → CompressedBarycentric
let dec  = bary.to_decoupled();     // Barycentric  → DecoupledModel
let bary = comp.to_barycentric();   // CompressedBarycentric → Barycentric
// …and the same three methods on every model.
```

All six directed edges exist and are **exact** (the dense `Barycentric` is the hub;
indirect edges route through it). `From`/`Into` impls back every method, so
`let c: CompressedBarycentric = bary.into();` works too.

| from ↓ / to → | `Barycentric` | `DecoupledModel` | `CompressedBarycentric` |
|---|---|---|---|
| **`Barycentric`** | — | `to_decoupled()` | `to_compressed()` |
| **`DecoupledModel`** | `to_barycentric()` | — | `to_compressed()` |
| **`CompressedBarycentric`** | `to_barycentric()` | `to_decoupled()` | — |

The only subtlety: a `Barycentric` folds the numerator into `w = C_num/C_den`, so it
carries no separate numerator *at a pole* (`C_den = 0`). Thus the round-trip
`DecoupledModel → Barycentric → DecoupledModel` is lossy *only at pole nodes*; every
non-pole model round-trips bit-exactly, and all conversions agree on `eval`.

### What does "compressing" actually do?

`CompressedBarycentric` factorizes the dense coefficient tensor (size `∏ kₗ`,
exponential in `n`) into a **tensor train**: a chain of small cores
`r_{l} × k_l × r_{l+1}`. For a structured model the bond ranks `r_l` stay small,
so storage drops from `∏ kₗ` to `Σ r_l² k_l` — **exponential → linear in `n`** —
and evaluation likewise (`O(r²·Σ kₗ)`). On a separable `n=7` model this was ~52×
smaller and ~11× faster to evaluate, **exact**. An incompressible model just falls
back to full rank (never worse than dense).

### Is compression lossy? Do I have to pick a tolerance?

**No, it does not have to be lossy.** The `tol` argument is a *truncation* knob:

- `tol = 0` → **exact / lossless** (keeps full rank; the TT is a bit-exact
  refactorization of the dense tensor).
- `tol > 0` → drop bonds whose singular values fall below `tol·‖·‖_F`, trading a
  bounded error for smaller rank.

If you want guaranteed-exact arithmetic with **no SVD and no tolerance at all**,
use the **`exact` module** (`exact::tt`): over ℚ / 𝔽_p the TT ranks are computed by
exact rank factorization (RREF), a theorem of the data rather than a threshold.

---

## The 1-D layer — TWO genuinely different "single-variable" views

This is the part that looks redundant but **is not** (verified numerically — the
two views differ by hundreds in magnitude on the same model; they compute
different things). There are two distinct notions of "the model as 1-D functions":

### View 1 — **model slices** (`fibre`): evaluate the model on a line

| Type | What it is |
|------|-----------|
| **`BarycentricRational1D`** | a classic 1-D barycentric `(nodes, values, weights)` triple — the **universal 1-D interchange format** (what AAA / classic 1-D routines consume) |
| **`LabelledFibre`** | a `BarycentricRational1D` + which axis / pin it came from |

`fibre(axis, pin)` pins every coordinate except `axis` and collapses the n-D model
**exactly** to a 1-D barycentric rational in that variable —
`fibre(l, pin).eval(t) == model.eval(pin with t in slot l)` (verified to ~2e-16).
**This is a slice of the model**: it returns model values. Use it to view/extract
the model along any line, or hand a 1-D barycentric to other code.

`fibre_set()` / `fibres_by_axis()` return the canonical set of these slices over
the grid lines (`1 + k₁ + k₁k₂ + …` of them).

### View 2 — **KST decomposition factors** (`decoupled_functions`): the Φ_l building blocks

| Type | What it is |
|------|-----------|
| **`DecoupledFn`** | one decoupled **factor** `Φ_l(x_l)` of the KST representation — `{axis, lead, λ, values}`, evaluated by Lagrange through node values |

`DecoupledModel::decoupled_functions()` returns `Vec<Vec<DecoupledFn>>` — the
explicit single-variable functions of the paper's rational-KST formula (Eq. 24).
**These are NOT model values**: they are the per-axis factors whose **Kronecker
products / superposition reconstruct** the model (`g = Σ ∏_l Φ_l`-style). A single
`DecoupledFn.eval(t)` is a component, not `g` on a line.

### Why both exist (the key distinction)

- `fibre` answers *"what does the model look like along this axis?"* → **a slice**.
- `decoupled_functions` answers *"what are the single-variable functions the KST
  theorem says the model factors into?"* → **the factors**.

They share the word "single-variable" and similar shapes, which is why they look
duplicate — but one is the model restricted to a line, the other is a multiplicative
building block. Keep both; the cleanup is to **name/document them so the difference
is obvious**, not to merge them.

---

## The inspection view — `ChannelDecomposition`

`CompressedBarycentric::channel_decomposition()` re-expresses the TT cores as
per-axis **matrices of 1-D "channel" functions**. The model is then a matrix
product `g = (∏_l M^num_l(x_l)) / (∏_l M^den_l(x_l))`, and each matrix entry is a
genuine 1-D function of one variable. `reconstruct(x)` multiplies them back to
`g(x)` (it does **not** "lead nowhere" — it returns the value).

**This is an analysis tool, not a pipeline step.** Its purpose is to *inspect the
multivariate coupling*: which channels interact = the TT rank structure; a
channel's pole pattern is its own 1-D function. Rank-1 (separable) models collapse
to one channel per axis (`g = ∏num_l/∏den_l`). Reach for it when you want to
*understand* a model's structure, not to convert it to something else.

---

## The problems this doc is fixing (see `models_current.png`)

1. **`DecoupledFn` LOOKS like a duplicate of the fibre types but is NOT** (verified:
   different values by orders of magnitude). It is the KST *decomposition factor*
   `Φ_l`; the fibre types are model *slices*. → **do not merge; rename/document to
   make the slice-vs-factor distinction obvious** (e.g. keep `DecoupledFn` but its
   docs must say "KST factor, not a model slice; combines via Kronecker product").
2. **`fibre`/`fibre_set` are copy-pasted** on `Barycentric`, `DecoupledModel`, and
   `CompressedBarycentric` (three separate impls). → **one `Fibres` trait.**
3. **Conversions were ad-hoc inherent methods with holes** (`to_barycentric`,
   `from_model`, but no `Barycentric → DecoupledModel`, no `* → DecoupledModel`) and
   weren't `From`/`Into`, so they were undiscoverable. → **DONE: all six edges as
   `From`/`Into` *and* idiomatic `to_barycentric()` / `to_compressed()` /
   `to_decoupled()` methods on the `Model` trait; complete graph through the
   `Barycentric` hub.**
4. **Compression's `tol` looks mandatory/lossy.** → **document `tol=0` = exact, add
   a clear lossless constructor, point at the `exact` module for true exactness.**

## Target (see `models_target.png`)

- A **`Model`** trait (`eval`, `ndim`) implemented by all three core models.
- A **`Fibres`** trait (`fibre`, `fibre_set`, `fibres_by_axis`) — one definition,
  uniform across models, no duplication.
- **`From`/`Into` + idiomatic `to_*()` methods** between all three core models
  (every one of the six edges, exact where the math allows; `Barycentric` is the
  hub). See "Converting between them" above.
- **One** 1-D type (`BarycentricRational1D`) + one labelled wrapper
  (`LabelledFibre`); `DecoupledFn` removed.
- `ChannelDecomposition` kept, clearly labeled **analysis/inspection only**.
