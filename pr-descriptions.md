# `fix-svdsolve-gkl-breakdown`: Stop GKL at `α = 0` instead of normalising noise

For the issue "`svdsolve`: false singular triplets, too many values, or a hang when the start vector overlaps the null space of `A'`".

GKL ends on `α = 0` whenever the start vector overlaps the null space of `A'`, and `α` was never tested. It now stops there, as for `β = 0`, and returns only valid triplets.

**What changed**

- `gklrecurrence`: `α` at rounding level gives `v = 0`, `r = 0`, `β = 0`, so the existing stopping logic applies.
- `svdsolve`: drops the zero singular value of that factorization.
- `bidiagsvd!`: throws on non-finite input, since `LAPACK.bdsqr!` can hang on `NaN`.
- `ClassicalGramSchmidt2`: adds the `V` pass that `ModifiedGramSchmidt2` already has.
- Single-pass recurrences: one extra pass over `V` when `α < sqrt(eps) * norm(B)`.

**Behaviour changes you may object to**

- `σ = 0` is not returned here, so `:SR` can miss it. Its left vector is valid, but its right vector isn't in the Krylov subspace. Before, it came with a wrong right vector.
- The result can be empty (zero matrix, or `x` in the null space of `A'`), with the invariant-subspace warning.
- A value unreachable from `x` stays missing: the 3×3 example gives `[6]` although 12 exists, and random starts give `[12, 6]`.
- `GKLFactorization` can now end with a zero last vector in `V`.

**Please look closely at**

- `αtol = 10 * eps * max(αs, βs)` in `expand!`. The factor 10 is a judgement call.
- The `sqrt(eps)` gate for the single-pass pass.
- The `K′` handling in `svdsolve`.

Reorthogonalizing runs only gain scalar checks, except `ClassicalGramSchmidt2`, which gains its `V` pass. AD tests and the rest of the suite were not run.

<details><summary>Details</summary>

### Why this is correct at `α = 0`

At `α_k = 0`, `A * V[1:k-1] = U * B[:, 1:k-1]` and `A' * U = V[1:k-1] * B[:, 1:k-1]'` hold, with `B[:, 1:k-1]` of size `k×(k-1)`. So `U` and `V[1:k-1]` are closed under `A'` and `A`, and those `k - 1` singular values are exact.

Keeping `α_k = 0` makes the last column of the square `B` zero. `bdsqr!` then gives a singular value 0, and the other Ritz vectors have a zero coefficient on the zero `V[K]`. Using `K′ = K - 1` drops that value with no other change to the return path. `normres` is then `0` for all returned values, since `r = 0`.

### Threshold

`α / (eps * max(αs, βs))` at the breakdown, measured on `master` for random matrices:

| Orthogonalizer | Ratio |
|---|---|
| `ModifiedGramSchmidt2` | below `1e-15` in Float64, about `1e-7` in Float32 |
| `ModifiedGramSchmidtIR` | the same, except `2.0` when scaled by `1e-6` and `3e-12` for a graded matrix |
| `ClassicalGramSchmidtIR` | underflows towards 0 |
| All, rank-5 40×40 | about 170: real tiny singular values, and continuing is valid |
| `ClassicalGramSchmidt`, `ModifiedGramSchmidt`, 8×3 | 2 to 45 |

- **False negatives** with `*2`/`*IR` were harmless in these runs: `v` stays a valid orthonormal direction.
- **False positives** can only lose values of the threshold's order.
- **Single-pass:** only `V[end]` is removed, so `α` at a breakdown is the size of the orthogonality error, not of rounding. A fixed factor can't separate this from real small values. So when `α < sqrt(eps) * norm(B)`, the recurrence orthogonalizes against all of `V` and tests what's left. A larger `α` costs nothing extra.

Number of values returned with plain `ModifiedGramSchmidt`, `krylovdim = 30`, `maxiter = 2`, random matrices. The first two rows are 8×3 for seeds 1 to 12:

```
10 eps test only:    3 3 3 10 3 3 19 3 19 5 10 3 
with the extra pass: 3 3 3 3 3 3 3 3 3 3 3 3 
before: 60x20 Random.seed! 1:6 MGS 3 2 2 1 1 1 
before: 40x12 Random.seed! 1:6 MGS 2 2 2 4 2 2 
before: 16x6 Random.seed! 1:6 MGS 5 5 1 5 6 3 
after:  60x20 Random.seed! 1:6 MGS 3 2 2 20 1 1 
after:  40x12 Random.seed! 1:6 MGS 12 12 12 4 12 12 
after:  16x6 Random.seed! 1:6 MGS 6 6 6 6 6 6 
```

Counts below the matrix size that remain, such as 40×12 seed 4, stop early at unconverged duplicates. That is the separate ghost-value problem, not handled here.

### `ClassicalGramSchmidt2`

Its old comment said the `V` pass was unnecessary. Without it, `V` lost orthogonality and `α` didn't vanish: a random 60×20 gave 21 values with true residual 9.2. With it, 20 values and residual `6.2e-14`.

### Results at 321bc0a

Same matrices and settings as the issue:

```
tol=1.0e-6                        S=[6.0] conv=1 maxres=4.4e-16
tol=1.0e-10                       S=[6.0] conv=1 maxres=4.4e-16
tol=1.0e-14                       S=[6.0] conv=1 maxres=4.4e-16
krylovdim=2                       S=[6.0] conv=1 maxres=4.4e-16
krylovdim=3                       S=[6.0] conv=1 maxres=4.4e-16
krylovdim=4                       S=[6.0] conv=1 maxres=4.4e-16
krylovdim=8                       S=[6.0] conv=1 maxres=4.4e-16
krylovdim=30                      S=[6.0] conv=1 maxres=4.4e-16
eager=true                        S=[6.0] conv=1 maxres=4.4e-16
ModifiedGramSchmidt2              S=[6.0] conv=1 maxres=4.4e-16
ClassicalGramSchmidt2             S=[6.0] conv=1 maxres=4.4e-16
ModifiedGramSchmidt               S=[6.0] conv=1 maxres=4.4e-16
ClassicalGramSchmidt              S=[6.0] conv=1 maxres=4.4e-16
3x3 seed=1 krylovdim=8            S=[12.0, 6.0] conv=2 maxres=3.8e-15
3x3 seed=2 krylovdim=8            S=[12.0, 6.0] conv=2 maxres=2.0e-15
3x3 seed=3 krylovdim=8            S=[12.0, 6.0] conv=2 maxres=1.8e-15
3x3 seed=4 krylovdim=8            S=[12.0, 6.0] conv=2 maxres=6.9e-15
3x3 seed=1 default                S=[12.0, 6.0] conv=2 maxres=3.8e-15
8x3 seed=1 krylovdim=5            S=[4.112, 3.652, 2.031] conv=3 maxres=1.2e-15
8x3 seed=2 krylovdim=5            S=[3.756, 2.573, 2.329] conv=3 maxres=1.7e-15
rank-2 8x8 seed=1 krylovdim=5     S=[7.436, 4.616] conv=2 maxres=2.2e-15
full-rank 4x4 seed=1              S=[3.462, 2.209, 1.798, 0.3855] conv=4 maxres=1.4e-15
wide [1 1 1; 1 1 1]               S=[2.449] conv=1 maxres=3.1e-16
complex 3x3                       S=[2.0] conv=1 maxres=2.5e-16
A + I (control)                   S=[7.0, 1.0] conv=2 maxres=2.0e-15
ModifiedGramSchmidtIR             S=[6.0] conv=1 maxres=4.4e-16
ClassicalGramSchmidtIR            S=[6.0] conv=1 maxres=4.4e-16
[1 1; 1 1], x=[1,-1]              S=Float64[] conv=0 maxres=0.0
```

The issue's example:

```
S = [5.999999999999999]
info.normres = [0.0]
info.converged = 1
svdvals(A) = [12.0, 6.0, 0.0]
[norm(A * V[i] - S[i] * U[i]) for i = eachindex(S)] = [4.440892098500626e-16]
S = [2.302775637731995, 1.3027756377319948]
info.normres = [0.0, 0.0]
info.converged = 2
svdvals(A) = [2.3027756377319952, 1.3027756377319946]
[norm(A * V[i] - S[i] * U[i]) for i = eachindex(S)] = [0.0, 7.28021923224409e-16]
S = [4.82842712474619, 0.8284271247461902]
info.normres = [0.0, 0.0]
info.converged = 2
svdvals(A) = [4.828427124746188, 4.0, 0.8284271247461898, 1.0467283057891842e-16]
[norm(A * V[i] - S[i] * U[i]) for i = eachindex(S)] = [3.1401849173675503e-16, 4.1725779438208954e-16]
S = [1.0]
info.normres = [0.0]
info.converged = 1
svdvals(A) = [1.0, 0.0]
[norm(A * V[i] - S[i] * U[i]) for i = eachindex(S)] = [0.0]
```

### Tests

Run on Julia 1.13.2: the GKL parts of `test/factorize.jl` (4058 pass) and `test/svdsolve.jl` (732 pass).

New tests cover the issue's cases with default settings, a wide matrix, a complex matrix, the dimer matrix, a tall matrix with single-pass `orth`, `x` in the null space of `A'`, vector modes, the warning, `:SR`, a `β` control, the `NaN` guard, and a factorization ending on `α = 0`.

On `master`, run per case under a time limit, the issue's cases fail or hang for `ClassicalGramSchmidt2`, `ModifiedGramSchmidt2` and both `IR` variants. The factorization test and the `NaN` guard test hang. The control passes. The single-pass and random-tall cases were added afterwards, and were not run this way.

### Limits and open questions

- Past a breakdown, continuing with a new random direction is valid in theory, and would find unreachable values. That's a separate feature.
- Single-pass `orth` can lose orthogonality too far for the extra pass on larger tall matrices.
- The `β` loop still uses an absolute `eps`, and `β <= tol` stays absolute. Matrices scaled by `1e-6`, `1e6` and `1e10` gave clean results, but those are a few cases.
- `:SR` users who want a null vector can't get it now. Returning a left null vector alone would need an API decision.
- The `svdsolve` rrule when fewer values come back is not tested.

### Merging with `fix-gram-schmidt-ghost-values`

`git merge-tree` reports a conflict only in `test/svdsolve.jl`, but `src/eigsolve/svdsolve.jl` auto-merges to something wrong: both define `K′`, and that branch's `K′ = K` overwrites this one's. This is from reading the code; a merged result was not run. Whichever merges second needs a rebase that combines them.

</details>

---

# `fix-gram-schmidt-ghost-values`: Drop ghost values of single-pass Gram-Schmidt

For the issue "`svdsolve`/`eigsolve` with single-pass Gram-Schmidt return duplicate values".

With `ClassicalGramSchmidt` or `ModifiedGramSchmidt`, `GKL` and `Lanczos` repeat converged values and count the copies. This keeps the chosen recurrence, with no extra operator applications per cycle, and post-processes the Ritz values instead: parallel copies are moved out, and vectors are normalized.

**What changed**

- `_ghosts_last`: finds Ritz values closer than their residuals whose Ritz vectors overlap by more than `1/√2`, keeping the one with the smallest residual.
- `svdsolve`, Lanczos `eigsolve`: move those copies out before counting `converged`, restart from the remaining ones, and normalize returned vectors and residuals.
- A warning when copies are dropped, at the default verbosity.
- The docstring of `ModifiedGramSchmidt`.

Reorthogonalizing `orth` take none of these paths.

**Behaviour changes you may object to**

- `vals` can get shorter, and `converged` smaller, than before.
- A true repeated value can lose a copy: `σ = 5` three times went from `converged = 3` (overlap 0.82) to `converged = 2`.
- Runs can take longer: 3 values with `krylovdim = 20`, `maxiter = 100` now take 424 operations and 25 restarts. `master` stopped after 40 with three copies of 100.

**Please look closely at**

- Whether post-processing is wanted at all, rather than only a warning and docs. Without it, `howmany = 2` returns `[100, 100, ...]` with no sign of trouble, so docs alone leave wrong results. Selective or partial reorthogonalization would prevent ghosts, but would change what a single-pass `orth` costs.
- The overlap threshold `1/√2`, and the closeness test in `_ghosts_last`.
- The restart, which doesn't reorthogonalize `r`.

AD tests and the rest of the suite were not run.

<details><summary>Details</summary>

### Why dropping is correct

In exact arithmetic, a single Krylov vector gives an unreduced `B` or `T`, whose Ritz values are simple, so parallel copies can only come from rounding. Measured for full-rank 8×8 and 400×400 on `master`: copies agree to 10 digits with overlap 1.000, and distinct values have overlap 0.000.

Copies had equal `‖u‖` and `‖v‖` in those runs, from 0.13 to 1.55, so normalizing both preserves the triplet. Some copies are still forming, with a large residual and a value close to the converged one. They sat in the prefix counted by `converged`, so they are dropped too.

### Cost

No operator applications are added per cycle. Ritz vectors are only built for values closer than their residuals, at most `K` of them at each convergence check. So tight clusters cost vector operations, and with `eager = true` that can be every step. This cost was not timed.

Steps spent on ghosts are not recovered: full-rank 8×8 with `krylovdim = 30` still uses 60 operations, against 16 with `ModifiedGramSchmidt2`.

### Results at 226f232

Same matrices and columns as the issue:

```
8x8                    MGS   n=2   conv=2   res=1.2e-13 ov=5.9e-15 norm=2.2e-16 dist=2.2e-15 ops=60   it=1 S=[6.2117, 3.1038]
8x8 hm=3               MGS   n=8   conv=8   res=9.3e-14 ov=7.3e-15 norm=2.2e-16 dist=2.2e-15 ops=104  it=2 S=[6.2117, 3.1038, 2.8674, 2.2221, 1.2259, 0.7755, 0.1814, 0.1298]
8x8 :SR                MGS   n=5   conv=5   res=2.2e-14 ov=3.0e-14 norm=1.1e-16 dist=4.0e-15 ops=60   it=1 S=[0.1298, 0.1814, 0.7755, 1.2259, 2.2221]
8x8                    CGS   n=2   conv=2   res=1.2e-13 ov=5.9e-15 norm=2.2e-16 dist=2.2e-15 ops=60   it=1 S=[6.2117, 3.1038]
8x8 hm=3               CGS   n=8   conv=8   res=9.3e-14 ov=7.3e-15 norm=2.2e-16 dist=2.2e-15 ops=104  it=2 S=[6.2117, 3.1038, 2.8674, 2.2221, 1.2259, 0.7755, 0.1814, 0.1298]
8x8 :SR                CGS   n=5   conv=5   res=2.2e-14 ov=3.0e-14 norm=1.1e-16 dist=4.0e-15 ops=60   it=1 S=[0.1298, 0.1814, 0.7755, 1.2259, 2.2221]
400 kd=30 hm=1         MGS   n=2   conv=2   res=9.3e-14 ov=8.6e-16 norm=2.2e-16 dist=1.1e-13 ops=60   it=1 S=[100.0, 50.0]
400 kd=30 hm=2         MGS   n=2   conv=2   res=9.3e-14 ov=8.6e-16 norm=2.2e-16 dist=1.1e-13 ops=60   it=1 S=[100.0, 50.0]
400 kd=60 hm=2         MGS   n=2   conv=2   res=1.5e-13 ov=7.0e-15 norm=0.0e+00 dist=1.6e-13 ops=120  it=1 S=[100.0, 50.0]
400 kd=30 hm=1         CGS   n=2   conv=2   res=9.3e-14 ov=8.6e-16 norm=2.2e-16 dist=1.1e-13 ops=60   it=1 S=[100.0, 50.0]
400 kd=30 hm=2         CGS   n=2   conv=2   res=9.3e-14 ov=8.6e-16 norm=2.2e-16 dist=1.1e-13 ops=60   it=1 S=[100.0, 50.0]
400 kd=60 hm=2         CGS   n=2   conv=2   res=1.5e-13 ov=7.0e-15 norm=0.0e+00 dist=1.6e-13 ops=120  it=1 S=[100.0, 50.0]
400 rs kd=20 hm=2      MGS   n=2   conv=2   res=7.8e-14 ov=5.6e-17 norm=2.2e-16 dist=5.7e-14 ops=40   it=1 S=[100.0, 50.0]
400 rs kd=20 hm=3      MGS   n=3   conv=3   res=4.9e-11 ov=8.1e-13 norm=2.2e-16 dist=5.7e-14 ops=424  it=25 S=[100.0, 50.0, 1.0]
400 rs kd=40 hm=5      MGS   n=6   conv=6   res=2.1e-11 ov=5.9e-10 norm=2.2e-16 dist=8.5e-14 ops=462  it=13 S=[100.0, 50.0, 1.0, 0.9987, 0.9975, 0.9962]
diag                   MGS   n=8   conv=8   res=4.0e-13 ov=5.3e-14 norm=0.0e+00 dist=2.7e-15 ops=16   it=1 S=[8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
400 kd=8               MGS   n=2   conv=2   res=4.0e-14 ov=0.0e+00 norm=0.0e+00 dist=1.4e-14 ops=16   it=1 S=[100.0, 50.0]
deg6                   MGS   n=3   conv=3   res=1.9e-15 ov=4.7e-16 norm=0.0e+00 dist=1.3e-15 ops=6    it=1 S=[3.0, 2.0, 1.0]
deg σ=5×3              MGS   n=3   conv=2   res=1.2e-06 ov=6.1e-01 norm=0.0e+00 dist=1.8e-15 ops=60   it=1 S=[5.0, 5.0, 5.0]
sym8                   MGS   n=2   conv=2   res=1.9e-13 ov=7.6e-15 norm=0.0e+00 ops=30   it=1 vals=[3.5214, -2.3827]
sym8 hm=3              MGS   n=8   conv=8   res=1.4e-13 ov=4.1e-14 norm=2.2e-16 ops=52   it=2 vals=[3.5214, -2.3827, 2.1536, -1.9642, 0.8476, -0.3364, -0.1283, 0.0318]
sym8 :SR               MGS   n=6   conv=6   res=5.6e-14 ov=4.1e-14 norm=2.2e-16 ops=30   it=1 vals=[-2.3827, -1.9642, -0.3364, -0.1283, 0.0318, 0.8476]
sym8                   CGS   n=1   conv=1   res=1.4e-14 ov=0.0e+00 norm=0.0e+00 ops=30   it=1 vals=[3.5214]
sym8 hm=3              CGS   n=3   conv=0   res=4.8e-09 ov=2.3e-10 norm=1.1e-16 ops=52   it=2 vals=[3.5214, -2.3827, 2.1536]
sym8 :SR               CGS   n=7   conv=7   res=3.2e-14 ov=1.9e-15 norm=2.2e-16 ops=52   it=2 vals=[-2.3827, -1.9642, -0.3364, -0.1283, 0.0318, 0.8476, 2.1536]
400 kd=30 hm=1 it=1    MGS   n=1   conv=0   res=3.4e-09 ov=0.0e+00 norm=2.2e-16 ops=30   it=1 vals=[-104.1278]
400 kd=60 hm=2 it=1    MGS   n=4   conv=4   res=2.2e-13 ov=1.2e-16 norm=2.2e-16 ops=60   it=1 vals=[-104.1278, 95.8879, 50.1621, -49.7261]
400 kd=20 hm=3 it=100  MGS   n=4   conv=4   res=7.6e-12 ov=5.0e-15 norm=2.2e-16 ops=28   it=2 vals=[-104.1278, 95.8879, 50.1621, -49.7261]
```

The issue's example:

```
S = [100.00000000000004, 50.00000000000002]
info.converged = 2
norm.(V) = [1.0, 0.9999999999999999]
vals = [100.0, -50.00000000000005]
info.converged = 2
norm.(W) = [0.9999999999999999, 0.9999999999999999]
```

### Tests

Run on Julia 1.13.2: `test/svdsolve.jl` (353 pass), and the 10 of 15 testsets in `test/eigsolve.jl` with `Lanczos` in the name, including the new one (all pass). The other 5 are `Arnoldi` testsets on unchanged code, and were not run.

New tests cover `svdsolve` with a restart and `eigsolve`, on a 60×60 matrix with `ClassicalGramSchmidt` and `ModifiedGramSchmidt`, and a clean case with no warning. On `master` they fail: 6 pass, 12 fail and 2 error for `svdsolve`, and 2 pass, 6 fail and 2 error for `eigsolve`.

### Limits and open questions

- `normres` is a lower bound once orthogonality is lost: `f = 0` was measured with a true residual around `1e-13`.
- With restarts, returned vectors overlap up to `6e-10` and true residuals reach `5e-11`, for `tol = 1e-10`. `sym8 hm=3` with `ClassicalGramSchmidt` and `maxiter = 2` ends at `converged = 0`, honestly, with residual `5e-9`.
- Pairs overlapping by less than `1/√2` stay in the list. Overlap 0.61 remained for the true triple value.
- `GolubYe`/`geneigsolve` with plain `orth` weren't examined. `BlockLanczos` is unchanged.
- The `α = 0` problem is not fixed here: the hang on `[0 0; 0 1]` remains on this branch.

### Merging with `fix-svdsolve-gkl-breakdown`

`git merge-tree` reports a conflict only in `test/svdsolve.jl`, but `src/eigsolve/svdsolve.jl` auto-merges to something wrong: both define `K′`, and this branch's `K′ = K` overwrites the other's. This is from reading the code; a merged result was not run. Whichever merges second needs a rebase that combines them.

</details>
