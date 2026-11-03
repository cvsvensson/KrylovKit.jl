# `svdsolve`: false singular triplets, too many values, or a hang when the start vector overlaps the null space of `A'`

If the start vector has a component in the null space of `A'`, `svdsolve` can report a triplet as converged when its true residual is large, return more values than the matrix has, or never return. This covers any tall matrix, and any singular matrix with a generic start vector. It is separate from the issue "`svdsolve`/`eigsolve` with single-pass Gram-Schmidt return duplicate values".

```julia
using KrylovKit, LinearAlgebra

function check(A, x; kwargs...)
    S, U, V, info = svdsolve(A, x, 1, :LR; kwargs...)
    @show S info.normres info.converged svdvals(A)
    @show [norm(A * V[i] - S[i] * U[i]) for i in eachindex(S)]
    flush(stdout)
end

check([5.0 5 2; 5 5 2; 2 2 8], [1.0, 0, -1])  # x has no σ = 12 component
check([2.0 0; 0 1; 1 1], [1.0, 2, 3])         # 3×2: at most 2 singular values
check([0.0 0 -1 -1; 0 0 1 1; -1 1 4 0; -1 1 0 4], [1.0, 0, 0, 0]; krylovdim = 8, maxiter = 2)
check([0.0 0; 0 1], [1.0, 1])                 # never returns
```

Key output (the last call was killed after 45 s):

```
S = [11.999999999999998, 5.999999999999999, 2.3240656663666136e-46]
info.normres = [1.9720351111327974e-45, 0.0, 1.88411095042053e-15]
info.converged = 3
[norm(A * V[i] - S[i] * U[i]) for i = eachindex(S)] = [8.702335715267317e-15, 4.440892098500626e-16, 6.0]
S = [2.302775637731995, 1.3027756377319948, 1.0894704990131556e-32]
S = [4.82842712474619, 3.969233549842873, 0.8284271247461902, 1.1620312179068146e-32]
svdvals(A) = [4.828427124746188, 4.0, 0.8284271247461898, 1.0467283057891842e-16]
[norm(A * V[i] - S[i] * U[i]) for i = eachindex(S)] = [3.1401849173675503e-16, 0.030796011929079905, 4.1725779438208954e-16, 2.0000014082472264]
```

In the first call, `normres` is `1.9e-15` but the true residual is `6.0`. In the third, `3.969` is not a singular value at all.

**Expected:** stop at the breakdown, as for `β ≈ 0`, and return only valid triplets, never more than `minimum(size(A))`: `[6]`, `[2.30, 1.30]`, `[4.83, 0.83]` and `[1]`.

**Versions:** `master` at d51fa25, and v0.10.4 (the latest tag) with the same output. Julia 1.13.2, Windows 11. The digits depend on rounding.

**Proposed fix:** [`fix-svdsolve-gkl-breakdown`](https://github.com/cvsvensson/KrylovKit.jl/tree/fix-svdsolve-gkl-breakdown).

<details><summary>Details</summary>

### Cause

Measured: `svdsolve` only tests for breakdown on `β`, and `gklrecurrence` divides by `α` unguarded (`v = scale!!(v, inv(α))`).

Theory, which the traces below agree with: GKL from `x` in the codomain ends in one of two ways. With `d` the number of distinct nonzero singular values that `x` reaches:

- `x ⟂ null(A')`: `β_d = 0`, with `U` and `V` both of dimension `d`.
- Otherwise: `α_{d+1} = 0`, with `U` of dimension `d + 1` and `V` of dimension `d`, because `A' * u_{d+1}` lies in the span of `V`.

So `α = 0` is the normal ending for a tall or singular matrix with a generic start. There, `A * V_d = U_{d+1} * B̃` and `A' * U_{d+1} = V_d * B̃'` hold with a `(d+1)×d` bidiagonal `B̃`, so its `d` singular values are exact. Its left null vector is a null vector of `A'`, whose right partner is not in the subspace.

`normres = |β Q[i, k]|` relies on this factorization, so it misses the error once normalised noise enters `V`. Measured:

- **3×3:** at k = 3, `‖V'V - I‖` jumps to 1.4, and `‖A V - U B - r e_k'‖` jumps from `9e-15` to `6.0`, the false residual. Guess: reorthogonalising `r` against `U` removes components that `B` doesn't hold.
- **`[0 0; 0 1]`:** `α = 0` exactly at k = 2, so `v` and everything after it is `NaN`.
- **Hang:** inside `LAPACK.bdsqr!`. The block below involves no KrylovKit, prints `calling bdsqr!`, and did not return within 20 s.

```julia
using LinearAlgebra
B = Bidiagonal([0.707, 0.0, NaN], [0.707, NaN], :L)
n = 3
println("calling bdsqr!"); flush(stdout)
r = LAPACK.bdsqr!(B.uplo, B.dv, B.ev, Matrix(1.0I, n, n), Matrix(1.0I, n, n), zeros(n, 0))
println("returned ", r[1])
```

Not every `NaN` input hangs: `d = [1.0, NaN]`, `e = [NaN]` returned `[NaN, NaN]`, and `LAPACKException(7)` also occurs (see below). Why the patterns differ is not known.

### Settings tried on `master`

`[5 5 2; 5 5 2; 2 2 8]`, `x = [1, 0, -1]`, `krylovdim = 8`, `maxiter = 2`, unless the row says otherwise. Seeded rows use `Random.seed!(s)`, then `randn` for the matrix and start vector. The `IR` rows were killed after 20 s.

```
tol=1.0e-6                        S=[12.0, 6.0, 2.324e-46] conv=3 maxres=6.0
tol=1.0e-10                       S=[12.0, 6.0, 2.324e-46] conv=3 maxres=6.0
tol=1.0e-14                       S=[12.0, 6.0, 2.324e-46] conv=3 maxres=6.0
krylovdim=2                       S=[6.0] conv=1 maxres=4.4e-16
krylovdim=3                       S=[12.0, 6.0, 2.324e-46] conv=3 maxres=6.0
krylovdim=4                       S=[12.0, 6.0, 2.324e-46] conv=3 maxres=6.0
krylovdim=8                       S=[12.0, 6.0, 2.324e-46] conv=3 maxres=6.0
krylovdim=30                      S=[12.0, 6.0, 2.324e-46] conv=3 maxres=6.0
eager=true                        S=[6.0] conv=1 maxres=4.4e-16
ModifiedGramSchmidt2              S=[12.0, 6.0, 2.324e-46] conv=3 maxres=6.0
ClassicalGramSchmidt2             S=[12.0, 6.0, 6.28e-16] conv=3 maxres=6.0
ModifiedGramSchmidt               S=[12.0, 12.0, 12.0, 6.0, 6.0, 6.0, 6.0] conv=7 maxres=6.7e-15
ClassicalGramSchmidt              S=[12.0, 12.0, 12.0, 6.0, 6.0, 6.0, 6.0] conv=7 maxres=6.7e-15
3x3 seed=1 krylovdim=8            THREW LAPACKException(7)
3x3 seed=2 krylovdim=8            S=[12.0, 6.0, 4.679999999999999e-31] conv=3 maxres=10.0
3x3 seed=3 krylovdim=8            S=[12.0, 6.0, 2.64e-32] conv=3 maxres=6.9
3x3 seed=4 krylovdim=8            S=[12.0, 6.0, 2.108e-31] conv=3 maxres=6.5
3x3 seed=1 default                THREW LAPACKException(29)
8x3 seed=1 krylovdim=5            S=[4.112, 3.652, 2.031, 1.916e-32] conv=4 maxres=3.7
8x3 seed=2 krylovdim=5            S=[3.756, 2.573, 2.329, 1.3089999999999999e-33] conv=4 maxres=2.6
rank-2 8x8 seed=1 krylovdim=5     S=[7.436, 4.616, 3.871e-16] conv=3 maxres=3.4e-15
full-rank 4x4 seed=1              S=[3.462, 2.209, 1.798, 0.3855] conv=4 maxres=1.4e-15
wide [1 1 1; 1 1 1]               S=[2.449, 2.7000000000000004e-32] conv=2 maxres=2.4
complex 3x3                       S=[2.0, 2.092e-31] conv=2 maxres=2.0
A + I (control)                   S=[7.0, 1.0] conv=2 maxres=2.0e-15
ModifiedGramSchmidtIR             did not return
ClassicalGramSchmidtIR            did not return
[1 1; 1 1], x=[1,-1]              THREW LAPACKException(7)
```

These trace back to `α`:

- **`krylovdim = 2`, `eager = true`:** clean, because they stop before the breakdown.
- **`*IR`:** `α` reaches `1e-323` or `0`, then `Inf` or `NaN`, then the hang in `bdsqr!`.
- **Random start, seed 1:** `α = 0` at k = 3, then `NaN`, and `bdsqr!` throws.
- **Plain `ModifiedGramSchmidt`, `ClassicalGramSchmidt`:** noise `α = 7.7e-16` is normalised at k = 2, giving `‖V'V - I‖ = 1.3`.

### Recurrence traces on `master`

```
== A size (3, 3) orth ModifiedGramSchmidt2 svdvals=[12.0, 6.0, 0.0]
k=1 α=5.20e+00 β=3.00e+00 |U'U-I|=2.2e-16 |V'V-I|=5.6e-16 |AV-UB-re'|=6.3e-16 |A'U-VB'|=1.1e-15
k=2 α=2.56e-16 β=1.20e+01 |U'U-I|=5.2e-16 |V'V-I|=2.2e-15 |AV-UB-re'|=8.7e-15 |A'U-VB'|=1.2e-15
k=3 α=1.26e-29 β=1.88e-15 |U'U-I|=5.7e-16 |V'V-I|=1.4e+00 |AV-UB-re'|=6.0e+00 |A'U-VB'|=2.0e-14
k=4 α=4.78e-30 β=3.08e-15 |U'U-I|=1.4e+00 |V'V-I|=2.0e+00 |AV-UB-re'|=1.3e+01 |A'U-VB'|=5.2e+00
k=5 α=2.58e-30 β=3.63e-16 |U'U-I|=2.0e+00 |V'V-I|=2.8e+00 |AV-UB-re'|=1.5e+01 |A'U-VB'|=1.3e+01
== A size (3, 2) orth ModifiedGramSchmidt2 svdvals=[2.3028, 1.3028]
k=1 α=1.89e+00 β=9.64e-01 |U'U-I|=0.0e+00 |V'V-I|=2.2e-16 |AV-UB-re'|=0.0e+00 |A'U-VB'|=0.0e+00
k=2 α=1.56e+00 β=2.77e-01 |U'U-I|=5.6e-16 |V'V-I|=2.6e-16 |AV-UB-re'|=5.6e-16 |A'U-VB'|=6.3e-16
k=3 α=1.11e-32 β=3.93e-16 |U'U-I|=5.8e-16 |V'V-I|=1.4e+00 |AV-UB-re'|=2.2e+00 |A'U-VB'|=6.7e-16
k=4 α=1.54e-16 β=2.61e-16 |U'U-I|=1.4e+00 |V'V-I|=2.0e+00 |AV-UB-re'|=2.6e+00 |A'U-VB'|=2.3e+00
k=5 α=0.00e+00 β=NaN |U'U-I|=2.0e+00 |V'V-I|=NaN |AV-UB-re'|=NaN |A'U-VB'|=NaN
NaN, stop
== A size (2, 2) orth ModifiedGramSchmidt2 svdvals=[1.0, 0.0]
k=1 α=7.07e-01 β=7.07e-01 |U'U-I|=2.2e-16 |V'V-I|=0.0e+00 |AV-UB-re'|=0.0e+00 |A'U-VB'|=0.0e+00
k=2 α=0.00e+00 β=NaN |U'U-I|=3.3e-16 |V'V-I|=NaN |AV-UB-re'|=NaN |A'U-VB'|=NaN
NaN, stop
== A size (3, 3) orth ClassicalGramSchmidt2 svdvals=[12.0, 6.0, 0.0]
k=1 α=5.20e+00 β=3.00e+00 |U'U-I|=2.2e-16 |V'V-I|=5.6e-16 |AV-UB-re'|=6.3e-16 |A'U-VB'|=1.1e-15
k=2 α=7.69e-16 β=4.00e+00 |U'U-I|=5.2e-16 |V'V-I|=1.3e+00 |AV-UB-re'|=5.7e+00 |A'U-VB'|=1.3e-15
k=3 α=1.13e+01 β=1.31e-15 |U'U-I|=1.0e-15 |V'V-I|=1.4e+00 |AV-UB-re'|=6.0e+00 |A'U-VB'|=1.4e-15
k=4 α=1.04e+01 β=6.17e-15 |U'U-I|=1.4e+00 |V'V-I|=2.1e+00 |AV-UB-re'|=7.7e+00 |A'U-VB'|=1.5e-15
k=5 α=1.12e+01 β=6.70e-01 |U'U-I|=2.4e+00 |V'V-I|=2.9e+00 |AV-UB-re'|=8.6e+00 |A'U-VB'|=1.5e-15
== A size (3, 3) orth ModifiedGramSchmidt svdvals=[12.0, 6.0, 0.0]
k=1 α=5.20e+00 β=3.00e+00 |U'U-I|=2.2e-16 |V'V-I|=5.6e-16 |AV-UB-re'|=6.3e-16 |A'U-VB'|=1.1e-15
k=2 α=7.69e-16 β=6.93e+00 |U'U-I|=5.2e-16 |V'V-I|=1.3e+00 |AV-UB-re'|=6.3e-16 |A'U-VB'|=1.3e-15
k=3 α=4.90e+00 β=1.04e+01 |U'U-I|=1.2e+00 |V'V-I|=1.4e+00 |AV-UB-re'|=6.3e-16 |A'U-VB'|=1.2e-15
k=4 α=3.55e-15 β=8.49e+00 |U'U-I|=1.4e+00 |V'V-I|=2.3e+00 |AV-UB-re'|=6.3e-16 |A'U-VB'|=1.8e-15
k=5 α=6.00e+00 β=8.49e+00 |U'U-I|=2.2e+00 |V'V-I|=2.8e+00 |AV-UB-re'|=6.3e-16 |A'U-VB'|=1.8e-15
k=6 α=2.30e-15 β=6.01e+00 |U'U-I|=2.8e+00 |V'V-I|=3.7e+00 |AV-UB-re'|=6.3e-16 |A'U-VB'|=1.8e-15
k=7 α=6.86e-01 β=1.20e+01 |U'U-I|=3.7e+00 |V'V-I|=4.2e+00 |AV-UB-re'|=6.3e-16 |A'U-VB'|=1.8e-15
== A size (3, 3) orth ClassicalGramSchmidt svdvals=[12.0, 6.0, 0.0]
k=1 α=5.20e+00 β=3.00e+00 |U'U-I|=2.2e-16 |V'V-I|=5.6e-16 |AV-UB-re'|=6.3e-16 |A'U-VB'|=1.1e-15
k=2 α=7.69e-16 β=6.93e+00 |U'U-I|=5.2e-16 |V'V-I|=1.3e+00 |AV-UB-re'|=6.3e-16 |A'U-VB'|=1.3e-15
k=3 α=4.90e+00 β=1.04e+01 |U'U-I|=1.2e+00 |V'V-I|=1.4e+00 |AV-UB-re'|=6.3e-16 |A'U-VB'|=1.2e-15
k=4 α=3.55e-15 β=8.49e+00 |U'U-I|=1.4e+00 |V'V-I|=2.3e+00 |AV-UB-re'|=6.3e-16 |A'U-VB'|=1.8e-15
k=5 α=6.00e+00 β=8.49e+00 |U'U-I|=2.2e+00 |V'V-I|=2.8e+00 |AV-UB-re'|=6.3e-16 |A'U-VB'|=1.8e-15
k=6 α=2.30e-15 β=6.01e+00 |U'U-I|=2.8e+00 |V'V-I|=3.7e+00 |AV-UB-re'|=6.3e-16 |A'U-VB'|=1.8e-15
k=7 α=6.86e-01 β=1.20e+01 |U'U-I|=3.7e+00 |V'V-I|=4.2e+00 |AV-UB-re'|=6.3e-16 |A'U-VB'|=1.8e-15
== A size (3, 3) orth ModifiedGramSchmidtIR svdvals=[12.0, 6.0, 0.0]
k=1 α=5.20e+00 β=3.00e+00 |U'U-I|=2.2e-16 |V'V-I|=5.6e-16 |AV-UB-re'|=6.3e-16 |A'U-VB'|=1.1e-15
k=2 α=2.56e-16 β=1.20e+01 |U'U-I|=5.2e-16 |V'V-I|=5.7e-16 |AV-UB-re'|=6.3e-16 |A'U-VB'|=1.2e-15
k=3 α=1.31e-30 β=1.09e+01 |U'U-I|=5.7e-16 |V'V-I|=1.4e+00 |AV-UB-re'|=6.3e-16 |A'U-VB'|=3.9e-15
k=4 α=8.41e-31 β=7.89e+00 |U'U-I|=1.4e+00 |V'V-I|=2.0e+00 |AV-UB-re'|=1.1e-15 |A'U-VB'|=4.3e+00
k=5 α=2.42e-31 β=1.15e+01 |U'U-I|=2.1e+00 |V'V-I|=2.8e+00 |AV-UB-re'|=1.1e-15 |A'U-VB'|=7.3e+00
== A size (3, 3) orth ClassicalGramSchmidtIR svdvals=[12.0, 6.0, 0.0]
k=1 α=5.20e+00 β=3.00e+00 |U'U-I|=2.2e-16 |V'V-I|=5.6e-16 |AV-UB-re'|=6.3e-16 |A'U-VB'|=1.1e-15
k=2 α=2.56e-16 β=1.20e+01 |U'U-I|=5.2e-16 |V'V-I|=5.7e-16 |AV-UB-re'|=6.3e-16 |A'U-VB'|=1.2e-15
k=3 α=9.88e-324 β=Inf |U'U-I|=5.7e-16 |V'V-I|=NaN |AV-UB-re'|=NaN |A'U-VB'|=NaN
k=4 α=Inf β=NaN |U'U-I|=1.0e+00 |V'V-I|=NaN |AV-UB-re'|=NaN |A'U-VB'|=NaN
NaN, stop
== A size (2, 2) orth ModifiedGramSchmidtIR svdvals=[1.0, 0.0]
k=1 α=7.07e-01 β=7.07e-01 |U'U-I|=2.2e-16 |V'V-I|=0.0e+00 |AV-UB-re'|=0.0e+00 |A'U-VB'|=0.0e+00
k=2 α=0.00e+00 β=NaN |U'U-I|=3.3e-16 |V'V-I|=NaN |AV-UB-re'|=NaN |A'U-VB'|=NaN
NaN, stop
== A size (2, 2) orth ClassicalGramSchmidtIR svdvals=[1.0, 0.0]
k=1 α=7.07e-01 β=7.07e-01 |U'U-I|=2.2e-16 |V'V-I|=0.0e+00 |AV-UB-re'|=0.0e+00 |A'U-VB'|=0.0e+00
k=2 α=0.00e+00 β=NaN |U'U-I|=3.3e-16 |V'V-I|=NaN |AV-UB-re'|=NaN |A'U-VB'|=NaN
NaN, stop
```

### Full output of the example on `master`

The last call never returns.

```
S = [11.999999999999998, 5.999999999999999, 2.3240656663666136e-46]
info.normres = [1.9720351111327974e-45, 0.0, 1.88411095042053e-15]
info.converged = 3
svdvals(A) = [12.0, 6.0, 0.0]
[norm(A * V[i] - S[i] * U[i]) for i = eachindex(S)] = [8.702335715267317e-15, 4.440892098500626e-16, 6.0]
S = [2.302775637731995, 1.3027756377319948, 1.0894704990131556e-32]
info.normres = [0.0, 0.0, 3.925844416168177e-16]
info.converged = 3
svdvals(A) = [2.3027756377319952, 1.3027756377319946]
[norm(A * V[i] - S[i] * U[i]) for i = eachindex(S)] = [0.0, 7.28021923224409e-16, 2.23606797749979]
S = [4.82842712474619, 3.969233549842873, 0.8284271247461902, 1.1620312179068146e-32]
info.normres = [0.0, 5.592508159223071e-17, 0.0, 4.9817437224255605e-18]
info.converged = 4
svdvals(A) = [4.828427124746188, 4.0, 0.8284271247461898, 1.0467283057891842e-16]
[norm(A * V[i] - S[i] * U[i]) for i = eachindex(S)] = [3.1401849173675503e-16, 0.030796011929079905, 4.1725779438208954e-16, 2.0000014082472264]
```

### Related

[PR #176](https://github.com/Jutho/KrylovKit.jl/pull/176), merged after 0.10.4, changed how `β` breakdown with `K < howmany` is reported. It did not touch `gkl.jl`, and all of the above reproduces after it.

</details>

---

# `svdsolve`/`eigsolve` with single-pass Gram-Schmidt return duplicate values

With `ClassicalGramSchmidt` or `ModifiedGramSchmidt`, `GKL` and `Lanczos` return many copies of a converged value, count them all as converged, and leave out true values. Vectors can be far from unit norm. It needs no null space and no large Krylov dimension. The default `ModifiedGramSchmidt2` is not affected. It is separate from the issue "`svdsolve`: false singular triplets, too many values, or a hang when the start vector overlaps the null space of `A'`".

```julia
using KrylovKit, LinearAlgebra, Random

Random.seed!(1)
Q = Matrix(qr(randn(60, 60)).Q)
H = Q * Diagonal([100.0; -50.0; range(1, 0.5, 58)]) * Q'
H = (H + H') / 2
x = randn(60)
kwargs = (; orth = ModifiedGramSchmidt(), krylovdim = 30, maxiter = 1, tol = 1e-10, verbosity = 0)

S, U, V, info = svdsolve(H, x, 2, :LR; kwargs...)
@show S info.converged norm.(V)
vals, W, info = eigsolve(H, x, 2, :LM, Lanczos(; kwargs...))
@show vals info.converged norm.(W)
```

Output:

```
S = [100.00000000000004, 100.00000000000003, 100.00000000000003, 100.00000000000001, 100.00000000000001]
info.converged = 5
norm.(V) = [1.0000000000000024, 1.0000000000000022, 0.9999999999999964, 1.0000000000000004, 1.0000000000000007]
vals = [100.00000000000003, 100.0, 99.99999999999999, 99.99999999999999]
info.converged = 4
norm.(W) = [0.5324878350705626, 1.4459916767380652, 0.3006955935787294, 1.2390104665855097]
```

**Expected:** `[100, 50]` from `svdsolve` and `[100, -50]` from `eigsolve`, with unit vectors. `ModifiedGramSchmidt2` gives exactly this on the same call.

**Versions:** `master` at d51fa25, Julia 1.13.2, Windows 11. The digits and the number of copies depend on rounding.

**Proposed fix:** [`fix-gram-schmidt-ghost-values`](https://github.com/cvsvensson/KrylovKit.jl/tree/fix-gram-schmidt-ghost-values).

<details><summary>Details</summary>

### Cause

Measured: `‖V'V - I‖` grows from `2e-16` to order 1 as the smallest Ritz residual falls, while `α` stays far from zero. Copies agree to 10 digits with Ritz vector overlap 1.000, against 0.000 for distinct values. `Arnoldi` with `ModifiedGramSchmidt` orthogonalizes against the whole basis and was clean on the symmetric 8×8 matrix below.

This matches Paige's theory of the Lanczos process: orthogonality is lost as Ritz values converge, and copies ("ghosts") follow. That the theory is the full explanation is an inference from these measurements.

Each copy has a small true residual, so `converged` is correct about the residual criterion. The harm is that copies count against `howmany` and push out true values, and that `norm(v)` can be far from 1.

### Trace on `master`

Full-rank 8×8 from `Random.seed!(1)`, with `randn(8, 3)` and `randn(8)` drawn and discarded first. There is no null space, and `α` is never small. `αratio` is `α / (eps * max(αs, βs))`, and `minRitzres` is the smallest Ritz residual.

```
-- trace MGS size (8, 8) σ=[6.212, 3.104, 2.867, 2.222, 1.226, 0.775, 0.181, 0.13]
k= 1 α=1.51e+00 β=3.28e+00 αratio=2.1e+15 |U'U-I|=0.0e+00 |V'V-I|=2.2e-16 |AV-UB-re'|=4.0e-16 minRitzres=3.3e+00
k= 2 α=4.34e+00 β=3.24e+00 αratio=4.5e+15 |U'U-I|=1.5e-17 |V'V-I|=3.9e-16 |AV-UB-re'|=7.6e-16 minRitzres=2.1e+00
k= 3 α=2.04e+00 β=1.47e+00 αratio=2.1e+15 |U'U-I|=7.3e-16 |V'V-I|=1.5e-15 |AV-UB-re'|=8.1e-16 minRitzres=2.4e-01
k= 4 α=1.51e+00 β=1.76e+00 αratio=1.6e+15 |U'U-I|=5.4e-15 |V'V-I|=2.0e-14 |AV-UB-re'|=9.2e-16 minRitzres=1.9e-02
k= 5 α=1.18e+00 β=2.00e+00 αratio=1.2e+15 |U'U-I|=6.7e-14 |V'V-I|=3.2e-13 |AV-UB-re'|=9.6e-16 minRitzres=1.3e-03
k= 6 α=1.71e+00 β=9.26e-01 αratio=1.8e+15 |U'U-I|=9.5e-13 |V'V-I|=3.1e-12 |AV-UB-re'|=1.1e-15 minRitzres=6.7e-05
k= 7 α=4.97e-01 β=4.10e-02 αratio=5.2e+14 |U'U-I|=1.9e-11 |V'V-I|=2.3e-10 |AV-UB-re'|=1.1e-15 minRitzres=3.7e-08
k= 8 α=1.35e-01 β=7.01e-06 αratio=1.4e+14 |U'U-I|=3.5e-08 |V'V-I|=1.6e-06 |AV-UB-re'|=1.1e-15 minRitzres=9.0e-16
k= 9 α=6.21e+00 β=2.65e-05 αratio=4.5e+15 |U'U-I|=1.4e+00 |V'V-I|=1.4e+00 |AV-UB-re'|=1.2e-15 minRitzres=3.7e-20
k=10 α=3.09e+00 β=1.19e-01 αratio=2.2e+15 |U'U-I|=2.0e+00 |V'V-I|=2.0e+00 |AV-UB-re'|=1.4e-15 minRitzres=4.2e-22
k=11 α=2.87e+00 β=2.08e-01 αratio=2.1e+15 |U'U-I|=2.4e+00 |V'V-I|=2.4e+00 |AV-UB-re'|=1.4e-15 minRitzres=5.6e-24
k=12 α=1.56e+00 β=9.80e-01 αratio=1.1e+15 |U'U-I|=2.8e+00 |V'V-I|=2.8e+00 |AV-UB-re'|=1.4e-15 minRitzres=2.0e-25
```

### Results on `master`

`ov` is `max|v_i' v_j|` over returned vectors, `norm` is the largest `|‖v‖ - 1|`, and `dist` is the distance of converged values to the true ones. `krylovdim = 30`, `maxiter = 2` unless named.

- `8x8`, `sym8`: the seeded matrix above and its symmetric part.
- `400 ...`: 400×400 with singular values `[100; 50; range(1, 0.5, 398)]`. Duplicates do not need `krylovdim` to exceed the space dimension.
- `deg σ=5×3`: a true triple value. `deg6`: `Diagonal([1, 1, 2, 2, 3, 3])`.

```
8x8                    MGS   n=8   conv=8   res=3.4e-13 ov=1.1e+00 norm=9.2e-01 dist=5.3e-15 ops=60   it=1 S=[6.2117, 6.2117, 6.2117, 6.2117, 6.2117, 3.1038, 3.1038, 3.1038]
8x8 hm=3               MGS   n=8   conv=8   res=3.4e-13 ov=1.1e+00 norm=9.2e-01 dist=5.3e-15 ops=60   it=1 S=[6.2117, 6.2117, 6.2117, 6.2117, 6.2117, 3.1038, 3.1038, 3.1038]
8x8 :SR                MGS   n=1   conv=1   res=1.6e-13 ov=0.0e+00 norm=6.1e-03 dist=4.4e-16 ops=60   it=1 S=[0.1298]
8x8                    CGS   n=8   conv=8   res=3.4e-13 ov=1.1e+00 norm=9.2e-01 dist=5.3e-15 ops=60   it=1 S=[6.2117, 6.2117, 6.2117, 6.2117, 6.2117, 3.1038, 3.1038, 3.1038]
8x8 hm=3               CGS   n=8   conv=8   res=3.4e-13 ov=1.1e+00 norm=9.2e-01 dist=5.3e-15 ops=60   it=1 S=[6.2117, 6.2117, 6.2117, 6.2117, 6.2117, 3.1038, 3.1038, 3.1038]
8x8 :SR                CGS   n=1   conv=1   res=1.6e-13 ov=0.0e+00 norm=6.1e-03 dist=4.4e-16 ops=60   it=1 S=[0.1298]
400 kd=30 hm=1         MGS   n=4   conv=4   res=7.2e-14 ov=1.0e+00 norm=3.8e-15 dist=1.1e-13 ops=60   it=1 S=[100.0, 100.0, 100.0, 100.0]
400 kd=30 hm=2         MGS   n=4   conv=4   res=7.2e-14 ov=1.0e+00 norm=3.8e-15 dist=1.1e-13 ops=60   it=1 S=[100.0, 100.0, 100.0, 100.0]
400 kd=60 hm=2         MGS   n=7   conv=7   res=2.1e-13 ov=1.0e+00 norm=6.9e-15 dist=1.6e-13 ops=120  it=1 S=[100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
400 kd=30 hm=1         CGS   n=4   conv=4   res=7.2e-14 ov=1.0e+00 norm=3.8e-15 dist=1.1e-13 ops=60   it=1 S=[100.0, 100.0, 100.0, 100.0]
400 kd=30 hm=2         CGS   n=4   conv=4   res=7.2e-14 ov=1.0e+00 norm=3.8e-15 dist=1.1e-13 ops=60   it=1 S=[100.0, 100.0, 100.0, 100.0]
400 kd=60 hm=2         CGS   n=7   conv=7   res=2.1e-13 ov=1.0e+00 norm=6.9e-15 dist=1.6e-13 ops=120  it=1 S=[100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
400 rs kd=20 hm=2      MGS   n=3   conv=3   res=5.4e-14 ov=1.0e+00 norm=3.3e-15 dist=5.7e-14 ops=40   it=1 S=[100.0, 100.0, 100.0]
400 rs kd=20 hm=3      MGS   n=3   conv=3   res=5.4e-14 ov=1.0e+00 norm=3.3e-15 dist=5.7e-14 ops=40   it=1 S=[100.0, 100.0, 100.0]
400 rs kd=40 hm=5      MGS   n=6   conv=6   res=1.2e-13 ov=1.0e+00 norm=4.2e-15 dist=1.3e-13 ops=110  it=2 S=[100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
diag                   MGS   n=8   conv=8   res=4.0e-13 ov=5.3e-14 norm=8.9e-16 dist=2.7e-15 ops=16   it=1 S=[8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
400 kd=8               MGS   n=1   conv=1   res=3.8e-14 ov=0.0e+00 norm=6.7e-16 dist=0.0e+00 ops=16   it=1 S=[100.0]
deg6                   MGS   n=3   conv=3   res=1.5e-15 ov=4.7e-16 norm=4.4e-16 dist=1.3e-15 ops=6    it=1 S=[3.0, 2.0, 1.0]
deg σ=5×3              MGS   n=3   conv=3   res=4.8e-15 ov=8.2e-01 norm=4.7e-15 dist=1.8e-15 ops=60   it=1 S=[5.0, 5.0, 5.0]
sym8                   MGS   n=4   conv=4   res=2.2e-15 ov=1.0e+00 norm=9.9e-01 ops=30   it=1 vals=[3.5214, 3.5214, 3.5214, 3.5214]
sym8 hm=3              MGS   n=4   conv=4   res=2.2e-15 ov=1.0e+00 norm=9.9e-01 ops=30   it=1 vals=[3.5214, 3.5214, 3.5214, 3.5214]
sym8 :SR               MGS   n=10  conv=10  res=3.5e-14 ov=1.3e+00 norm=2.6e-01 ops=42   it=2 vals=[-2.3827, -2.3827, -2.3827, -2.3827, -2.3827, -1.9642, -1.9642, -1.9642, -1.
sym8                   CGS   n=4   conv=4   res=6.4e-13 ov=1.3e+00 norm=4.4e-01 ops=30   it=1 vals=[3.5214, 3.5214, 3.5214, 3.5214]
sym8 hm=3              CGS   n=4   conv=4   res=6.4e-13 ov=1.3e+00 norm=4.4e-01 ops=30   it=1 vals=[3.5214, 3.5214, 3.5214, 3.5214]
sym8 :SR               CGS   n=10  conv=10  res=3.2e-14 ov=1.9e+00 norm=6.7e-01 ops=42   it=2 vals=[-2.3827, -2.3827, -2.3827, -2.3827, -2.3827, -1.9642, -1.9642, -1.9642, -1.
400 kd=30 hm=1 it=1    MGS   n=1   conv=0   res=3.4e-09 ov=0.0e+00 norm=1.5e-02 ops=30   it=1 vals=[-104.1278]
400 kd=60 hm=2 it=1    MGS   n=2   conv=0   res=2.0e-10 ov=5.8e-01 norm=3.8e-01 ops=60   it=1 vals=[-104.1278, -104.1278]
400 kd=20 hm=3 it=100  MGS   n=6   conv=6   res=1.4e-12 ov=1.3e+00 norm=4.5e-01 ops=36   it=3 vals=[-104.1278, -104.1278, -104.1278, 95.8879, 95.8879, 95.8879]
```

### Documentation

`ModifiedGramSchmidt` is described as "typically a reasonable choice for linear systems but not for eigenvalue solvers with a large Krylov dimension", and `ClassicalGramSchmidt` as "typically not an optimal choice". Neither mentions duplicates.

</details>
