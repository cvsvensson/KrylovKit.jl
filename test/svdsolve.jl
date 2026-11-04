@testset "GKL - svdsolve full ($mode)" for mode in (:vector, :inplace, :outplace, :mixed)
    scalartypes = mode === :vector ? (Float32, Float64, ComplexF32, ComplexF64) :
        (ComplexF64,)
    orths = mode === :vector ? (cgs2, mgs2, cgsr, mgsr) : (mgsr,)
    @testset for T in scalartypes
        @testset for orth in orths
            A = rand(T, (n, n))
            alg = GKL(; orth = orth, krylovdim = 2 * n, maxiter = 1, tol = tolerance(T))
            S, lvecs, rvecs, info = @constinferred svdsolve(
                wrapop(A, Val(mode)),
                wrapvec(A[:, 1], Val(mode)), n,
                :LR, alg
            )
            @test S ≈ svdvals(A)
            @test info.converged == n

            n1 = div(n, 2)
            @test_logs svdsolve(
                wrapop(A, Val(mode)), wrapvec(A[:, 1], Val(mode)), n1, :LR,
                alg
            )
            alg = GKL(;
                orth = orth, krylovdim = 2 * n, maxiter = 1, tol = tolerance(T),
                verbosity = WARN_LEVEL
            )
            @test_logs svdsolve(
                wrapop(A, Val(mode)), wrapvec(A[:, 1], Val(mode)), n1, :LR,
                alg
            )
            alg = GKL(;
                orth = orth, krylovdim = n1 + 1, maxiter = 1, tol = tolerance(T),
                verbosity = WARN_LEVEL
            )
            @test_logs (:warn,) svdsolve(
                wrapop(A, Val(mode)), wrapvec(A[:, 1], Val(mode)),
                n1, :LR,
                alg
            )
            alg = GKL(;
                orth = orth, krylovdim = 2 * n, maxiter = 1, tol = tolerance(T),
                verbosity = STARTSTOP_LEVEL
            )
            @test_logs (:info,) svdsolve(
                wrapop(A, Val(mode)), wrapvec(A[:, 1], Val(mode)),
                n1, :LR,
                alg
            )
            alg = GKL(;
                orth = orth, krylovdim = 2 * n, maxiter = 1, tol = tolerance(T),
                verbosity = EACHITERATION_LEVEL + 1
            )
            @test_logs min_level = Logging.Warn svdsolve(
                wrapop(A, Val(mode)),
                wrapvec(A[:, 1], Val(mode)),
                n1, :LR,
                alg
            )

            U = stack(unwrapvec, lvecs)
            V = stack(unwrapvec, rvecs)
            @test U' * U ≈ I
            @test V' * V ≈ I
            @test A * V ≈ U * Diagonal(S)
        end
    end
end

@testset "GKL - svdsolve iteratively ($mode)" for mode in
    (:vector, :inplace, :outplace, :mixed)
    scalartypes = mode === :vector ? (Float32, Float64, ComplexF32, ComplexF64) :
        (ComplexF64,)
    orths = mode === :vector ? (cgs2, mgs2, cgsr, mgsr) : (mgsr,)
    @testset for T in scalartypes
        @testset for orth in orths
            A = rand(T, (2 * N, N))
            v = rand(T, (2 * N,))
            n₁ = div(n, 2)
            alg = GKL(;
                orth = orth, krylovdim = n, maxiter = 10, tol = tolerance(T), eager = true,
                verbosity = SILENT_LEVEL
            )
            S, lvecs, rvecs, info = @constinferred svdsolve(
                wrapop(A, Val(mode)),
                wrapvec(v, Val(mode)),
                n₁, :LR, alg
            )

            l = info.converged
            @test S[1:l] ≈ svdvals(A)[1:l]

            U = stack(unwrapvec, lvecs)
            V = stack(unwrapvec, rvecs)
            @test U[:, 1:l]' * U[:, 1:l] ≈ I
            @test V[:, 1:l]' * V[:, 1:l] ≈ I

            R = stack(unwrapvec, info.residual)
            @test A' * U ≈ V * Diagonal(S)
            @test A * V ≈ U * Diagonal(S) + R
        end
    end
end

@testset "GKL - svdsolve with an invariant subspace smaller than howmany" begin
    # with degenerate singular values, the Krylov subspace of a single start vector becomes
    # invariant after one direction per distinct singular value
    A = Diagonal([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
    alg = GKL(; krylovdim = 6, tol = 1.0e-12, verbosity = WARN_LEVEL)
    S, lvecs, rvecs, info = @test_logs (:warn, r"returning 3 values") svdsolve(A, ones(6), 4, :LR, alg)
    @test S ≈ [3.0, 2.0, 1.0]
    @test length(lvecs) == length(rvecs) == length(info.normres) == 3
    @test info.converged == 3
    for (s, u, v) in zip(S, lvecs, rvecs)
        @test A * v ≈ s * u
        @test A' * u ≈ s * v
    end
end

@testset "GKL - svdsolve when the recurrence ends on α == 0" begin
    # all arithmetic is exact, so that `α == 0` does not depend on rounding
    A = [2.0 2; 2 0; 0 0; 0 -2]
    x₀ = ones(4)
    vals = [2 * sqrt(3), 2.0]
    @testset "$mode" for mode in (:vector, :inplace, :outplace, :mixed)
        S, lvecs, rvecs, info = @constinferred svdsolve(
            wrapop(A, Val(mode)), wrapvec(x₀, Val(mode)), 1, :LR, GKL()
        )
        @test S ≈ vals
        @test info.converged == 2
        for (s, u, v) in zip(S, lvecs, rvecs)
            @test A * unwrapvec(v) ≈ s * unwrapvec(u)
            @test A' * unwrapvec(u) ≈ s * unwrapvec(v)
        end
    end
    @testset "$orth" for orth in (cgs2, mgs2, cgsr, mgsr)
        S, = svdsolve(A, x₀, 1, :LR, GKL(; orth = orth))
        @test S ≈ vals
    end
    @testset "$which, howmany = 3" for (which, S′) in ((:LR, vals), (:SR, reverse(vals)))
        S, = @test_logs (:warn, r"returning 2 values") svdsolve(A, x₀, 3, which)
        @test S ≈ S′
    end
    @testset "one column" begin
        S, = svdsolve(ones(2, 1), [1.0, 0.0])
        @test S ≈ [sqrt(2)]
    end
    @testset "start vector in the null space of A'" begin
        S, lvecs, rvecs, = @test_logs (:warn, r"returning 0 values") svdsolve(zeros(3, 3), ones(3))
        @test isempty(S) && isempty(lvecs) && isempty(rvecs)
    end
end
