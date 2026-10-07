# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# JSON bridge only: the published library performs every TCI update.
using JSON3, LinearAlgebra, Random
import TensorCrossInterpolation as TCI

function construct_target(spec, n, options, seed)
    Random.seed!(seed)
    samples = Dict{Tuple,ComplexF64}()
    requests = Ref(0)
    function amplitude(bits)
        requests[] += 1
        get!(samples, Tuple(bits)) do
            if spec.kind == "gaussian"
                x = sum((bits[j] - 1) * 2.0^(-j) for j in 1:n)
                ComplexF64(exp(-spec.coordinate * (x - 0.5)^2 / 0.04))
            else
                index = 1 + sum((bits[j] - 1) << (j - 1) for j in 1:n)
                ComplexF64(spec.real[index], spec.imag[index])
            end
        end
    end
    pivot = spec.kind == "gaussian" ? ones(Int, n) : Int.(spec.initial_pivot)
    started = time_ns()
    tci, ranks, errors = TCI.crossinterpolate2(
        ComplexF64, amplitude, fill(2, n), [pivot];
        tolerance=Float64(options.tolerance), maxbonddim=Int(options.max_bond),
        maxiter=Int(options.max_iterations), pivotsearch=:full,
        sweepstrategy=:backandforth, normalizeerror=true,
    )
    cores = TCI.TensorTrain(tci).sitetensors
    encode(t) = Dict("shape" => size(t), "real" => vec(real.(t)), "imag" => vec(imag.(t)))
    Dict(
        "cores" => encode.(cores), "ranks" => ranks, "errors" => errors,
        "max_sample_value" => tci.maxsamplevalue,
        "pivot_errors" => tci.pivoterrors,
        "function_requests" => requests[], "unique_function_evaluations" => length(samples),
        "initial_pivot" => pivot, "seed" => seed,
        "wall_seconds" => (time_ns() - started) * 1e-9,
    )
end

function main()
    pkgversion(TCI) == v"0.9.19" || error("Require pinned TensorCrossInterpolation 0.9.19")
    BLAS.set_num_threads(1)
    request = JSON3.read(read(stdin, String))
    targets = [construct_target(s, Int(request.qubits), request.options,
        Int(request.options.seed) + i - 1) for (i, s) in enumerate(request.targets)]
    JSON3.write(stdout, Dict("schema" => 1, "julia_version" => string(VERSION),
        "library_version" => string(pkgversion(TCI)), "targets" => targets))
end

main()
