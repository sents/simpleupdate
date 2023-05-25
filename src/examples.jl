# [[file:../../notes.org::*Test Ising simple update][Test Ising simple update:1]]
module Examples
using ..OptimalContraction
using ..Operators
using ..Util
using ..gPEPS
using ..ConnectionMatrices
# Test Ising simple update:1 ends here

# [[file:../../notes.org::*Test Ising simple update][Test Ising simple update:2]]
import LinearAlgebra: I

σ_z = [1 0; 0 -1.0]

σ_x = [
    0 1
    1 0.0
]
σ_y = [
    0 -im
    im 0
]
σ_z = [
    1 0
    0 -1.0
]

s_x = σ_x / 2
s_y = σ_y / 2
s_z = σ_z / 2


heisenberg_2site(J) = -J * (s_x ⊗ s_x + s_y ⊗ s_y + s_z ⊗ s_z)
spin_1site(μ) = -μ * s_z

ising_2site(J) = -J * (σ_z / 2 ⊗ σ_z / 2)

ising_ops(u::UnitCell, J, μ) = [
    Site2Operator(ising_2site(J)) + Site2Operator(site1_op) for
    site1_op in normalized_1site_ops(spin_1site(μ), u)
]

M_squareab = [
    1 2 3 4
    3 4 1 2
]

function test_ising(J=1.0, μ=0.0; cuda=false)
    tocuda = identity
    if cuda
        initf = CUDA.rand
        tocuda = CuArray
    else
        initf = Base.rand
    end

    u = unitcell_from_structurematrix(M_squareab, [2, 3, 4, 5])

    op2 = ising_2site(J)
    ops1 = normalized_1site_ops(spin_1site(μ), u)
    ops = [Site2Operator(op .+ op2 |> tocuda) for op in ops1]

    info = simple_update_information(u, 2)
    A = u.sites[1]
    B = u.sites[2]
    logger = Logger([], 50)
    simple_update(
        u,
        ops;
        τ₀=1.0,
        max_bond_rank=6,
        min_τ=1e-5,
        convergence=1e-8,
        sv_cutoff=1e-8,
        maxit=50000,
        logger=logger,
    )
    E = sum([calc_2site_ev(u, op, i) for (i, op) in enumerate(ops)]) / 2
    return logger, u, ops, E


    # simple_update_step!(u.sites[1],
    #     u.sites[2],
    #     eops[1],
    #     b1, S, info, 10,
    #     1e-10)
end


# R = test_ising(-1.0)
# Test Ising simple update:2 ends here

# [[file:../../notes.org::*Test Heisenberg with simple update][Test Heisenberg with simple update:1]]
function test_heisenberg(J=1.0, μ=0.0)
    u = unitcell_from_structurematrix(M_squareab, [2, 3, 4, 5], rand)

    op2 = heisenberg_2site(J)
    ops1 = normalized_1site_ops(spin_1site(μ), u)
    ops = [Site2Operator(op .+ op2) for op in ops1]

    info = simple_update_information(u, 2)
    A = u.sites[1]
    B = u.sites[2]
    logger = Logger([], 50)
    simple_update(
        u,
        ops;
        τ₀=1.0,
        max_bond_rank=20,
        min_τ=1e-5,
        convergence=1e-7,
        sv_cutoff=1e-8,
        maxit=50000,
        logger=logger,
    )
    E = sum([calc_2site_ev(u, op, i) for (i, op) in enumerate(ops)]) / 2
    return logger, u, ops, E


    # simple_update_step!(u.sites[1],
    #     u.sites[2],
    #     eops[1],
    #     b1, S, info, 10,
    #     1e-10)
end
# Test Heisenberg with simple update:1 ends here

# [[file:../../notes.org::*Try simple update on Floret Pentagon][Try simple update on Floret Pentagon:1]]
M_floretpent_6petal = [
    1 2 3 0 0 0 0 0 0 0 0 0 0 0 0
    3 0 0 1 2 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 3 1 2 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 1 2 3 0 0 0 0 0 0
    0 0 0 0 0 0 0 3 0 1 2 0 0 0 0
    0 0 2 0 0 0 0 0 0 0 1 3 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 1 2 3 0
    0 0 0 0 0 3 0 0 0 0 0 0 1 0 2
    0 1 0 4 0 0 0 0 2 5 0 0 0 3 6
]

function heisenberg_simple_update(M, J=-1.0, μ=0.0; bonddims=fill(5, size(M)[2]))
    u = unitcell_from_structurematrix(M, bonddims, fill(2, size(M)[1]), rand)
    op2 = heisenberg_2site(J)

    ops1 = normalized_1site_ops(spin_1site(μ), u)
    ops = [Site2Operator(op .+ op2) for op in ops1]

    logger = Logger([], 1)
    simple_update(
        u,
        ops;
        τ₀=1.0,
        max_bond_rank=20,
        min_τ=1e-5,
        convergence=1e-6,
        sv_cutoff=1e-8,
        maxit=50000,
        logger=logger,
    )
    E = sum([calc_2site_ev(u, op, i) for (i, op) in enumerate(ops)]) / 2
    return logger, u, ops, E
end
# Try simple update on Floret Pentagon:1 ends here

# [[file:../../notes.org::*Try simple update on Floret Pentagon][Try simple update on Floret Pentagon:2]]
end
# Try simple update on Floret Pentagon:2 ends here
