# [[file:../../notes.org::*Util][Util:1]]
module Util
using ..OptimalContraction: ContractionCache, optimal_contraction_inds
export ncon_indices,
    moveind!, padded_inner_product,
    Logger, record!,
    tile_structurematrix,
    tile_structurematrix_with_origin,
    make_ordered_structurematrix
# Util:1 ends here

# [[file:../../notes.org::*Util][Util:2]]
using LinearAlgebra: norm
"""
    ncon_indices(sizes, contractions, open_inds; optimize=false)
Calculate the ncon style indices for a series of indice contractions and open indices

sizes: A Vector of tuples representing the sizes of the contracted tensors

contractions: A Vector of `((tensor_num_A, ind_A), (tensor_num_B, ind_B))`

open_inds: A Vector deciding the order of open indices. Written as e.g.

`[(tensor_num_A, (1,2,3)), (tensor_num_B, (1,2,3)), (tensor_num_A, (4,)), (tensor_num_B, (4,))]`
"""
function ncon_indices(sizes, contractions, open_inds, optimize=false)
    inds = _ncon_indices(sizes, contractions, open_inds)
    if optimize
        return optimal_contraction_inds(sizes, inds)
    else
        return inds
    end
end

function ncon_indices(sizes, contractions, open_inds, cache::ContractionCache)
    inds = _ncon_indices(sizes, contractions, open_inds)
    return optimal_contraction_inds(sizes, inds; cache=cache)
end

function _ncon_indices(sizes::Vector{<:Tuple}, contractions, open_inds)
    dims = length.(sizes)
    inds = [zeros(Int, dim) for dim in dims]
    closed = 0
    for ((A, Aind), (B, Bind)) in contractions
        inds[A][Aind] = inds[B][Bind] = closed += 1
    end
    open = 0
    for (A, oinds) in open_inds
        for i in oinds
            inds[A][i] = open -= 1
        end
    end
    return Tuple(inds)
end

function moveind!(a, from, to)
    val = popat!(a,from)
    insert!(a, to, val)
    a
end

struct Logger{LogStep}
    log :: Vector{LogStep}
    printit :: Int
end

Logger{LogStep}(printit=0) where LogStep = Logger{LogStep}(LogStep[], printit)

function record!(logger::Logger{LogStep}, step::LogStep) where LogStep
    if logger.printit > 0 && length(log) % logger.printit == 0
        println(step)
    end
    push!(logger.log, step)
end

record!(logger::Logger{LogStep},
    t::Tuple) where {LogStep<:NamedTuple} = record!(logger, LogStep(t))

pad(a::AbstractVector, n, f) = [a; fill(f, n-length(a))]

"""
    padded_inner_product(a, b)
Calculate the inner product of two vectors, that can be of different lengths.
If one vector is longer the other is padded with zeros for the calculation of
the inner product.
"""
function padded_inner_product(a::V, b::V) where {T, V<:AbstractVector{T}}
    max_length = max(length(a), length(b))
    norm(pad(a, max_length, zero(T)) .- pad(b, max_length, zero(T)))
end


function tile_structurematrix_with_origin(m, tile_pattern)
    N = maximum(tile_pattern) # number of distinct unit cells
    D = ndims(tile_pattern) # number of tile directions
    n = Int(size(m)[1]/(2^D)) # number of sites in unit cell
    nS = size(m)[2] # Number of Simplices connected to primitive unit cell
    parts = Matrix{Tuple{Int,Int}}[]
    for ci in CartesianIndices(tile_pattern)
        cells_inds = mod_ind.(cartesian_positive_adjacents(ci), Ref(size(tile_pattern)))
        tocells = tile_pattern[cells_inds]

        m_struct = fill((0,0), (N*n, nS))
        for (fromcell, tocell) in enumerate(tocells)
            for (row_to, row_from) in zip(((tocell-1)*n+1):tocell*n, ((fromcell-1)*n+1):fromcell*n)
                for col in 1:nS
                    celli = fld_ind(row_from, n)
                    if m[row_from, col] != 0
                        m_struct[row_to, col] = (m[row_from, col], celli)
                    end
                end
            end
        end
        push!(parts, m_struct)
    end
    reduce(hcat, parts)
end

function tile_structurematrix(m, tile_pattern)
    N = maximum(tile_pattern) # number of distinct unit cells
    D = ndims(tile_pattern) # number of tile directions
    n = Int(size(m)[1]/(2^D)) # number of sites in unit cell
    nS = size(m)[2] # Number of Simplices connected to primitive unit cell
    parts = Matrix{Int}[]
    for ci in CartesianIndices(tile_pattern)
        cells_inds = mod_ind.(cartesian_positive_adjacents(ci), Ref(size(tile_pattern)))
        tocells = tile_pattern[cells_inds]

        m_struct = zeros(Int, (N*n, nS))
        for (fromcell, tocell) in enumerate(tocells)
            m_struct[((tocell-1)*n+1):tocell*n, :] .+= m[((fromcell-1)*n+1):fromcell*n, :]
        end
        push!(parts, m_struct)
    end
    reduce(hcat, parts)
end

fld_ind(i, l) = fld((i - 1) , l) + 1
mod_ind(i, l) = (i - 1) % l + 1
mod_ind(c::CartesianIndex, s) = CartesianIndex(mod_ind.(c.I, s))

function cartesian_positive_adjacents(ci::CartesianIndex{D}) where D
    offsets = CartesianIndices(ntuple(_->D,Val(D))) .- Ref(one(CartesianIndex{D}))
    return offsets .+ Ref(ci)
end


function make_ordered_structurematrix(m0)
    m = copy(m0)
    D=Dict{Int,Int}()
    for (Si,Scol) in enumerate(eachcol(m))
        for si in findall(x->x!=0, Scol)
            n = get(D, si, 0) + 1
            m[si, Si] = D[si] = n
        end
    end
    m
end
# Util:2 ends here

# [[file:../../notes.org::*Util][Util:3]]
end
# Util:3 ends here
