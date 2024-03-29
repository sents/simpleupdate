# [[file:../SimpleUpdate.org::*Util][Util:1]]
module Util
using ..OptimalContraction: ContractionCache, optimal_contraction_inds
using LinearAlgebra
export ncon_indices,
    moveind!,
    padded_inner_product,
    Logger,
    record!,
    tile_structurematrix,
    tile_structurematrix_with_origin,
    make_ordered_structurematrix,
    connection_matrix_from_connections,
    n_array_type,
    similar_atype,
    eigf,
    @ctensor
# Util:1 ends here

# [[file:../SimpleUpdate.org::*Util][Util:2]]
using LinearAlgebra: norm, eigen!
import TensorOperations: use_cache, cache, similar_from_structure, taskid
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
    val = popat!(a, from)
    insert!(a, to, val)
    a
end

struct Logger{LogStep}
    logf::Any
    log::Vector{LogStep}
end

Logger{LogStep}(f) where LogStep = Logger{LogStep}(f, LogStep[])

Logger{LogStep}(; printit::Int=0) where {LogStep} =
    Logger{LogStep}(LogStep[]) do logv, step
        if length(logv) % printit == 0
            println(step)
        end
    end

function record!(logger::Logger{LogStep}, step::LogStep) where {LogStep}
    push!(logger.log, step)
    logger.logf(logger.log, step)
end

record!(logger::Logger{LogStep}, t::Tuple) where {LogStep<:NamedTuple} =
    record!(logger, LogStep(t))

pad(a::AbstractVector, n, f) = [a; fill(f, n - length(a))]

"""
    padded_inner_product(a, b)
Calculate the inner product of two vectors, that can be of different lengths.
If one vector is longer the other is padded with zeros for the calculation of
the inner product.
"""
function padded_inner_product(a::V, b::V) where {T,V<:AbstractVector{T}}
    max_length = max(length(a), length(b))
    norm(pad(a, max_length, zero(T)) .- pad(b, max_length, zero(T)))
end


function tile_structurematrix_with_origin(m, tile_pattern)
    N = maximum(tile_pattern) # number of distinct unit cells
    D = ndims(tile_pattern) # number of tile directions
    n = Int(size(m)[1] / (2^D)) # number of sites in unit cell
    nS = size(m)[2] # Number of Simplices connected to primitive unit cell
    parts = Matrix{Tuple{Int,CartesianIndex{D}}}[]
    cinds = CartesianIndices(Tuple(2 for _ = 1:D))
    for ci in CartesianIndices(tile_pattern)
        cells_inds = mod_ind.(cartesian_positive_adjacents(ci), Ref(size(tile_pattern)))
        tocells = tile_pattern[cells_inds]

        m_struct = fill((0, zero(CartesianIndex{D})), (N * n, nS))
        for (fromcell, tocell) in enumerate(tocells)
            for (row_to, row_from) in
                zip(((tocell-1)*n+1):tocell*n, ((fromcell-1)*n+1):fromcell*n)
                for col = 1:nS
                    if m[row_from, col] != 0
                        m_struct[row_to, col] = (m[row_from, col], cinds[fromcell])
                    end
                end
            end
        end
        push!(parts, m_struct)
    end
    reduce(hcat, unique(parts))
end

function tile_structurematrix(m, tile_pattern)
    N = maximum(tile_pattern) # number of distinct unit cells
    D = ndims(tile_pattern) # number of tile directions
    n = Int(size(m)[1] / (2^D)) # number of sites in unit cell
    nS = size(m)[2] # Number of Simplices connected to primitive unit cell
    parts = Matrix{Int}[]
    for ci in CartesianIndices(tile_pattern)
        cells_inds = mod_ind.(cartesian_positive_adjacents(ci), Ref(size(tile_pattern)))
        tocells = tile_pattern[cells_inds]

        m_struct = zeros(Int, (N * n, nS))
        for (fromcell, tocell) in enumerate(tocells)
            m_struct[((tocell-1)*n+1):tocell*n, :] .+= m[((fromcell-1)*n+1):fromcell*n, :]
        end
        push!(parts, m_struct)
    end
    reduce(hcat, unique(parts))
end

fld_ind(i, l) = fld((i - 1), l) + 1
mod_ind(i, l) = (i - 1) % l + 1
mod_ind(c::CartesianIndex, s) = CartesianIndex(mod_ind.(c.I, s))

function cartesian_positive_adjacents(ci::CartesianIndex{D}) where {D}
    offsets = CartesianIndices(ntuple(_ -> D, Val(D))) .- Ref(one(CartesianIndex{D}))
    return offsets .+ Ref(ci)
end


function make_ordered_structurematrix(m0)
    m = copy(m0)
    D = Dict{Int,Int}()
    for (Si, Scol) in enumerate(eachcol(m))
        for si in findall(x -> x != 0, Scol)
            n = get(D, si, 0) + 1
            m[si, Si] = D[si] = n
        end
    end
    m
end

function connection_matrix_from_connections(
    connections, # Tuples of ((n1,cell1,index1), (n2,cell2,index2))
    n_cells=maximum([getindex.(c, 2) for c in connections] |> Iterators.flatten),
)
    n_sites = maximum([getindex.(c, 1) for c in connections] |> Iterators.flatten)
    m = zeros(Int, (n_sites * n_cells, length(connections)))
    for (i, sites) in enumerate(connections)
        for site in sites
            m[(site[2]-1)*n_sites+site[1], i] = site[3]
        end
    end
    m
end


# See https://github.com/JuliaLang/julia/issues/35543
"Strip type parameters of type. Works only for concrete types, not for UnionAll"
basetype(T::Type) = T.name.wrapper

"Strip dimension Parameter from AbstractArray"
n_array_type(A::Type{<:AbstractArray{T,N}}, ::Val{M}) where {T,N,M} = basetype(A){T,M}

similar_atype(A,N=A.parameters[2],T=A.parameters[1])=basetype(A){T,N}

eigf(T) = eigen!(T)


function cached_similar_ordered_inds(A, T, sym, sizes, o_from_i_inds)
    structure = map(((tnum, tdim),)->sizes[tnum][tdim], o_from_i_inds)
    if use_cache()
        type = Core.Compiler.return_type(
            similar,
            Tuple{typeof(A), Type{T}, typeof(structure)}
        )
        key = (sym, taskid(), type, structure)
        C::type = get!(cache, key) do
            similar_from_structure(A, T, structure)
        end
    else
        C = similar_from_structure(A, T, structure)
    end
    return C
end

# Takes tuple of ncon indices, returns tuple of tuples (intensor, intensor_ind) in order
# of dimensions in the resulting tensor
ordered_inds(tensors_inds) = getindex.(
    sort(reduce(vcat,
        [
        [(ncon_ind,tensor_num,tensor_ind) for (tensor_ind, ncon_ind)
         in enumerate(tensor_inds) if ncon_ind<0]
        for (tensor_num, tensor_inds) in enumerate(tensors_inds)]
    );
        by=i -> -i[1]),
    Ref(2:3)
) |> Tuple


function _ex_out(ex)
    if (ex.head == :call && ex.args[1] == :*)
        return map(_ex_out, ex.args[2:end])
    elseif (ex.head == :ref)
        return (ex.args[1], ex.args[2:end])
    end
end

macro ctensor(ex)
    ex.head != :(:=) && return :(error("Called cten without := assignment"))
    ex.args[1].args[2] != :(:) && return :(error("For now only ncon indices in the input tensors are supported!"))
    name_out = ex.args[1].args[1]
    inputs = _ex_out(ex.args[2])
    in_inds = Tuple(Tuple.(getindex.(inputs,2)))
    onames = Tuple(getindex.(inputs,1))
    oinds = Util.ordered_inds(in_inds)
    ex_size = :(size.(($((o for o in onames)...),)))
    ex_etype = :(promote_type(eltype.(( $((o for o in onames)...),))...))
    sym = gensym()
    eex = Expr(:(=), ex.args...)
    quote
        $name_out = Util.cached_similar_ordered_inds($(onames[1]),
            $ex_etype,
            $(QuoteNode(sym)),
            $ex_size,
            $oinds)
        @tensor $eex
    end |> esc
end
# Util:2 ends here

# [[file:../SimpleUpdate.org::*Util][Util:3]]
end
# Util:3 ends here
