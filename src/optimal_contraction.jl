# [[file:../SimpleUpdate.org::*Optimal Contraction][Optimal Contraction:1]]
module OptimalContraction
export ContractionCache, optimal_contraction_inds, optimal_contraction, save!, ==
# Optimal Contraction:1 ends here

# [[file:../SimpleUpdate.org::*Optimal Contraction][Optimal Contraction:2]]
import JLD2: save_object, load_object
import TensorOperations: Poly, Power, optimaltree
import .Iterators: flatten

function optimal_contraction_inds(sizes, indices; cache=nothing)
    order, costs = optimal_contraction(sizes, indices; cache=cache)
    return ordered_contraction_indices(indices, order)
end

function optimal_contraction(sizes, indices; cache=nothing) #:: OptimalContractionInfo
    contraction_signature = ContractionSignature(collect.(sizes), indices)
    return optimal_contraction(contraction_signature, cache)
end


function optimal_contraction_inds(
    tensors::Vector{T},
    indices;
    cache=nothing,
) where {T<:AbstractArray}
    optimal_contraction_inds(size.(tensors), indices; cache=cache)
end

function optimal_contraction(
    tensors::Vector{T},
    indices;
    cache=nothing,
) where {T<:AbstractArray}
    optimal_contraction(size.(tensors), indices; cache=cache)
end

struct ContractionSignature{T,N}
    tensorsizes::NTuple{N,NTuple{M,T} where M}
    indices::NTuple{N,NTuple{M,Int} where M}
end
ContractionSignature(tensorsizes, indices) =
    ContractionSignature(Tuple(Tuple.(tensorsizes)), Tuple(Tuple.(indices)))
function ContractionSignature(tensorsizes::Vector{Vector{Int}}, indices)
    ContractionSignature(convert.(Vector{BigInt}, tensorsizes), indices)
end

# For dicts to properly work we need a hash
Base.hash(c::ContractionSignature, h::UInt) = hash(c.tensorsizes,
                                                hash(c.indices,
                                                    hash(ContractionSignature,h)))

Base.:(==)(c1::ContractionSignature, c2::ContractionSignature) =
    c1.indices == c2.indices && c1.tensorsizes == c2.tensorsizes

const OptimalContractionInfo = @NamedTuple begin
    order::Vector{Any}
    cost::Union{BigInt,Poly{:χ}}
end

struct ContractionCache
    table::Dict{ContractionSignature,OptimalContractionInfo}
    filename::Union{Nothing,String}
    autosave::Bool
end

function ContractionCache(; filename=nothing, autosave=false)
    @assert !autosave || !isnothing(filename)
    ContractionCache(
        Dict{ContractionSignature,OptimalContractionInfo}(),
        filename,
        autosave,
    )
end

function optimal_contraction(
    contraction_signature::ContractionSignature,
    cache::Union{Nothing,ContractionCache},
)
    if !isnothing(cache)
        return cached_optimal_contraction(contraction_signature, cache)
    else
        return optimal_contraction(contraction_signature)
    end
end

without(vs, ns) = [v for (i, v) in enumerate(vs) if i ∉ ns]

function make_ind_isless(start_inds)
    function lt_inds(
        ((ft1, fi1, s1), tot1, toi1),
        ((ft2, fi2, s2), tot2, toi2),
        visited=Int[],
    )
        inds = without(start_inds, visited)
        s1 < s2 && return true
        next1 = get(inds, tot1, (0, (0, 0)))
        next2 = get(inds, tot2, (0, (0, 0)))
        (next2 == 0) && return false
        (next1 == 0) && (next2 != 0) && return true

        (o1, on1) = only([(n[2][1], i) for (i, n) in enumerate(next1)])
        (o2, on2) = only([(n[2][1], i) for (i, n) in enumerate(next2)])

        (o1, next1[2][1]) == (o2, next2[2][1]) && return false # equal
        is1 = sort(without(next1, (on1,)); lt=(a, b) -> lt_inds(a, b, [o1, o2]))
        is2 = sort(without(next2, (on2,)); lt=(a, b) -> lt_inds(a, b, [o1, o2]))



    end
end

function normalize_contraction_signature(sizes, inds)
    cont = zip.(sizes, ninds_conts(inds)) .|> collect
end

ninds_conts(ninds) = [
    [
        [
            (k, r) for
            (k, r) in enumerate(get(findall(oinds .== ind), 1, 0) for oinds in ninds) if
            r != 0 && k != i
        ] |> x -> get(x, 1, (0, 0)) for ind in inds
    ]::Vector{Tuple{Int,Int}} for (i, inds) in enumerate(ninds)
]

function cached_optimal_contraction(contraction_signature, cache::ContractionCache)
    if haskey(cache.table, contraction_signature)
        return cache.table[contraction_signature]
    end
    optimal_contraction_info = optimal_contraction(contraction_signature)
    register!(cache, contraction_signature, optimal_contraction_info)
    return optimal_contraction_info
end

function register!(cache::ContractionCache, contraction_signature, optimal_contraction_info)
    cache.table[contraction_signature] = optimal_contraction_info
    cache.autosave && save!(cache)
end

function save!(cache::ContractionCache)
    @assert !isnothing(cache.filename) "ContractionCache needs a filename to save!"
    table = Dict(
        (collect(collect.(k.tensorsizes)), collect(collect.(k.indices))) => v for
        (k, v) in cache.table
    )
    save_object(cache.filename, (table, cache.filename, cache.autosave))
end

function ContractionCache(filename::String)
    (vtable, filename, autosave) = load_object(filename)
    table = Dict(
        ContractionSignature(Tuple(Tuple.(k[1])), Tuple(Tuple.(k[2]))) => v for
        (k, v) in vtable
    )
    ContractionCache(table, filename, autosave)
end

function optimal_contraction(contraction_signature::ContractionSignature{T}) where {T}
    tensorsizes = contraction_signature.tensorsizes
    indices = contraction_signature.indices

    symbol_dict = Dict{Int,Symbol}()
    cont_network = Vector{Symbol}[]
    cost_dict = Dict{Symbol,T}()
    for (tcosts, tinds) in zip(tensorsizes, indices)
        cont = Symbol[]
        for (cost, ind) in zip(tcosts, tinds)
            if haskey(symbol_dict, ind)
                sym = symbol_dict[ind]
            else
                sym = gensym()
                symbol_dict[ind] = sym
                cost_dict[sym] = cost
            end
            push!(cont, sym)
        end
        push!(cont_network, cont)
    end
    order, cost = optimaltree(cont_network, cost_dict)
    return OptimalContractionInfo((order, cost))
end


ordered_contraction_indices(inds, order) =
    ordered_contraction_indices(inds, contraction_indices_order(order, inds))

function ordered_contraction_indices(inds, ind_map::Dict{Int,Int})
    minds = [[i for i in ind] for ind in inds]
    for (i, tensor) in enumerate(inds)
        for (j, ind) in enumerate(tensor)
            if haskey(ind_map, ind)
                minds[i][j] = ind_map[ind]
            end
        end
    end
    return Tuple(Tuple.(minds))
end

function contraction_indices_order(contraction_tree, indices)
    c, ind_map, involved = _contraction_indices_order(contraction_tree, indices)
    return ind_map
end

function _contraction_indices_order(nodes, network, c=0, ind_map=Dict{Int,Int}())
    a, b = nodes
    c, ind_map, involved_a = _contraction_indices_order(a, network, c, ind_map)
    c, ind_map, involved_b = _contraction_indices_order(b, network, c, ind_map)
    for i in intersect(
        flatten([network[i] for i in involved_a]),
        flatten([network[i] for i in involved_b]),
    )
        ind_map[i] = c += 1
    end
    return c, ind_map, [involved_a; involved_b]
end

_contraction_indices_order(node::Int, network, c, ind_map) = c, ind_map, node

function testf(D, p)
    a = rand(D, D, D, D, p)
    b = rand(D, D, D, D, p)
    c = rand(p, p, p, p)
    op = rand(p, p, p, p)
    cache = ContractionCache()
    inds = [[1, -1, -2, 2], [1, -3, -4, 3], [2, -5, -6, 4], [3, -7, -8, 4]]
    order, costs = optimal_contraction(size.([a, b, c, op]), inds, cache=cache)
    return order, inds
end
# order, inds = testf(10,2)
# Optimal Contraction:2 ends here

# [[file:../SimpleUpdate.org::*Optimal Contraction][Optimal Contraction:3]]
end
# Optimal Contraction:3 ends here
