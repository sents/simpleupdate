module SimpleUpdate_CUDA
using SimpleUpdate.Util
using SimpleUpdate.Operators
import SimpleUpdate.Util: cached_similar_ordered_inds
using cuTENSOR, CUDA
using LinearAlgebra
import TensorOperations: similar_from_structure

Util.similar_atype(
    A::Type{<:CuArray},
    N=A.parameters[2],
    T=A.parameters[1])=Util.basetype(A){T,N,A.parameters[3]}


Util.eigf(T::Hermitian{D,A}) where {D, A<:CuArray} = eigen(T)


# Don't use caching for CUDA ops
function cached_similar_ordered_inds(T, A::CuArray, sym, sizes, ainds)
    structure = map(((tnum, tdim),)->sizes[tnum][tdim], ainds)
    similar_from_structure(A, T, structure)
end

function Base.exp(A::Operator{T,N,Arr}) where {T,N,Arr<:CuArray}
    s = size(A.tensor)
    F = eigen(Hermitian(reshape(A.tensor, (prod(s[1:N]), prod(s[N+1:2N])))))
    retmat = (F.vectors * Diagonal(exp.(F.values))) * F.vectors'
    retmat -= Diagonal(imag(diag(retmat)) * im)
    return Operator(reshape(retmat, s))
end

Util.n_array_type(A::Type{CuArray{T,N,D}}, ::Val{M}) where {T,N,D,M} = CuArray{T,M,D}

end
