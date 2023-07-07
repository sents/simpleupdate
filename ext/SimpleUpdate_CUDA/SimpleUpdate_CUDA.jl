module SimpleUpdate_CUDA
using SimpleUpdate.Util
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

end
