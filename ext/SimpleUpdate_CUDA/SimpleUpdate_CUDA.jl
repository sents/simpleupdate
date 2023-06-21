module SimpleUpdate_CUDA
using SimpleUpdate.Util
using cuTENSOR, CUDA
using LinearAlgebra

Util.similar_atype(
    A::Type{<:CuArray},
    N=A.parameters[2],
    T=A.parameters[1])=Util.basetype(A){T,N,A.parameters[3]}


Util.eigf(T::Hermitian{D,A}) where {D, A<:CuArray} = eigen(T)
end
