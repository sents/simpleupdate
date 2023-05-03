# [[file:../../notes.org::*Operator][Operator:1]]
module Operators
export AbstractOperator, Operator, exp, +, *, ⊗, nsite_op
# Operator:1 ends here

# [[file:../../notes.org::*Operator][Operator:2]]
using TensorOperations
using LinearAlgebra

abstract type AbstractOperator{T<:AbstractArray, N} end

struct Operator{T,N,A<:AbstractArray{T}} <: AbstractOperator{A, N}
    tensor :: A
    function Operator(tensor::A) where {T,N2,A<:AbstractArray{T,N2}}
        @assert iseven(N2) "The tensor dimension has to be even! In=Out"
        new{T,Int(N2/2),A}(tensor)
    end
end

Base.show(io::IO, S::Operator{T,N,A}) where {T,N,A} = print(io,
    "Operator{$T,$N,$A}: ", repr(size(S.tensor))
)

Base.ndims(::Type{<:AbstractOperator{<:Any, N}}) where N = N*2

function Base.exp(op::Operator{T,N}) where {T,N}
    s = size(op.tensor)
    return reshape(exp(reshape(op.tensor, (prod(s[1:N]), prod(s[N+1:2N])))), s)
end

function Base.:+(a::Operator{T,N}, b::Operator{T,N}) where {T,N}
    return Operator(a.tensor .+ b.tensor)
end

function Base.:*(a::Number, b::Operator)
    return Operator(a .* b.tensor)
end

if  Base.PkgId(Base.UUID("052768ef-5323-5732-b1bb-66c8b64840ba"),"CUDA") ∈ keys(Base.loaded_modules)
    import LinearAlgebra: Hermitian
    import CUDA: CuArray
    function Base.exp(A::Operator{T,N,Arr}) where {T,N, Arr<:CuArray}
        s = size(A.tensor)
        F = eigen(Hermitian(reshape(A.tensor, (prod(s[1:N]), prod(s[N+1:2N])))))
        retmat = (F.vectors * Diagonal(exp.(F.values))) * F.vectors'
        retmat -= Diagonal(imag(diag(retmat)) * im)
        return Operator(reshape(retmat, s))
    end
end

⊗(a::AbstractArray{T1,2}, b::AbstractArray{T2,2}) where {T1,T2} = @tensor c[i, j, k, l] := a[i, k] * b[j, l]

@generated function ⊗(xs::Vararg{AbstractArray, N}) where {N}
    dims = Int.(ndims.(xs) ./ 2)
    half_inds = sum(dims)
    indsleft = Vector{Int}[]
    indsright = Vector{Int}[]
    c = 1
    for dim in dims
        iis = -1 * (c:c+(dim-1))
        push!(indsleft, iis)
        push!(indsright, iis .- half_inds)
        c += dim
    end
    rightside = Expr(:call, :*,
    (:(xs[$i][$((indsleft[i]...,indsright[i]...)...)]) for i in 1:length(dims))...)
    return :(@tensor _[:] := $rightside)
end

⊗(os::Vararg{Operator, N}) where N = ⊗(getproperty.(os, :tensor))

function nsite_op(op::Operator{T}, inds, dims) where {T}
    N = length(dims)
    op_left = inds |> collect
    op_right = op_left .+ N
    I_left = deleteat!(1:N |> collect, op_left)
    I_right = I_left .+ N
    I_op = ⊗([UniformScaling{T}(one(T))(d) |> collect
              for (i, d) in enumerate(dims)
              if i ∉ inds]...)
    Operator(
        ncon((op.tensor, I_op),
            ([op_left; op_right] .* -1, [I_left; I_right] .* -1))
    )
end
# Operator:2 ends here

# [[file:../../notes.org::*Operator][Operator:3]]
end
# Operator:3 ends here
