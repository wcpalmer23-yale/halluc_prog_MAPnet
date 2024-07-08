# Custom distribution extending Gen.random
# By William Palmer
# Dirichlet
# hello 
struct Dirichlet <: Gen.Distribution{Vector{Int}} end
const dirichlet = Dirichlet()
function Gen.random(::Dirichlet, αs::AbstractVector{U}) where {U <: Real}
    d = Distributions.Dirichlet(αs)
    return Distributions.rand(d)
end
function Gen.logpdf(::Dirichlet, x::AbstractVector{T}, αs::AbstractVector{U}) where {T <: Real, U <: Real}
    d = Distributions.Dirichlet(αs)
    return Distributions.logpdf(d, x)
end
