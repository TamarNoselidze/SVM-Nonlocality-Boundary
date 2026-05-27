"""
    polytope_icosahedron([T = Float64])

Produces a vector of three dimensional vectors forming an icosahedron.
"""
function polytope_icosahedron(::Type{T}) where {T<:Number}
    ϕ = (1 + sqrt(T(5))) / 2
    scale = inv(sqrt(T(2) + ϕ))
    coords = ((0, 1, ϕ), (0, 1, -ϕ), (1, ϕ, 0), (1, -ϕ, 0), (ϕ, 0, 1), (ϕ, 0, -1))

    verts = Vector{SVector{3,T}}(undef, 12)
    for i ∈ 1:6
        v = scale * SVector{3,T}(coords[i])
        verts[i] = v
        verts[i+6] = -v
    end
    return verts
end
polytope_icosahedron() = polytope_icosahedron(Float64)
export polytope_icosahedron

"""
    polytope_fibonacci([T = Float64,] m::Integer)

Produces `m` three dimensional vectors distributed in the sphere according to the Fibonacci lattice.

Reference: [Fibonacci lattices](https://observablehq.com/@meetamit/fibonacci-lattices)
"""
function polytope_fibonacci(::Type{T}, m::Integer) where {T<:Number}
    vertices = Vector{SVector{3,T}}(undef, m)
    ϕ = pi * (sqrt(T(5)) - 1)
    for i ∈ 1:m
        z = 1 - 2 * T(i - 1) / (m - 1)
        radius = sqrt(1 - z^2)
        θ = ϕ * (i - 1)
        s, c = sincos(θ)
        vertices[i] = SVector{3,T}(radius * c, radius * s, z)
    end
    return vertices
end
polytope_fibonacci(m::Integer) = polytope_fibonacci(Float64, m)
export polytope_fibonacci

"""
    polytope_half_fibonacci([T = Float64,] m::Integer)

Produces `m` three-dimensional vectors distributed in the upper hemisphere of the unit sphere, with a distribution inspired by Fibonacci lattices.

References:
- [Fibonacci lattices](https://observablehq.com/@meetamit/fibonacci-lattices)
- Porto, Designolle, Pokutta, Quintino, [arXiv:2506.03045](https://arxiv.org/abs/2506.03045)
"""
function polytope_half_fibonacci(::Type{T}, m::Integer) where {T<:Number}
    vertices = Vector{SVector{3,T}}(undef, m)
    ϕ = pi * (sqrt(T(5)) - 1)
    for i ∈ 1:m
        z = 1 - T(i - 1) / (m - 1)
        radius = sqrt(1 - z^2)
        θ = ϕ * (i - 1)
        s, c = sincos(θ)
        vertices[i] = SVector{3,T}(radius * c, radius * s, z)
    end
    return vertices
end
polytope_half_fibonacci(m::Integer) = polytope_half_fibonacci(Float64, m)
export polytope_half_fibonacci

"""
    polytope_spherical_covering(m::Integer)

Produces `m` three-dimensional vectors distributed according to the best spherical covering found by [Sloane](http://neilsloane.com/icosahedral.codes/).
The input `m` must be in [`72`, `92`, `122`, `132`, `162`, `432`, `612`, `1032`, `5532`, `8192`].
"""
function polytope_spherical_covering(m::Integer)
    @assert m ∈ [72, 92, 122, 132, 162, 192, 212, 272, 432, 612, 1032, 5532, 8192]
    return deserialize(
        joinpath(@__DIR__, "..", "data", "polytopes", "spherical_covering$(m).dat")
    )::Vector{SVector{3,Float64}}
end
export polytope_spherical_covering
