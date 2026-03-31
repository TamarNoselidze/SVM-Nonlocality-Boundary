"""
    shrinking_factor(vertices::AbstractVector{<:AbstractVector{T}})

Computes the radius of the largest sphere contained in the polytope formed by `vertices`.
"""
function shrinking_factor(vertices::AbstractVector{<:AbstractVector{<:Real}})
    return Polyhedra.maximum_radius_with_center(
        Polyhedra.doubledescription(Polyhedra.vrep(vertices)),
        zero(vertices[1])
    )
end
shrinking_factor(all_ρ::AbstractVector{<:AbstractMatrix}) = shrinking_factor([bloch_vector(ρ)[2:end] for ρ ∈ all_ρ])
shrinking_factor(all_Aax::AbstractVector{<:Measurement}) = shrinking_factor(reduce(vcat, all_Aax))
export shrinking_factor
