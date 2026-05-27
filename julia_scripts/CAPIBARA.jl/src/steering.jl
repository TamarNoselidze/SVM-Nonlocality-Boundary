"""
    visibility_steering_cr(
        ρ::AbstractMatrix{T},
        polytope::AbstractVector{<:AbstractVector{T2}} = polytope_spherical_covering(92);
        return_model = false,
        upper = false,
        verbose = false,
        solver = Hypatia.Optimizer{Ket._solver_type(T)}) 

Computes a lower bound for the maximum `η` for which `η * ρ + (1 - η) Id ⊗ Trₐ(ρ)` admits a local hidden state model for all dichotomic measurements using the critical radius method.
If `upper = true`, computes an upper bound for such an `η`.
If `return_model = true`, returns a local hidden state model, based on the states of `polytope`.

Reference: Ngyuen, Nguyen, Gühne, [arXiv:1808.09349](https://arxiv.org/abs/1808.09349)
"""
function visibility_steering_cr(
    ρ::AbstractMatrix{T1},
    polytope::AbstractVector{<:AbstractVector{T2}} = polytope_spherical_covering(92);
    return_model = false,
    upper = false,
    verbose = false,
    solver = Hypatia.Optimizer{Ket._solver_type(promote_type(real(T1), T2))}
) where {T1<:Number,T2<:Real}
    @assert size(ρ) == (4, 4)
    n = length(polytope)
    if upper
        s = shrinking_factor(polytope)
        polytope = polytope ./ s
    end

    F = kron(I(2), sqrt(inv(partial_trace(ρ, 1, [2, 2]))))
    ρ = F * ρ * F
    ρ ./= tr(ρ)

    a = SVector{3}(bloch_vector(partial_trace(ρ, 2, [2, 2]))[2:end])
    C = @SMatrix [real(dot(ρ, pauli(T1, [i, j]))) for i ∈ 1:3, j ∈ 1:3]

    model = Model(solver)
    verbose || set_silent(model)

    @variable(model, r)
    @variable(model, probs[1:n] ≥ 0)

    normal = zeros(T2, 3)
    for i ∈ 1:n, j ∈ (i+1):n, k ∈ (j+1):n
        offset = plane!(normal, polytope[i], polytope[j], polytope[k])
        inv_denom = inv(norm(C * normal - offset * a))
        @constraint(model, sum(abs(dot(normal, polytope[l]) - offset) * inv_denom * probs[l] for l ∈ 1:n) ≥ r)
    end
    @constraint(model, sum(probs) == 1)
    for i ∈ 1:3
        @constraint(model, sum(probs[j] * polytope[j][i] for j ∈ 1:n) == 0)
    end

    @objective(model, Max, r)
    optimize!(model)
    is_solved_and_feasible(model) || @warn raw_status(model)
    if !upper && return_model
        return objective_value(model), value.(probs)
    else
        return objective_value(model)
    end
end
function visibility_steering_cr(
    ρ::AbstractMatrix{T1},
    polytope::AbstractVector{<:AbstractMatrix{T2}};
    return_model = false,
    upper = false,
    verbose = false,
    solver = Hypatia.Optimizer{Ket._solver_type(real(promote_type(T1, T2)))}
) where {T1<:Number,T2<:Number}
    @assert all(v -> size(v) == (2, 2), polytope)
    R = real(promote_type(T1, T2))
    vertices = Vector{SVector{3,R}}(undef, length(polytope))
    for i ∈ eachindex(polytope)
        b = bloch_vector(polytope[i])
        vertices[i] = SVector{3,R}(b[2], b[3], b[4])
    end
    return visibility_steering_cr(ρ, vertices; return_model, upper, verbose, solver)
end
export visibility_steering_cr

#Given three three-dimensional vectors, returns the plane that passes through them (in the form dot(normal, x) = offset)
function plane!(normal, p1::AbstractVector{<:Real}, p2::AbstractVector{<:Real}, p3::AbstractVector{<:Real})
    normal[1] = (p2[2]-p1[2])*(p3[3]-p1[3])-(p2[3]-p1[3])*(p3[2]-p1[2])
    normal[2] = (p2[3]-p1[3])*(p3[1]-p1[1])-(p2[1]-p1[1])*(p3[3]-p1[3])
    normal[3] = (p2[1]-p1[1])*(p3[2]-p1[2])-(p2[2]-p1[2])*(p3[1]-p1[1])
    return dot(p1, normal)
end
