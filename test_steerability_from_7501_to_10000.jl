using MosekTools
using Serialization

include("CAPIBARA.jl/src/CAPIBARA.jl")
using .CAPIBARA

pol = polytope_spherical_covering(92)
s = shrinking_factor(pol)

ρ = deserialize("random_ent_states.dat")

first_index = 7501
last_index = 10000

is_lhs = zeros(last_index - first_index + 1)

count = 1
for i ∈ first_index:last_index
    println("Checking if the state $i is steerable")
    v_lower = visibility_steering_cr(ρ[i], pol; solver = Mosek.Optimizer)
    if v_lower ≥ 1 - 1e-5
        is_lhs[count] = 1
    elseif v_lower / s ≤ 1 - 1e-5
        is_lhs[count] = -1
    end
    global count += 1
end

serialize("is_lhs_from_$(first_index)_to_$(last_index).dat", is_lhs)