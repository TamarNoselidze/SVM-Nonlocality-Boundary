using MosekTools
using Serialization

include("CAPIBARA.jl/src/CAPIBARA.jl")
using .CAPIBARA

pol = polytope_spherical_covering(92)
s = shrinking_factor(pol)

ρ = deserialize("random_ent_states.dat")

first_index = 2
last_index = 5

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

#The lines below are more or less what we discussed in the meeting. I've used the script above instead.
#Checks if it is entangled
#rho_is_ent = eigmin(Ket.partial_transpose(rho, 1, [2,2])) < 0 #if rho_is_ent == false, rho is separable (and, therefore, is also lhs)

#Checks if rho is inside LHS
#rho_is_lhs = CAPIBARA.visibility_steering_cr(rho) > 1 #if rho_is_lhs == false, we don't know anything

#Maybe think about visibility later, i.e.:
#η = CAPIBARA.visibility_steering_cr(rho)
#rho_lhs = η * rho + (1 - η) * kron(LinearAlgebra.I(2), Ket.partial_trace(rho, 1, [2,2]))

#Check if rho is outside LHS
#rho_is_not_lhs = CAPIBARA.visibility_steering_cr(rho; upper = true) < 1 #if rho_is_not_lhs == false, we don't know anything