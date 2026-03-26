using Ket
using Serialization

n = 10000
ρ = [random_state(4) for _ ∈ 1:n]

for i ∈ 1:n
    while eigmin(partial_transpose(ρ[i], 1, [2,2])) ≥ 1e-6
        ρ[i] = random_state(4)
    end
end

serialize("random_ent_states.dat", ρ)