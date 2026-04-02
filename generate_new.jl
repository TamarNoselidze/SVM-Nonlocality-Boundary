using QuantumInformation
using LinearAlgebra
using HDF5

n = 5000
dim = 4 
# states = zeros(ComplexF64, dim, dim, n)

h = HilbertSchmidtStates(4)



states = [rand(h) for _ ∈ 1:n]

for i ∈ 1:n
    while ppt(states[i], [2, 2], 1) ≥ -1e-6
        states[i] = rand(h)
    end
end


states_3d = reshape(reduce(hcat, states), dim, dim, n)

h5open("entangled_states.h5", "w") do file
    write(file, "rho", states_3d)
end