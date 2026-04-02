using MosekTools
using HDF5               
using QuantumInformation 
using LinearAlgebra

include("CAPIBARA.jl/src/CAPIBARA.jl")
using .CAPIBARA

pol = polytope_spherical_covering(92)
s = shrinking_factor(pol)

file_in = h5open("entangled_states.h5", "r")
ρ_all = read(file_in, "rho")
close(file_in)

first_index = 1001
last_index = 1500

is_lhs = zeros(last_index - first_index + 1)

count = 1
for i ∈ first_index:last_index
    println("Checking if the state $i is steerable")
    
    ρ_i = ρ_all[:, :, i]
    v_lower = visibility_steering_cr(ρ_i, pol; solver = Mosek.Optimizer)
    
    if v_lower ≥ 1 - 1e-5
        is_lhs[count] = 1
    elseif v_lower / s ≤ 1 - 1e-5
        is_lhs[count] = -1
    end
    
    global count += 1
end

filename_out = "is_lhs_from_$(first_index)_to_$(last_index).h5"
h5open(filename_out, "w") do file
    write(file, "labels", is_lhs)
end

println("Finished! Labels saved to $filename_out")c