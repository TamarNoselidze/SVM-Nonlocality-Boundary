using MosekTools
using HDF5
using QuantumInformation
using LinearAlgebra

include("CAPIBARA.jl/src/CAPIBARA.jl")
using .CAPIBARA

const pol = polytope_spherical_covering(92)

function generate_boundary_test_set(n_target)
    h = HilbertSchmidtStates(4)
    valid_states = Vector{Any}(undef, n_target)
    valid_labels = Vector{Int}(undef, n_target)
    
    count = 0
    total_generated = 0
    I_half = [1.0 0.0; 0.0 1.0] / 2.0

    while count < n_target
        rho = rand(h)
        total_generated += 1
        
        # entangled states
        if ppt(rho, [2, 2], 1) < -1e-6
            v_upper = visibility_steering_cr(ρ[i], pol_2; upper=true, solver = Mosek.Optimizer)
            
            # If v_lower < 1, the state is strictly outside the LHS polytope.
            if v_upper < 1.0 - 1e-5 
                v_lower = visibility_steering_cr(rho, pol; solver = Mosek.Optimizer)
                # bobs subsystem
                rho_B = ptrace(rho, [2, 2], 1)
                
                # Push the state exactly onto the LHS boundary
                rho_boundary_lhs = v_lower * rho + (1 - v_lower) * kron(I_half, rho_B)
                rho_boundary_nlhs = v_upper * rho + (1 - v_upper) * kron(I_half, rho_B)
                
                count += 1
                valid_states[count] = rho_boundary
                valid_labels[count] = 1 # 1 because it is now strictly an LHS state
                
                if count % 100 == 0
                    println("Generated $count / $n_target boundary states... (Attempts: $total_generated)")
                end
            end
        end
    end
    
    println("Done! Generated 1000 boundary states.")
    return valid_states, valid_labels
end

n = 100
states_list, labels_list = generate_boundary_test_set(n)










#need to reshape
states_3d = reshape(reduce(hcat, states_list), 4, 4, n)

h5open("stress_states.h5", "w") do file
    write(file, "rho", states_3d)
end

h5open("stress_labels.h5", "w") do file
    write(file, "labels", labels_list)
end