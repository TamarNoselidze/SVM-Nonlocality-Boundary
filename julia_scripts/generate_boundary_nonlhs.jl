using MosekTools
using HDF5
using QuantumInformation
using LinearAlgebra

include("CAPIBARA.jl/src/CAPIBARA.jl")
using .CAPIBARA


# Read command line arguments
if length(ARGS) < 1
    println("Usage: julia generate_and_label.jl [BatchID]")
    exit(1)
end

const batch_id = ARGS[1]

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


        if total_generated % 50 == 0
            GC.gc()
        end

    
        # entangled states
        if ppt(rho, [2, 2], 1) < -1e-6
            v_upper = visibility_steering_cr(rho, pol; upper=true, solver = Mosek.Optimizer)

            if v_upper < 1-1e-5   #already non-LHS, and definitely not ambiguous

                # eta = v_upper

                u = rand()
                k = 3
                eta = v_upper + (1 - v_upper) * u^k  # skewed toward boundary

                rho_B = ptrace(rho, [2, 2], 1)
                rho_non_lhs = eta * rho + (1 - eta) * kron(I_half, rho_B)

                count += 1
                valid_states[count] = rho_non_lhs
                valid_labels[count] = -1 
                
                if count % 10 == 0
                    println("Generated $count / $n_target boundary states... (Attempts: $total_generated)")
                end
            end
        end
    end
    
    println("Done! Generated $n_target boundary states.")
    return valid_states, valid_labels
end

n = 200
states_list, labels_list = generate_boundary_test_set(n)



#need to reshape
states_3d = reshape(reduce(hcat, states_list), 4, 4, n)

h5open("states_non_lhs_boundary_$(batch_id).h5", "w") do file
    write(file, "rho", states_3d)
end

h5open("labels_non_lhs_boundary_$(batch_id).h5", "w") do file
    write(file, "labels", labels_list)
end