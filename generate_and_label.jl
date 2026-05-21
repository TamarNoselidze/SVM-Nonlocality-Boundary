using MosekTools
using HDF5
using QuantumInformation
using LinearAlgebra


include("CAPIBARA.jl/src/CAPIBARA.jl")
using .CAPIBARA


const pol = polytope_spherical_covering(92)
const s = shrinking_factor(pol)

const pol_2 = polytope_spherical_covering(122)
const s_2 = shrinking_factor(pol_2)


function test_steerability(rho, i)
    println("Checking if the state $i is steerable")
    v_lower = visibility_steering_cr(rho, pol; solver = Mosek.Optimizer)

    if v_lower ≥ 1 - 1e-5
        return 1, v_lower
    elseif v_lower / s ≤ 1 - 1e-5 
        return -1, v_lower
    else
        println("  State $i is ambiguous. Testing with pol_2")
        
        v_lower_2 = visibility_steering_cr(rho, pol_2; solver = Mosek.Optimizer)
        
        if v_lower_2 ≥ 1 - 1e-5
            return 1, v_lower_2   # Resolved as LHS with pol_2
        elseif v_lower_2 / s_2 ≤ 1 - 1e-5
            return -1, v_lower_2  # Resolved as steerable with pol_2
        else
            println("  State $i remains ambiguous even with pol_2.")
            return 0, v_lower_2   # 0 indicates it is still unresolved
        end
    end
    return label
end


function generate_dataset(n_target)
    h = HilbertSchmidtStates(4)

    valid_states = Vector{Any}(undef, n_target)
    valid_labels = Vector{Int}(undef, n_target)

    count = 0
    total_generated = 0

    I_half = [1.0 0.0; 0.0 1.0] / 2.0

    while count < n_target
        # Generate a random state
        rho = rand(h)
        total_generated += 1
        
        # Check for entanglement
        if ppt(rho, [2, 2], 1) < -1e-6
        # if eigmin(partial_transpose(rho, 1, [2, 2])) < -1e-6
            
            # if entangled, test steerability
            label, eta = test_steerability(rho, total_generated)

            if label == 0
                println("the state is still ambiguous")
                # Calculate the extrapolation factor needed to hit the inner boundary
                eta_steerable = eta / s_2
                
                # Trace out the first subsystem (Alice)
                rho_B = ptrace(rho, [2, 2], 1)
                
                # Extrapolate away from local white noise
                rho_new = eta_steerable * rho + (1 - eta_steerable) * kron(I_half, rho_B)
                
                # Verify that extrapolating didn't break the physical validity of the state
                # We wrap it in Hermitian() to avoid small imaginary floating-point errors
                if eigmin(Hermitian(rho_new)) >= -1e-6
                    rho = rho_new
                    label = -1 # Successfully mapped to non-LHS boundary
                    println("successfully mapped to non lhs!")
                else
                    # State became unphysical, so we discard it and try another
                    println("couldnt map to non lhs, skipping")
                    continue
                end
            end

            if label == 1
                continue
            end
            
            # Check if the label is definitive
            if label == -1 # 1 for lhs, -1 for non-lhs
                count += 1
                
                valid_states[count] = rho
                valid_labels[count] = label
                
                # Print every 10% of completion
                if count % max(1, div(n_target, 10)) == 0
                    println("Found $count / $n_target valid states... (Total attempts: $total_generated)")
                end
            end
            # discard the label == 0 (ambiguous) states
        end
    end
    
    println("Success! Generated $n_target valid states out of $total_generated total attempts.")
    return valid_states, valid_labels
end



n = 10
states_list, labels_list = generate_dataset(n)

# Reshape and save using HDF5
states_3d = reshape(reduce(hcat, states_list), 4, 4, n)

h5open("states.h5", "w") do file
    write(file, "rho", states_3d)
end

h5open("labels.h5", "w") do file
    write(file, "labels", labels_list)
end