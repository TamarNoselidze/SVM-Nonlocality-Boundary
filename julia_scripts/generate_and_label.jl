using MosekTools
using HDF5
using QuantumInformation
using LinearAlgebra

include("CAPIBARA.jl/src/CAPIBARA.jl")
using .CAPIBARA

# Read command line arguments
if length(ARGS) < 1 || !(ARGS[1] in ["LHS", "NON-LHS"])
    println("Usage: julia generate_and_label.jl [LHS|NON-LHS] [BatchID]")
    exit(1)
end

const mode = ARGS[1]
const batch_id = length(ARGS) >= 2 ? ARGS[2] : "1" # Default to 1 if not provided
println("Starting generation mode: $mode, Batch: $batch_id")


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
            return 1, v_lower_2   
        elseif v_lower_2 / s_2 ≤ 1 - 1e-5
            return -1, v_lower_2  
        else
            println("  State $i remains ambiguous even with pol_2.")
            return 0, v_lower_2   
        end
    end
end

function generate_dataset(target_count, target_label)
    h = HilbertSchmidtStates(4)
    states = Any[]
    
    count = 0
    total_generated = 0

    while count < target_count
        rho = rand(h)
        total_generated += 1
        
        # Force garbage collection every 50 states to clear old Mosek models
        if total_generated % 50 == 0
            GC.gc()
        end
        
        # Check for entanglement
        if ppt(rho, [2, 2], 1) < -1e-6
            
            label, eta = test_steerability(rho, total_generated)

            # Only save the state if it matches the label we are currently looking for
            if label == target_label
                count += 1
                push!(states, rho)
                
                if count % max(1, div(target_count, 10)) == 0
                    label_str = target_label == 1 ? "LHS" : "non-LHS"
                    println("Found $count / $target_count $label_str states...")
                end
            end
        end
    end
    
    label_str = target_label == 1 ? "LHS" : "non-LHS"
    println("Success! Generated $target_count $label_str states out of $total_generated total attempts.")
    
    valid_states = states
    valid_labels = fill(target_label, target_count)
    
    return valid_states, valid_labels
end

n_target = 4000
target_lbl = mode == "LHS" ? 1 : -1

states_list, labels_list = generate_dataset(n_target, target_lbl)

# Reshape and save using HDF5
states_3d = reshape(reduce(hcat, states_list), 4, 4, n_target)

# Dynamically name the output files based on the argument
file_suffix = mode == "LHS" ? "lhs" : "non_lhs"

h5open("states_$(file_suffix)_normal_$(batch_id).h5", "w") do file
    write(file, "rho", states_3d)
end

h5open("labels_$(file_suffix)_normal_$(batch_id).h5", "w") do file
    write(file, "labels", labels_list)
end