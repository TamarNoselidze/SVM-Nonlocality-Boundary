using MosekTools
using HDF5
using QuantumInformation
using LinearAlgebra

include("CAPIBARA.jl/src/CAPIBARA.jl")
using .CAPIBARA

const pol_2 = polytope_spherical_covering(122)

function generate_non_lhs_boundary(n_target; epsilon=1e-3)
    valid_states = Vector{Any}(undef, n_target)
    valid_labels = Vector{Int}(undef, n_target)
    
    count = 0
    total_generated = 0
    
    # Pre-allocate local white noise component
    I_half = [1.0 0.0; 0.0 1.0] / 2.0

    println("Starting generation of $n_target Non-LHS boundary states...")

    while count < n_target
        total_generated += 1
        
        # 1. Generate a random PURE state (guaranteed to be far outside LHS)
        # We generate a random complex vector of size 4 and normalize it
        psi = randn(ComplexF64, 4)
        psi = psi / norm(psi)
        
        # Convert the pure state vector into a density matrix
        rho_pure = psi * psi'
        
        # 2. Check steerability to find exactly where the boundary is
        v_lower = visibility_steering_cr(rho_pure, pol_2; solver = Mosek.Optimizer)

        # If v_lower < 1, the pure state is outside the LHS set (which it almost always will be)
        if v_lower < 1.0 - 1e-5
            
            # 3. Add the tiny epsilon to stop just short of the boundary
            # We use min() to ensure we never accidentally go above 1.0 (which would be unphysical)
            eta_epsilon = min(v_lower + epsilon, 1.0)
            
            # Trace out Alice's subsystem
            rho_B = ptrace(rho_pure, [2, 2], 1)
            
            # 4. Interpolate from the outside in!
            rho_boundary = eta_epsilon * rho_pure + (1 - eta_epsilon) * kron(I_half, rho_B)
            
            # Double check physical validity (eigenvalues must be >= 0)
            # Hermitian() prevents tiny floating-point imaginary numbers from ruining the check
            if eigmin(Hermitian(rho_boundary)) >= -1e-6
                count += 1
                valid_states[count] = rho_boundary
                
                # Label is -1 because it is safely on the non-LHS side of the boundary
                valid_labels[count] = -1 
                
                if count % max(1, div(n_target, 10)) == 0
                    println("Generated $count / $n_target Non-LHS boundary states... (Attempts: $total_generated)")
                end
            end
        end
    end
    
    println("Success! Generated $n_target Non-LHS boundary states out of $total_generated attempts.")
    return valid_states, valid_labels
end

n = 10
states_list, labels_list = generate_non_lhs_boundary(n)

# Reshape and save using HDF5
states_3d = reshape(reduce(hcat, states_list), 4, 4, n)

h5open("non_lhs_boundary_states.h5", "w") do file
    write(file, "rho", states_3d)
end

h5open("non_lhs_boundary_labels.h5", "w") do file
    write(file, "labels", labels_list)
end