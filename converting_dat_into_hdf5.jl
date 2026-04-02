using Serialization
using HDF5


rho = deserialize("random_ent_states.dat")

array_rho = reshape(reduce(hcat, rho), 4, 4, length(rho))

h5open("random_ent_states_array.h5", "w") do file
    write(file, "rho", array_rho)
end