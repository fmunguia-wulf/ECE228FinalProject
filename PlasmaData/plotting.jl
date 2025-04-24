module PlasmaDataLoader

using HDF5
using Random
using Statistics

# --- Import custom constants and settings ---
include("/utils/constants.jl")
include("/config/settings.jl")

export load_data, preprocess_data, add_noise

"""
    load_data(data_path::String) -> Dict{String, Any}

Loads and preprocesses plasma data from HDF5 file.
"""
function load_data(data_path::String)
    h5 = h5open(data_path, "r")
    x_x = read(h5["x_x"])
    x_y = read(h5["x_y"])
    x_z = read(h5["x_z"])
    x_t = read(h5["x_t"])
    y_den = read(h5["y_den"])
    y_Te = read(h5["y_Te"])
    close(h5)

    # Calculate weights
    init_weight_den = 1.0 / median(abs.(y_den))
    init_weight_Te = 1.0 / median(abs.(y_Te))

    # Sample training data
    N_train = floor(Int, TRAIN_FRACTION * length(y_den))
    idx = randperm(length(y_den))[1:N_train]

    return Dict(
        "x_train" => x_x[idx],
        "y_train" => x_y[idx],
        "z_train" => x_z[idx],
        "t_train" => x_t[idx],
        "v1_train" => y_den[idx],
        "v5_train" => y_Te[idx],
        "weights" => Dict(
            "den" => init_weight_den,
            "Te" => init_weight_Te
        )
    )
end

"""
    preprocess_data(data::Dict{String, Vector}) -> Dict{String, Any}

Preprocesses preloaded data.
"""
function preprocess_data(data::Dict{String, Vector})
    # Compute normalization factors
    init_weight_den = 1.0 / median(abs.(data["y_den"]))
    init_weight_Te = 1.0 / median(abs.(data["y_Te"]))

    # Create training indices
    N_train = floor(Int, TRAIN_FRACTION * length(data["y_den"]))
    idx = randperm(length(data["y_den"]))[1:N_train]

    return Dict(
        "x_train" => data["x_x"][idx],
        "y_train" => data["x_y"][idx],
        "z_train" => data["x_z"][idx],
        "t_train" => data["x_t"][idx],
        "v1_train" => data["y_den"][idx],
        "v5_train" => data["y_Te"][idx],
        "weights" => Dict(
            "den" => init_weight_den,
            "Te" => init_weight_Te
        )
    )
end

"""
    add_noise(data::Vector{T}) -> Vector{T} where T <: Number

Applies Gaussian noise to input data.
"""
function add_noise(data::Vector{T}) where T <: Number
    noise = randn(length(data)) .* NOISE_STD .+ NOISE_MEAN
    return data .* noise
end

end # module
