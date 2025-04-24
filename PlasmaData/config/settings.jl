module PlasmaSettings

export PROJECT_ROOT, DATA_PATH, RESULTS_PATH,
       LAYERS, N_OUTPUTS, TRAINING_HOURS, STEPS_PER_HOUR, TOTAL_EPOCHS,
       SAMPLE_BATCH_SIZE, TRAIN_FRACTION,
       INIT_WEIGHT_DEN, INIT_WEIGHT_TE,
       SCATTER_POINT_SIZE, PLOT_FREQUENCY, SAVE_FREQUENCY,
       NOISE_MEAN, NOISE_STD,
       NX, NY, NZ, DT, N_TIMESTEPS, INITIAL_FRAME, FINAL_FRAME,
       LX, LY, LZ, DX, DY, DZ, GRID_SPACING,
       USE_PDE, compute_diffusion_norms,
       DiffX_norm, DiffY_norm, DiffZ_norm

using Printf

# -----------------------------
# Paths (set relative to this file)
# -----------------------------
const PROJECT_ROOT = normpath(@__DIR__, "..")
const DATA_PATH    = joinpath(PROJECT_ROOT, "data", "plasma_data.h5")
const RESULTS_PATH = joinpath(PROJECT_ROOT, "results")

# -----------------------------
# Model architecture
# -----------------------------
const LAYERS     = [3, 50, 50, 50, 50, 50, 1]
const N_OUTPUTS  = 1

# -----------------------------
# Training hyperparameters
# -----------------------------
const TRAINING_HOURS    = 20.0
const STEPS_PER_HOUR    = 1000
const TOTAL_EPOCHS      = Int(TRAINING_HOURS * STEPS_PER_HOUR)
const SAMPLE_BATCH_SIZE = 500
const TRAIN_FRACTION    = 1.0

# -----------------------------
# Initial weights
# -----------------------------
const INIT_WEIGHT_DEN = 1.0
const INIT_WEIGHT_TE  = 1.0

# -----------------------------
# Plotting
# -----------------------------
const SCATTER_POINT_SIZE = 2.5
const PLOT_FREQUENCY     = 100
const SAVE_FREQUENCY     = 1000

# -----------------------------
# Data settings
# -----------------------------
const NOISE_MEAN = 1.0
const NOISE_STD  = 0.25

# -----------------------------
# Grid settings
# -----------------------------
const NX = 256
const NY = 128
const NZ = 32

# -----------------------------
# Time settings
# -----------------------------
const DT            = 5e-6
const N_TIMESTEPS   = 16000
const INITIAL_FRAME = 0
const FINAL_FRAME   = 398

# -----------------------------
# Domain settings (normalized)
# -----------------------------
const LX = 0.35
const LY = 0.25
const LZ = 20.0

# -----------------------------
# Grid spacing
# -----------------------------
const DX = LX / NX
const DY = LY / NY
const DZ = LZ / NZ
const GRID_SPACING = [DX, DY, DZ, DT]

# -----------------------------
# PDE toggle
# -----------------------------
const USE_PDE = true

# -----------------------------
# Function to compute diffusion norms
# -----------------------------
function compute_diffusion_norms(dx::Vector{Float64})
    DiffX = 2π / (dx[1] * 3.0)
    DiffY = 2π / (dx[2] * 3.0)
    DiffZ = 2π / (dx[3] * 3.0)
    return Dict(
        "DiffX_norm" => DiffX^2,
        "DiffY_norm" => DiffY^2,
        "DiffZ_norm" => DiffZ^2
    )
end

# -----------------------------
# Compute and expose norms
# -----------------------------
const DIFF_NORMS = compute_diffusion_norms(GRID_SPACING)
const DiffX_norm = DIFF_NORMS["DiffX_norm"]
const DiffY_norm = DIFF_NORMS["DiffY_norm"]
const DiffZ_norm = DIFF_NORMS["DiffZ_norm"]

end # module