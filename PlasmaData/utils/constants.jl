# Physical constants and normalization parameters for plasma simulation in Julia

# Fundamental physical constants
const ATOMIC_MASS_UNIT = 1.660539040e-27     # kg
const ELECTRON_MASS = 9.10938356e-30         # kg
const ELECTRON_CHARGE = 1.60217662e-19       # C
const SPEED_OF_LIGHT = 2.99792458e8          # m/s
const BOLTZMANN_CONST = 1.380649e-23         # J/K

# Geometry
const MINOR_RADIUS = 0.22                    # meters
const MAJOR_RADIUS = 0.68                    # meters
const B_FIELD = 5.0                          # Tesla

# Plasma composition
const ION_CHARGE = 1                         # Ionization level (Z)
const MASS_RATIO = 3672.3036                 # mi/me
const ION_MASS_NUMBER = 2.0                  # m_i/m_proton (mu)
const PROTON_MASS = 1.007276 * ATOMIC_MASS_UNIT  # kg
const ION_MASS = ION_MASS_NUMBER * PROTON_MASS   # kg

# Initial conditions
const ELECTRON_TEMP = 25.0                   # eV
const ION_TEMP = 25.0                        # eV
const PLASMA_DENSITY = 5e19                  # m^-3

# Magnetic field function
function compute_magnetic_field(radius::Float64)
    center_field = (B_FIELD * MAJOR_RADIUS) / (MAJOR_RADIUS + MINOR_RADIUS)
    return center_field
end

# Derived plasma parameters
const ELECTRON_THERMAL_SPEED = sqrt(ELECTRON_CHARGE * ELECTRON_TEMP / ION_MASS)
const ION_THERMAL_SPEED = sqrt(ELECTRON_CHARGE * ION_TEMP / ION_MASS)

# Time and space normalization
const REFERENCE_TIME = sqrt((MAJOR_RADIUS * MINOR_RADIUS) / 2.0) / ELECTRON_THERMAL_SPEED
const SPACE_FACTOR = 100.0 * MINOR_RADIUS    # cm conversion

# Plasma parameters
const ETA = 63.6094
const TAU_T = 1.0

# Dimensionless parameters
const EPS_R = 0.4889
const EPS_V = 0.3496
const ALPHA_D = 0.0012
const KAPPA_E = 7.6771
const KAPPA_I = 0.2184
const EPS_G = 0.0550
const EPS_GE = 0.0005

# Source terms
const N_SRC_A = 20.0
const ENER_SRC_A = 0.001
const X_SRC = -0.15
const SIG_SRC = 0.01

# Normalization
const DIFF_X_NORM = 50.0
const DIFF_Y_NORM = 50.0