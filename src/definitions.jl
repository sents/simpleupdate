module Definitions
# Useful definitions not related to tensor networks
export σ_x, σ_y, σ_z,
    s_x, s_y, s_z,
    heisenberg_2site,
    zeemanz_1site,
    ising_2site
import LinearAlgebra: I

σ_z = [1 0; 0 -1.0]

σ_x = [
    0 1
    1 0.0
]
σ_y = [
    0 -im
    im 0
]
σ_z = [
    1 0
    0 -1.0
]

s_x = σ_x / 2
s_y = σ_y / 2
s_z = σ_z / 2

heisenberg_2site(J) = -J * (s_x ⊗ s_x + s_y ⊗ s_y + s_z ⊗ s_z)
zeemanz_1site(μ) = -μ * s_z

ising_2site(J) = -J * (σ_z / 2 ⊗ σ_z / 2)
end
