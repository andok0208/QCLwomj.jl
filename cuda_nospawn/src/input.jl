# --- input.jl ---
"""
input for molecular system
the S2 vertical energy is by convention zero, i.e. E(2) = 0.d0
"""
function input!(state::SimulationState)
    ec   = 1.6021766e-19  # C
    hbar = 1.0545718e-34  # J s
    tofsi = ec * 1e-15 / hbar  # convert to 1/fs

    kB = 1.380649e-23  # J K-1
    kelv2eV = kB/ec  # Kelvin to eV
    state.eTemp = state.eTemp * kelv2eV * tofsi

    if state.model == "pyrz3d"
        println("model = $(state.model): 3D pyrazine (Schneider & Domcke, CPL 159 (1989) 61)")
        state.E_ho[1]  = -0.9 * tofsi
        state.w1    =  0.12620 * tofsi
        state.k1[1] =  0.03700 * tofsi
        state.k1[2] = -0.25340 * tofsi
        state.w2    =  0.07400 * tofsi
        state.k2[1] = -0.10540 * tofsi
        state.k2[2] =  0.14880 * tofsi
        state.wcoup =  0.11780 * tofsi
        state.wcoup0=  state.wcoup
        state.lambda_scalar = 0.26180 * tofsi
        println(" lambda   = ", state.lambda_scalar / tofsi)
    elseif state.model == "spinbo1d" || state.model == "spinbo3d"
        println("model = $(state.model): Spin-Boson model (1D / 3D)")
        state.E_ho[1]  = 0.3
        state.E_ho[2]  = 0.6
        g = 0.1  # eV
        state.gconst = g
        state.w1    = g
        state.k1[1] = 2.5g
        state.k1[2] = 3.5g
        state.w2    = 1.2g
        state.k2[1] = state.k1[1]
        state.k2[2] = state.k1[2]
        state.w3    = 0.9g
        state.k3[1] = state.k1[1]
        state.k3[2] = state.k1[2]
    else
        error("wrong model: $(state.model)")
    end
end

