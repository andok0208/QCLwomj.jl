# --- winit.jl ---
""" initialize molecular data variables for systems with nmod harmonic modes """
function winit!(state::SimulationState)
    T = eltype(state.gconst)
    
    if state.model == "pyrz3d"
        state.cmode  = true
        state.spinbo = false
    elseif state.model == "spinbo1d" || state.model == "spinbo3d"
        state.cmode  = false
        state.spinbo = true
    else
        error("wrong model: $(state.model)")
    end

    state.E[2] = state.E_ho[1]
    state.E[3] = state.E_ho[2]
    state.E[4] = state.gconst  # g=2V_12

    # coupling modes (nmodc1)
    if state.cmode
        state.wf[1, :]  .= state.wcoup0
        state.wf[2, :]  .= state.wcoup^2 / state.wcoup0
        state.wf[3, :]  .= state.wcoup^2 / state.wcoup0
        state.lambda[1, 1] = state.lambda_scalar
    end

    @inbounds begin
    # tuning modes (nmodt)
    for i in 1:state.nel
        m = 1
        if m > state.nmodt; continue; end
        state.wf[i, state.nmodc + m] = state.w1
        if i != 1; state.kappa[i, state.nmodc + m] = state.k1[i-1]; end

        m += 1
        if m > state.nmodt; continue; end
        state.wf[i, state.nmodc + m] = state.w2
        if i != 1; state.kappa[i, state.nmodc + m] = state.k2[i-1]; end

        m += 1
        if m > state.nmodt; continue; end
        state.wf[i, state.nmodc + m] = state.w3
        if i != 1; state.kappa[i, state.nmodc + m] = state.k3[i-1]; end
    end

    # Stabilization Energies
    @simd for m in 1:state.nmod
        @simd for i in 1:state.nel
            if state.wf[i, m] != zero(T)
                state.ecorv[i, m] = state.kappa[i, m]^2 / 2state.wf[i, m]
            else
                state.ecorv[i, m] = zero(T)
            end
        end
    end

    # contribution from coupling modes
    @simd for i in 1:state.nel
        @simd for m in 1:state.nmodc1
            if state.wf[i, m] != zero(T)
                state.ecorv[i, m] = state.lambda[1, m]^2 / 2state.wf[i, m]
            end
        end
        # skipping parts for nmodc2, nmodc3
    end

    # total stabilization energy
    @simd for i in 1:state.nel
        state.ecor[i] = zero(T)
        @simd for m in 1:state.nmod
            state.ecor[i] += state.ecorv[i, m]
        end
    end

    state.enull = zero(T)
    @simd for i in 1:state.nmod
        state.enull += state.wf[1, i]/2
    end

    # norms for the nuclear distribution
    @simd for m in 1:state.nmod
        if state.eTemp > 1e-4
            state.norm_vec[m] = sqrt( one(T) / 2tanh(state.wf[1, m] / 2state.eTemp) )
        else  # ground state (T=0)
            state.norm_vec[m] = sqrt(T(0.5))
        end
    end
    end  # @inbounds

    println("ntime = $(state.ntime)")
    println("kini  = $(state.kini)")
    println("nelc  = $(state.nelc), nel = $(state.nel), vib = $(state.vib)")
    println("nmodc = $(state.nmodc), nmodt = $(state.nmodt), nmod=$(state.nmod), neq=$(state.neq)")
    println("cmode = $(state.cmode), spinbo = $(state.spinbo)")
    println("eTemp = $(state.eTemp)")
    println("iran  = $(state.iran)")
    #println("E      = $(state.E)")
    #println("wf     = $(state.wf)")
    #println("kappa  = $(state.kappa)")
    #println("lambda = $(state.lambda)")
end

