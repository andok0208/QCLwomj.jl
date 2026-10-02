#--- SimulationOutput.jl ---
""" struct for output """
Base.@kwdef mutable struct SimulationOutput{T<:AbstractFloat, TI<:Integer}

    nmod  ::Int = 1
    ntime ::Int = 1

    Pad     ::Matrix{T} = zeros(T, ntime, 4) # adiabatic population
    Pd      ::Matrix{T} = zeros(T, ntime, 4) # diabatic population
    q_hist  ::Matrix{T} = zeros(T, ntime, nmod) # coordinates history
    p_hist  ::Matrix{T} = zeros(T, ntime, nmod) # momentua history

    c2c1re  ::Vector{T} = zeros(T, ntime)
    c2c1im  ::Vector{T} = zeros(T, ntime)
    econ    ::Vector{T} = zeros(T, ntime)
    econ_sos::Vector{T} = zeros(T, ntime)
    eoff    ::Vector{T} = zeros(T, ntime)
    Pd_sos  ::Vector{T} = zeros(T, ntime)
    Pad_sos ::Vector{T} = zeros(T, ntime)

    # ensemble average storage
    Padf     ::Matrix{T} = zeros(T, ntime, 4)
    Pdf      ::Matrix{T} = zeros(T, ntime, 4)
    qf       ::Matrix{T} = zeros(T, ntime, nmod) # (ntime, nmod)
    pf       ::Matrix{T} = zeros(T, ntime, nmod)
    c2c1ref  ::Vector{T} = zeros(T, ntime)
    c2c1imf  ::Vector{T} = zeros(T, ntime)
    econf    ::Vector{T} = zeros(T, ntime)
    econ_sosf::Vector{T} = zeros(T, ntime)
    eoffs    ::Vector{T} = zeros(T, ntime)
    Pd_sosf  ::Vector{T} = zeros(T, ntime)
    Pad_sosf ::Vector{T} = zeros(T, ntime)
end

function SimulationOutput(; T=Float64, TI=Int, nmod=1, ntime=1600, kwargs...)
    return SimulationOutput{T, TI}(; nmod=nmod, ntime=ntime, kwargs...)
end

