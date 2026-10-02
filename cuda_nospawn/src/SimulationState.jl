# --- SimulationState.jl ---
using Random
""" struct for simulation parameters """
Base.@kwdef mutable struct SimulationState{T<:AbstractFloat, TI<:Integer}

    model::String = "spinbo1d"

    nmodc1::Int = 0  # number of V_12 coupling modes
    nmodc2::Int = 0  # number of V_02 coupling modes
    nmodc3::Int = 0  # number of V_01 coupling modes
    nmodc ::Int      # = nmodc1+nmodc2+nmodc3
    
    nmodt ::Int = 0  # number of linearly coupled tuning modes (Bathmodes and system modes)
    nmod  ::Int      # = nmodc+nmodt+nmodq+nmodr

    nel   ::Int = 3  # nel+1 = number of diabatic potential matrix elements, e.g., = 3+1, V_0,V_1,V_2,V_12
    nelc  ::Int = 1  # number of diabatic coupling elements, e.g., =1, V_12
    neq   ::Int      # = 2*nmod

    nran   ::Int = 1    # maximal number of trajectories
    ntime  ::Int = 1    # number of time steps

    deltat::T = T(0.05)   # time step [fs]
    t_offset::T      = zero(T)

    max_walkers ::Int = 1

    # allocated when the struct is created
    E     ::Vector{T} = zeros(T, 4)
    wf    ::Matrix{T} = zeros(T, 3, neq)
    kappa ::Matrix{T} = zeros(T, 3, neq)
    lambda::Matrix{T} = zeros(T, 1, neq)
    ecor  ::Vector{T} = zeros(T, 3)
    ecorv ::Matrix{T} = zeros(T, 3, neq)
    enull ::T         = zero(T)

    cmode::Bool  = false  # involves nuclear mode as off-diagonal coupling in the diabatic picture
    spinbo::Bool = false  # spin-boson model where the off-diagonal coupling is a constant

    dh::Vector{T}      = zeros(T, max_walkers)
    
    dm_on::Vector{Int} = zeros(Int, max_walkers)
    dm_to::Vector{Int} = zeros(Int, max_walkers)
    Walker_Oflow::Bool = false
    
    E_ho::Vector{T}  = zeros(T, 2)
    gconst::T        = zero(T)
    lambda_scalar::T = zero(T)
    wcoup::T         = zero(T)
    wcoup0::T        = zero(T)
    w1::T = zero(T); w2::T = zero(T); w3::T = zero(T)
    k1::Vector{T} = zeros(T, 2); k2::Vector{T} = zeros(T, 2); k3::Vector{T} = zeros(T, 2); 

    s_mat::Vector{T} = zeros(T, 4)
    
    kini::Int                 = 3   
    kwrite::Int               = 100
    iran::Int                 = 123
    eTemp::T                  = zero(T)
    norm_vec::Vector{T}       = ones(T, neq)
    init_diab::Bool           = false

    is_adiab::Bool            = false

    adia::Bool  = true
    dia::Bool   = true
    vib::Bool   = true

    # trajectory loop and output
    itraj::Int = 1

    rk4_k1::Vector{T} = zeros(T, neq)
    rk4_k2::Vector{T} = zeros(T, neq)
    rk4_k3::Vector{T} = zeros(T, neq)
    rk4_k4::Vector{T} = zeros(T, neq)
    rk4_utmp::Vector{T} = zeros(T, neq)

    symplct_df::Vector{T} = zeros(T, neq)

    rng::Random.Xoshiro = Random.Xoshiro(iran)
end

""" helper wrapper """
function SimulationState(; T=Float64, TI=Int, 
        nmodc1=0, nmodc2=0, nmodc3=0, nmodt=0, kwargs...)
    nmodc = nmodc1+nmodc2+nmodc3
    nmod  = nmodc+nmodt
    neq   = 2nmod
    return SimulationState{T, TI}(; 
        nmodc1=nmodc1, nmodc2=nmodc2, nmodc3=nmodc3, nmodt=nmodt, 
        nmodc=nmodc, nmod=nmod, neq=neq, kwargs...)
end

