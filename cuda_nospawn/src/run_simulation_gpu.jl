# --- run_simulation_gpu.jl ---
# contains:
#   function run_gpu_simulation
#   function simulation_kernel!
#   function launch_simulation_kernel
#   function launch_todia_kernel
#   function launch_inwf_kernel
#   function initialize_simulation_on_gpu
using CUDA
using Random
using StaticArrays

struct GPUContext{T_f, T_E, T_wf, T_kappa, T_lambda, T_Rho, T_dm_on, TP}
    d_f     ::T_f
    d_E     ::T_E
    d_wf    ::T_wf
    d_kappa ::T_kappa
    d_lambda::T_lambda
    d_Rho   ::T_Rho
    d_dm_on ::T_dm_on
    params  ::TP       # NamedTuple
end

# borrow CPU version (symbolic link)
include("SimulationState.jl")
include("SimulationOutput.jl")
include("input.jl")
include("winit.jl")
include("output.jl")

# GPU version
include("bfactor_gpu.jl")
include("calc_phase_gpu.jl")
include("derivpx_gpu.jl")
#include("rk4_step_gpu.jl")
include("dmhop_gpu.jl")
include("propagate_walker_gpu.jl")
include("smatrix_gpu.jl")
include("inwf_gpu.jl")
include("reduce_metrics_gpu.jl")
include("symplct4_gpu.jl")


function run_gpu_simulation(cpu_state; nran_size=10_000_000, T=Float32, TI=Int32)
    println("===================================================")
    println("  mixed Quantum-Classical Liouville MD Simulation  ")
    println("     (CUDA GPU *without* trajectory spawning)      ")
    println("===================================================")

    #t_start = time_ns()
    nran_size = TI(nran_size)

    Random.seed!(cpu_state.iran)
    CUDA.seed!(cpu_state.iran)

    input!(cpu_state)
    winit!(cpu_state)

    params = (
        nmodc = TI(cpu_state.nmodc),
        nmod  = TI(cpu_state.nmod),
        neq   = TI(cpu_state.neq),
        kini  = TI(cpu_state.kini),
        cmode = cpu_state.cmode,
        spinbo = cpu_state.spinbo,
        init_diab = cpu_state.init_diab,
        deltat = cpu_state.deltat,
        t_offset = cpu_state.t_offset
    )

    d_f      = CUDA.zeros(T, cpu_state.neq, nran_size)
    d_Rho    = CUDA.zeros(Complex{T}, nran_size)
    d_dm_on = CUDA.ones(TI, nran_size)

    d_Pdf_step1 = CUDA.zeros(T, nran_size)
    d_Pdf_step2 = CUDA.zeros(T, nran_size)

    # GPU-side output arrays to prevent host-device synchronization during loop
    d_Padf_gpu    = CUDA.zeros(T, cpu_state.ntime, 4)
    d_Pdf_gpu     = CUDA.zeros(T, cpu_state.ntime, 4)
    d_c2c1ref_gpu = CUDA.zeros(T, cpu_state.ntime)
    d_c2c1imf_gpu = CUDA.zeros(T, cpu_state.ntime)
    d_qf_gpu      = CUDA.zeros(T, cpu_state.ntime, cpu_state.nmod)
    d_pf_gpu      = CUDA.zeros(T, cpu_state.ntime, cpu_state.nmod)

    # output arrays
    out = SimulationOutput(nmod=cpu_state.nmod, ntime=cpu_state.ntime, T=T, TI=TI)

    #t_end = time_ns()
    #println("--- elapsed time: $((t_end - t_start)/1e9) sec")

    t_start = time_ns()
    println("--- Initializing walkers ...")
    d_f, d_Rho, d_dm_on = 
        initialize_simulation_on_gpu(d_f, d_Rho, d_dm_on, cpu_state, nran_size, params)

    CUDA.synchronize()
    t_end = time_ns()
    println("--- elapsed time: $((t_end - t_start)/1e9) sec")
    t_start = time_ns()

    threads_per_block = 256
    blocks_per_grid = div(nran_size + threads_per_block - 1, threads_per_block)

    println("Starting main simulation loop on GPU... nran = $nran_size")
    println("threads_per_block = $threads_per_block, blocks_per_grid = $blocks_per_grid")

    d_zeta1 = CuArray{T}(undef, nran_size) 
    d_zeta2 = CuArray{T}(undef, nran_size)

    t0 = zero(T)
    for ti in 1:(cpu_state.ntime - 1)
        t1 = T(t0 + (ti - 1) * cpu_state.deltat)
        t2 = T(t1 + cpu_state.deltat)

        Random.rand!(d_zeta1)
        Random.rand!(d_zeta2)

        # one-step propagation of all walkers (benchmark first and last steps conditionally to avoid synchronization)
        if ti == 1 || ti == (cpu_state.ntime - 1)
            gpu_time = CUDA.@elapsed launch_simulation_kernel(
                d_f, d_Rho, d_dm_on, cpu_state.E, cpu_state.wf, cpu_state.kappa, cpu_state.lambda,
                d_zeta1, d_zeta2, t1, t2, 
                nran_size, Val(Int(params.neq)), params, threads_per_block, blocks_per_grid
            )
            println("gpu_time for simulation_kernel at ti = $ti : $gpu_time sec.")
        else
            launch_simulation_kernel(
                d_f, d_Rho, d_dm_on, cpu_state.E, cpu_state.wf, cpu_state.kappa, cpu_state.lambda,
                d_zeta1, d_zeta2, t1, t2, 
                nran_size, Val(Int(params.neq)), params, threads_per_block, blocks_per_grid
            )
        end

        # summing diabatic population
        if ti == 1 || ti == (cpu_state.ntime - 1)
            gpu_time = CUDA.@elapsed launch_todia_kernel(
                d_Pdf_step1, d_Pdf_step2, d_f, d_Rho, d_dm_on, cpu_state.E, cpu_state.lambda, cpu_state.kappa, 
                nran_size, Val(Int(params.neq)), params, threads_per_block, blocks_per_grid
            )
            #println("gpu_time for todia_kernel      at ti = $ti : $gpu_time sec.")
        else
            launch_todia_kernel(
                d_Pdf_step1, d_Pdf_step2, d_f, d_Rho, d_dm_on, cpu_state.E, cpu_state.lambda, cpu_state.kappa, 
                nran_size, Val(Int(params.neq)), params, threads_per_block, blocks_per_grid
            )
        end

        #=
        # reduction on GPU kernel
        @cuda(threads=threads_per_block, blocks=blocks_per_grid, reduce_metrics_kernel!(
            d_Padf_gpu, d_c2c1ref_gpu, d_c2c1imf_gpu, d_Pdf_gpu, d_qf_gpu, d_pf_gpu,
            d_dm_on, d_Rho, d_Pdf_step1, d_Pdf_step2, d_f,
            ti, nran_size, Val(Int(params.nmod)), cpu_state.vib
        ))
        =#

        # reduction with mapreduce
        reduce_metrics_gpu!(
            d_Padf_gpu, d_c2c1ref_gpu, d_c2c1imf_gpu, d_Pdf_gpu, d_qf_gpu, d_pf_gpu,
            d_dm_on, d_Rho, d_Pdf_step1, d_Pdf_step2, d_f, ti, params.nmod, cpu_state.vib
        )
    end

    CUDA.synchronize()
    
    # Copy results from GPU back to the SimulationOutput struct
    copyto!(out.Padf, d_Padf_gpu)
    copyto!(out.Pdf,  d_Pdf_gpu)
    copyto!(out.c2c1ref, d_c2c1ref_gpu)
    copyto!(out.c2c1imf, d_c2c1imf_gpu)
    if cpu_state.vib
        copyto!(out.qf, d_qf_gpu)
        copyto!(out.pf, d_pf_gpu)
    end

    t_end = time_ns()
    println("--- elapsed time: $((t_end - t_start)/1e9) sec")

    cpu_state.itraj = nran_size
    output!(cpu_state, out)
    #println("GPU Simulation completed successfully!")
end

""" simulation kernel for one thread """
function simulation_kernel!(
        d_f, d_Rho, d_dm_on, d_E, d_wf, d_kappa, d_lambda,
        d_zeta1, d_zeta2, t1, t2, nran, ::Val{NEQ}, params
) where {NEQ}
    itraj = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if itraj <= nran
        ctx = GPUContext(
            d_f,
            d_E,
            d_wf,
            d_kappa,
            d_lambda,
            d_Rho,
            d_dm_on,
            params
        )
        dmhop_gpu!(itraj, ctx, d_zeta1[itraj], d_zeta2[itraj], t2)
        propagate_walker_gpu!(itraj, ctx, t1, t2, Val(NEQ))
    end
    return nothing
end

""" Function barriers to compile kernels with static array type parameters """
function launch_simulation_kernel(
    d_f, d_Rho, d_dm_on, E_cpu, wf_cpu, kappa_cpu, lambda_cpu,
    d_zeta1, d_zeta2, t1, t2, nran_size, ::Val{NEQ}, params, threads_per_block, blocks_per_grid
) where {NEQ}
    T = eltype(d_f)
    d_E_static      = SVector{size(E_cpu, 1), T}(E_cpu)
    d_wf_static     = SMatrix{size(wf_cpu, 1), NEQ, T}(wf_cpu)
    d_kappa_static  = SMatrix{size(kappa_cpu, 1), NEQ, T}(kappa_cpu)
    d_lambda_static = SMatrix{size(lambda_cpu, 1), NEQ, T}(lambda_cpu)

    @cuda(threads=threads_per_block, blocks=blocks_per_grid,
        simulation_kernel!(
            d_f, d_Rho, d_dm_on, d_E_static, d_wf_static, d_kappa_static, d_lambda_static,
            d_zeta1, d_zeta2, t1, t2, 
            nran_size, Val(NEQ), params
        )
    )
end

function launch_todia_kernel(
    d_Pdf_step1, d_Pdf_step2, d_f, d_Rho, d_dm_on, E_cpu, lambda_cpu, kappa_cpu,
    nran_size, ::Val{NEQ}, params, threads_per_block, blocks_per_grid
) where {NEQ}
    T = eltype(d_f)
    d_E_static      = SVector{size(E_cpu, 1), T}(E_cpu)
    d_kappa_static  = SMatrix{size(kappa_cpu, 1), NEQ, T}(kappa_cpu)
    d_lambda_static = SMatrix{size(lambda_cpu, 1), NEQ, T}(lambda_cpu)

    @cuda(threads=threads_per_block, blocks=blocks_per_grid,
        todia_kernel!(
            d_Pdf_step1, d_Pdf_step2, d_f, d_Rho, d_dm_on, d_E_static, d_lambda_static, d_kappa_static, 
            nran_size, params
        )
    )
end

function launch_inwf_kernel(
    d_f, d_Rho, d_dm_on, nran_size, d_norm_vec, E_cpu, lambda_cpu, kappa_cpu,
    ::Val{NEQ}, params, threads, blocks
) where {NEQ}
    T = eltype(d_f)
    d_E_static      = SVector{size(E_cpu, 1), T}(E_cpu)
    d_kappa_static  = SMatrix{size(kappa_cpu, 1), NEQ, T}(kappa_cpu)
    d_lambda_static = SMatrix{size(lambda_cpu, 1), NEQ, T}(lambda_cpu)
    d_rand_x = CUDA.randn(T, nran_size, params.nmod)
    d_rand_p = CUDA.randn(T, nran_size, params.nmod)
    d_rand_zeta = CUDA.rand(T, nran_size)

    @cuda(threads=threads, blocks=blocks, 
        inwf_gpu_kernel!(
            d_f, d_Rho, d_dm_on, nran_size, d_norm_vec, d_E_static, d_lambda_static, d_kappa_static, params,
            d_rand_x, d_rand_p, d_rand_zeta
        )
    )
end

function initialize_simulation_on_gpu(
        d_f, d_Rho, d_dm_on, state::SimulationState, nran_size, params)
    T = eltype(d_f)
    TI = eltype(nran_size)

    d_norm_vec = CuArray{T}(state.norm_vec)

    threads = 256
    blocks  = cld(nran_size, threads)

    launch_inwf_kernel(
        d_f, d_Rho, d_dm_on, TI(nran_size), d_norm_vec, state.E, state.lambda, state.kappa,
        Val(Int(params.neq)), params, threads, blocks
    )
    CUDA.synchronize()
    return d_f, d_Rho, d_dm_on
end

