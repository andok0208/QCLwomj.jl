# --- main_gpu.jl ---
using Distributed
using TOML
using CUDA

input_file = length(ARGS) > 0 ? ARGS[1] : "input.toml"
config = isfile(input_file) ? TOML.parsefile(input_file) : Dict{String, Any}()

sim_cfg   = get(config, "simulation",   Dict{String, Any}())
model_cfg = get(config, "model_params", Dict{String, Any}())

requested_gpus = Int(get(sim_cfg, "n_gpus", 1))
available_gpus = length(CUDA.devices())

if requested_gpus > available_gpus
    println("------------------------------------------------------------------")
    println(" [WARNING] Requested n_gpus ($requested_gpus) > Available GPUs ($available_gpus).")
    println("           Falling back to N_GPUS = $available_gpus.")
    println("------------------------------------------------------------------")
    N_GPUS = available_gpus
elseif requested_gpus <= 0
    println(" [WARNING] Invalid n_gpus ($requested_gpus). Fallback to N_GPUS = 1.")
    N_GPUS = 1
else
    N_GPUS = requested_gpus
end

if N_GPUS == 0
    error("No CUDA-capable GPUs detected on this machine.")
end

total_nran = Int(get(sim_cfg, "total_nran", 12_000_000))
prec_str   = String(get(sim_cfg, "precision", "Float32"))

if prec_str == "Float64"
    T, TI = Float64, Int64
elseif prec_str == "Float32"
    T, TI = Float32, Int32
else
    error("Unsupported precision: '$prec_str'. Use 'Float32' or 'Float64'.")
end

model_params = Dict{String, Any}(
    "model"     => String(get(model_cfg, "model", "spinbo")),
    "nmodc1"    => Int(get(model_cfg, "nmodc1", 0)),
    "nmodt"     => Int(get(model_cfg, "nmodt",  1)),
    "init_diab" => Bool(get(model_cfg, "init_diab", true)),
    "ntime"     => Int(get(model_cfg, "ntime", 1600)),
    "deltat"    => Float64(get(model_cfg, "deltat", 0.05))
)

if nprocs() < N_GPUS + 1
    addprocs(N_GPUS)
end

@everywhere using CUDA
@everywhere using Random
@everywhere using StaticArrays

@everywhere include("../src/SimulationState.jl")
@everywhere include("../src/run_simulation_gpu.jl")


function merge_simulation_outputs(outputs)
    merged = deepcopy(outputs[1])
    for i in 2:length(outputs)
        merged.Padf    .+= outputs[i].Padf
        merged.Pdf     .+= outputs[i].Pdf
        merged.c2c1ref .+= outputs[i].c2c1ref
        merged.c2c1imf .+= outputs[i].c2c1imf
        
        if hasproperty(merged, :qf) && !isempty(merged.qf)
            merged.qf  .+= outputs[i].qf
            merged.pf  .+= outputs[i].pf
        end
        if hasproperty(merged, :econf) && !isempty(merged.econf)
            merged.econf .+= outputs[i].econf
            merged.eoffs .+= outputs[i].eoffs
        end
    end
    return merged
end


function main_multi_gpu(N_GPUS, total_nran, T, TI, model_params)
    nran_per_gpu = div(total_nran, N_GPUS)

    println("===================================================")
    println("  Running Multi-GPU Simulation on $N_GPUS GPU(s)")
    println("  Precision     : $T")
    println("  Total nran    : $total_nran ($nran_per_gpu per GPU)")
    println("  Model         : $(model_params["model"])")
    println("===================================================")

    base_cpu_state = SimulationState(
        model     = model_params["model"],
        nmodc1    = TI(model_params["nmodc1"]), 
        nmodt     = TI(model_params["nmodt"]), 
        init_diab = Bool(model_params["init_diab"]), 
        ntime     = TI(model_params["ntime"]),
        deltat    = T(model_params["deltat"]),
        T=T, TI=TI
    )

    # --- JIT Compilation ---
    println("\n--- JIT Compiling on GPUs ---")
    @sync for (gpu_id, worker_id) in enumerate(workers())
        @async remotecall_wait(worker_id) do
            CUDA.device!(gpu_id - 1)
            state = deepcopy(base_cpu_state)
            run_gpu_simulation(state, nran_size=1, T=T, TI=TI, save_output=false)
        end
    end
    println("--- JIT Compilation completed ---\n")

    println("--- Starting Main Parallel Execution ---")
    t_start = time_ns()

    futures = map(enumerate(workers())) do (gpu_id, worker_id)
        remotecall(worker_id) do
            CUDA.device!(gpu_id - 1)
            println("Worker $(myid()) starting on GPU $(CUDA.deviceid(CUDA.device()))")

            state = deepcopy(base_cpu_state)
            state.iran = base_cpu_state.iran + myid() * 10000

            return run_gpu_simulation(state, nran_size=nran_per_gpu, T=T, TI=TI, save_output=false)
        end
    end

    results = fetch.(futures)
    println("\n--- Merging outputs from all GPUs ---")
    merged_output = merge_simulation_outputs(results)

    final_state = deepcopy(base_cpu_state)
    final_state.itraj = total_nran
    output!(final_state, merged_output)

    t_end = time_ns()
    println("===================================================")
    println(" Simulation Total Elapsed Time: $((t_end - t_start)/1e9) sec")
    println(" Results correctly saved to outdat/")
    println("===================================================")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_multi_gpu(N_GPUS, total_nran, T, TI, model_params)
end

