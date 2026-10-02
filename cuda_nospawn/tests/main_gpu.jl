# --- main.jl ---
using Distributed
using TOML

input_file = length(ARGS) > 0 ? ARGS[1] : "input.toml"

config = isfile(input_file) ? TOML.parsefile(input_file) : Dict{String, Any}()

sim_cfg   = get(config, "simulation",   Dict{String, Any}())
model_cfg = get(config, "model_params", Dict{String, Any}())

N_GPUS     = Int(get(sim_cfg, "n_gpus", 1))
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
    "model"     => String(get(model_cfg, "model", "spinbo1d")),
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

function main_multi_gpu(N_GPUS, total_nran, T, TI, model_params)
    nran_per_gpu = div(total_nran, N_GPUS)

    println("===================================================")
    println("  Running Simulation on $N_GPUS GPU(s)")
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
            run_gpu_simulation(state, nran_size=1, T=T, TI=TI)
        end
    end
    println("--- JIT Compilation completed ---\n")

    println("--- Starting Main Parallel Execution ---")
    t_start = time_ns()

    @sync for (gpu_id, worker_id) in enumerate(workers())
        @async remotecall_wait(worker_id) do
            CUDA.device!(gpu_id - 1)
            println("Worker $(myid()) starting on GPU $(CUDA.deviceid(CUDA.device()))")

            state = deepcopy(base_cpu_state)
            state.iran = base_cpu_state.iran + myid() * 10000

            run_gpu_simulation(state, nran_size=nran_per_gpu, T=T, TI=TI)
        end
    end

    t_end = time_ns()
    println("===================================================")
    println(" Simulation Total Elapsed Time: $((t_end - t_start)/1e9) sec")
    println("===================================================")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_multi_gpu(N_GPUS, total_nran, T, TI, model_params)
end

