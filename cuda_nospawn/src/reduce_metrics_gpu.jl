# --- reduce_metrics_gpu.jl ---
using CUDA
using StaticArrays

""" with mapreduce """
function reduce_metrics_gpu!(
    d_Padf_gpu, d_c2c1ref_gpu, d_c2c1imf_gpu, d_Pdf_gpu, d_qf_gpu, d_pf_gpu,
    d_dm_on, d_Rho, d_Pdf_step1, d_Pdf_step2, d_f, ti, nmod, vib)

    T = eltype(d_f)
    zeroT = T(0.0)

    CUDA.@allowscalar d_Padf_gpu[ti, 1] = mapreduce(
        (dmo, rho) -> dmo == 1 ? real(rho) : zeroT,
        +,
        d_dm_on, d_Rho
    )

    CUDA.@allowscalar d_Padf_gpu[ti, 2] = mapreduce(
        (dmo, rho) -> dmo == 4 ? real(rho) : zeroT,
        +,
        d_dm_on, d_Rho
    )

    CUDA.@allowscalar d_c2c1ref_gpu[ti] = mapreduce(
        #(dmo, rho) -> dmo == 2 ? real(rho) : zeroT,
        (dmo, rho) -> (dmo == 2 || dmo == 3) ? real(rho)/2 : zeroT,
        +, d_dm_on, d_Rho
    )

    CUDA.@allowscalar d_c2c1imf_gpu[ti] = mapreduce(
        (dmo, rho) -> dmo == 2 ? imag(rho) : zeroT,
        +, d_dm_on, d_Rho
    )/2 - mapreduce(
        (dmo, rho) -> dmo == 3 ? imag(rho) : zeroT,
        +, d_dm_on, d_Rho
    )/2

    CUDA.@allowscalar d_Pdf_gpu[ti, 1] = mapreduce(x -> x, +, d_Pdf_step1)
    CUDA.@allowscalar d_Pdf_gpu[ti, 2] = mapreduce(x -> x, +, d_Pdf_step2)

    if vib
        sum_f = mapreduce(T, +, d_f, dims=2)
        d_qf_gpu[ti, :] .= sum_f[2*(1:nmod), 1]
        d_pf_gpu[ti, :] .= sum_f[2*(1:nmod) .- 1, 1]
    end

end

""" invoking kernel """
@inline function block_reduce!(val, shared_mem)
    tid = threadIdx().x
    shared_mem[tid] = val
    CUDA.sync_threads()
    
    s = 128
    while s > 0
        if tid <= s
            shared_mem[tid] += shared_mem[tid + s]
        end
        CUDA.sync_threads()
        s >>= 1
    end
    return shared_mem[1]
end


function reduce_metrics_kernel!(
    d_Padf, d_c2c1ref, d_c2c1imf, d_Pdf, d_qf, d_pf,
    d_dm_on, d_Rho, d_Pdf_step1, d_Pdf_step2, d_f,
    ti, nran_size, ::Val{NMOD}, vib
) where {NMOD}

    tid = threadIdx().x
    itraj = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    T = eltype(d_Padf)

    # Local thread-level accumulation
    sum_Padf1 = zero(T)
    sum_Padf2 = zero(T)
    sum_c2c1ref = zero(T)
    sum_c2c1imf = zero(T)
    sum_Pdf1 = zero(T)
    sum_Pdf2 = zero(T)

    sum_qf = zero(MVector{NMOD, T})
    sum_pf = zero(MVector{NMOD, T})

    while itraj <= nran_size
        dmo = d_dm_on[itraj]
        rho = d_Rho[itraj]
        r_rho = real(rho)
        i_rho = imag(rho)

        if dmo == 1
            sum_Padf1 += r_rho
        elseif dmo == 4
            sum_Padf2 += r_rho
        #elseif dmo == 2
        #    sum_c2c1ref += r_rho
        #    sum_c2c1imf += i_rho
        elseif dmo == 2
            sum_c2c1ref += r_rho * T(0.5)
            sum_c2c1imf += i_rho * T(0.5)
        elseif dmo == 3
            sum_c2c1ref += r_rho * T(0.5)
            sum_c2c1imf += (-i_rho) * T(0.5)
        end

        sum_Pdf1 += d_Pdf_step1[itraj]
        sum_Pdf2 += d_Pdf_step2[itraj]

        if vib
            for m in 1:NMOD
                sum_qf[m] += d_f[2 * m, itraj]
                sum_pf[m] += d_f[2 * m - 1, itraj]
            end
        end

        itraj += stride
    end

    # Shared memory for block-level reduction (block size is 256)
    shared_mem = CUDA.CuStaticSharedArray(T, 256)

    # Reduce variables sequentially using shared memory and atomics
    b_sum_Padf1 = block_reduce!(sum_Padf1, shared_mem)
    if tid == 1
        CUDA.@atomic d_Padf[ti, 1] += b_sum_Padf1
    end
    CUDA.sync_threads()

    b_sum_Padf2 = block_reduce!(sum_Padf2, shared_mem)
    if tid == 1
        CUDA.@atomic d_Padf[ti, 2] += b_sum_Padf2
    end
    CUDA.sync_threads()

    b_sum_c2c1ref = block_reduce!(sum_c2c1ref, shared_mem)
    if tid == 1
        CUDA.@atomic d_c2c1ref[ti] += b_sum_c2c1ref
    end
    CUDA.sync_threads()

    b_sum_c2c1imf = block_reduce!(sum_c2c1imf, shared_mem)
    if tid == 1
        CUDA.@atomic d_c2c1imf[ti] += b_sum_c2c1imf
    end
    CUDA.sync_threads()

    b_sum_Pdf1 = block_reduce!(sum_Pdf1, shared_mem)
    if tid == 1
        CUDA.@atomic d_Pdf[ti, 1] += b_sum_Pdf1
    end
    CUDA.sync_threads()

    b_sum_Pdf2 = block_reduce!(sum_Pdf2, shared_mem)
    if tid == 1
        CUDA.@atomic d_Pdf[ti, 2] += b_sum_Pdf2
    end
    CUDA.sync_threads()

    if vib
        for m in 1:NMOD
            b_sum_q = block_reduce!(sum_qf[m], shared_mem)
            if tid == 1
                CUDA.@atomic d_qf[ti, m] += b_sum_q
            end
            CUDA.sync_threads()

            b_sum_p = block_reduce!(sum_pf[m], shared_mem)
            if tid == 1
                CUDA.@atomic d_pf[ti, m] += b_sum_p
            end
            CUDA.sync_threads()
        end
    end

    return nothing
end

