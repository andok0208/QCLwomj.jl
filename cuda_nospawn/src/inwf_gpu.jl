# --- inwf_gpu.jl ---
""" initialize d_f, d_Rho, d_dm_on """
function inwf_gpu_kernel!(
         d_f, d_Rho, d_dm_on, nran_size, d_norm_vec, d_E, d_lambda, d_kappa, params,
         d_rand_x, d_rand_p, d_rand_zeta)

    itraj = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if itraj > nran_size; return nothing; end

    T = eltype(d_f)
    
    # Wigner distribution
    @inbounds for i in 1:params.nmod
        x = zero(T)
        p = zero(T)
        if params.kini == 3
            x = d_norm_vec[i] * d_rand_x[itraj, i]
            p = d_norm_vec[i] * d_rand_p[itraj, i]
        end
        d_f[2*i - 1, itraj] = p
        d_f[2*i,     itraj] = x
    end

    if !params.init_diab
        is_adiab = false

        r1 = zero(Complex{T})
        r2 = zero(Complex{T})
        r3 = zero(Complex{T})
        r4 = one(Complex{T})

        r1, r2, r3, r4, current_is_adiab = toadia_gpu!(
            d_f, itraj, r1, r2, r3, r4, is_adiab, d_E, d_lambda, d_kappa, params)

        w1 = abs(r1); w2 = abs(r2); w3 = abs(r3); w4 = abs(r4)
        w_sum = w1 + w2 + w3 + w4
        w1 /= w_sum; w2 /= w_sum; w3 /= w_sum; w4 /= w_sum

        bound4 = w4
        bound3 = w3 + bound4
        bound2 = w2 + bound3
        bound1 = w1 + bound2

        zeta = d_rand_zeta[itraj]
        
        if zeta <= bound4
            chosen_dmo = 4; chosen_weight = w4; chosen_rho = r4
        elseif zeta <= bound3
            chosen_dmo = 3; chosen_weight = w3; chosen_rho = r3
        elseif zeta <= bound2
            chosen_dmo = 2; chosen_weight = w2; chosen_rho = r2
        elseif zeta <= bound1
            chosen_dmo = 1; chosen_weight = w1; chosen_rho = r1
        else
            chosen_dmo = 4; chosen_weight = w4; chosen_rho = r4
        end

        d_dm_on[itraj] = chosen_dmo
        d_Rho[itraj] = chosen_rho / chosen_weight

    else
        d_dm_on[itraj] = 4
        d_Rho[itraj] = one(Complex{T})
        is_adiab = true
    end
    return nothing
end

