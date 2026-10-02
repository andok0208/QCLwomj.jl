# --- calc_phase_gpu.jl ---
""" calculate phase factor """
@inline function calc_phase_gpu(itraj, dt, dmo, ctx)
    T = eltype(ctx.d_f)

    delta_val = delta_E_gpu((@view ctx.d_f[:, itraj]), ctx)
    v1 = Epot_ad_gpu((@view ctx.d_f[:, itraj]), delta_val, ctx, 1)
    v2 = Epot_ad_gpu((@view ctx.d_f[:, itraj]), delta_val, ctx, 2)
    dE21dt = (v2 - v1) * dt

    if dmo == 1 || dmo == 4  # (rho11, rho22)
        return one(Complex{T})
    elseif dmo == 3          # rho21
        #return exp(-im*dE21dt)
        return cis(-dE21dt)
    elseif dmo == 2          # rho12
        #return exp( im*dE21dt)
        return cis(dE21dt)
    else
        return one(Complex{T})
    end
end

