# --- smatrix_gpu.jl ---
# contains:
#   function smat_c_gpu
#   function todia_kernel!
#   function toadia_gpu!
#   function delta_E_gpu
#   function Epot_ad_gpu
""" S matrix elements in a tuple """
@inline function smat_c_gpu(f_val, delta, d_lambda, d_E, is_adiab, params)
    T = eltype(d_E)

    W = zero(T)
    if params.cmode
        lQ = f_val
        lQ = d_lambda[1, 1] * lQ
        if lQ >= zero(T)
            sgn_c = one(T)
        else
            sgn_c = -one(T)
        end
        W = sqrt(delta^2 + lQ^2)
    end
    if params.spinbo
        g = d_E[4]
        sgn_c = one(T)
        W = sqrt(delta^2 + g^2)
    end

    sqrtwo = sqrt(T(2))
    s1 = (one(T) / sqrtwo) * sqrt(one(T) + delta / W) * sgn_c  # S11
    s2 = (one(T) / sqrtwo) * sqrt(one(T) - delta / W)          # S12
    s3 = -s2                                                   # S21
    s4 = s1                                                    # S22

    if is_adiab
        s2, s3 = s3, s2
    end
    return (s1, s2, s3, s4)
end

""" convert Rho_W (electronic density matrix) from adiabatic to diabatic rep. """
function todia_kernel!(d_pdf_step1, d_pdf_step2, d_f, d_Rho, d_dm_on, d_E, d_lambda, d_kappa, 
                       nran, params)
    itraj = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if itraj > nran; return; end

    T = eltype(d_f)
    czero = zero(Complex{T})

    dmo = d_dm_on[itraj]
    rho_val = d_Rho[itraj]

    r1, r2, r3, r4 = czero, czero, czero, czero
    if dmo == 1
        r1 = rho_val
    elseif dmo == 2
        r2 = rho_val
    elseif dmo == 3
        r3 = rho_val
    elseif dmo == 4
        r4 = rho_val
    end

    f_val = d_f[2, itraj]
    delta = delta_E_gpu(d_f, itraj, d_E, d_kappa, params)
    tfad_val = true
    s1, s2, s3, s4 = smat_c_gpu(f_val, delta, d_lambda, d_E, tfad_val, params)

    r1_new = r1 * s1 * s1 + (r2 + r3) * s1 * s3 + r4 * s3 * s3
    r4_new = r1 * s2 * s2 + s2 * s4 * (r2 + r3) + r4 * s4 * s4

    d_pdf_step1[itraj] = real(r1_new)
    d_pdf_step2[itraj] = real(r4_new)

    return nothing
end

""" convert Rho_W (electronic density matrix) from diabatic to adiabatic rep. """
@inline function toadia_gpu!(
        d_f, itraj, r1, r2, r3, r4, is_adiab, d_E, d_lambda, d_kappa, params)

    if is_adiab  # already adiabatic
        return r1, r2, r3, r4, true
    end

    delta = delta_E_gpu(d_f, itraj, d_E, d_kappa, params)

    f_val = d_f[2, itraj]
    s1, s2, s3, s4 = smat_c_gpu(f_val, delta, d_lambda, d_E, is_adiab, params)

    # (S+) * (Rho) * (S) 
    new_r1 = r1*s1*s1 + (r2+r3)*s1*s3 + r4*s3*s3
    new_r2 = r1*s1*s2 + r2*s1*s4 + r3*s3*s2 + r4*s4*s3
    new_r3 = r1*s1*s2 + r2*s3*s2 + r3*s1*s4 + r4*s3*s4
    new_r4 = r1*s2*s2 + s2*s4*(r2+r3) + r4*s4*s4

    return new_r1, new_r2, new_r3, new_r4, true
end

""" diabatic energy difference """
@inline function delta_E_gpu(u, ctx)
    T = eltype(u)
    @inbounds begin
        dummy = T(0.5) * (ctx.d_E[3] - ctx.d_E[2])
        for i in (1 + ctx.params.nmodc):ctx.params.nmod
            kak = T(0.5) * (ctx.d_kappa[3, i] - ctx.d_kappa[2, i])
            dummy += kak * u[2 * i]
        end
    end
    return dummy
end

# multiple dispatch for use in toadia_gpu() and todia_kernel!()
@inline function delta_E_gpu(f, itraj, d_E, d_kappa, params)
    T = eltype(f)
    @inbounds begin
        dummy = T(0.5) * (d_E[3] - d_E[2])
        for i in (1 + params.nmodc):params.nmod
            kak = T(0.5) * (d_kappa[3, i] - d_kappa[2, i])
            dummy += kak * f[2 * i, itraj]
        end
    end
    return dummy
end

""" adiabatic potential energy """
@inline function Epot_ad_gpu(u, delta, ctx, istate)
    T = eltype(u)
    @inbounds begin
        dummy = zero(T)
        for i in 1:ctx.params.nmod
            m = 2 * i  # index for nuclear coordinate
            dummy += ctx.d_wf[3, i] * u[m]^2
        end

        dummy1 = zero(T)
        for i in (1 + ctx.params.nmodc):ctx.params.nmod
            m = 2 * i
            dummy1 += (ctx.d_kappa[3, i] + ctx.d_kappa[2, i]) * u[m]
        end

        qquad = T(0.5) * (ctx.d_E[3] + ctx.d_E[2]) + T(0.5) * (dummy + dummy1)

        dummy2 = delta^2
        if ctx.params.cmode
            lam = ctx.d_lambda[1, 1]
            dummy2 = delta^2 + (lam * u[2])^2
        elseif ctx.params.spinbo
            g = ctx.d_E[4]
            dummy2 = delta^2 + g^2
        end

        if istate == 1
            return qquad - sqrt(dummy2)
        elseif istate == 2
            return qquad + sqrt(dummy2)
        else
            return zero(T)  # for error handling
        end
    end
end

