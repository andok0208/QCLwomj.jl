# --- dmhop_gpu.jl ---
""" transition among density matrix elements """
@inline function dmhop_gpu!(itraj, ctx::GPUContext, zeta1, zeta2, t)
    T = eltype(ctx.d_f)

    # indent to match threads_spawn
        bfac = b_factor_gpu((@view ctx.d_f[:, itraj]), ctx)
        sign_dxd = sign(bfac)
        epsilon = abs(bfac) * ctx.params.deltat
        branch_weight = one(T) + 2epsilon
        mc_norm = one(T) / branch_weight

        if t >= ctx.params.t_offset
            if zeta1 <= mc_norm  # stay
                dmo_new = ctx.d_dm_on[itraj]
                ctx.d_Rho[itraj] *= branch_weight
            else  # transition
                dmo = ctx.d_dm_on[itraj]
                if dmo == 1
                    if zeta2 <= T(0.5)
                        dmo_new = 2
                        Rho_new = sign_dxd * ctx.d_Rho[itraj]
                    else
                        dmo_new = 3
                        Rho_new = sign_dxd * ctx.d_Rho[itraj]
                    end
                elseif dmo == 2
                    if zeta2 <= T(0.5)
                        dmo_new = 1
                        Rho_new = -sign_dxd * ctx.d_Rho[itraj]
                    else
                        dmo_new = 4
                        Rho_new = sign_dxd * ctx.d_Rho[itraj]
                    end
                elseif dmo == 3
                    if zeta2 <= T(0.5)
                        dmo_new = 4
                        Rho_new = sign_dxd * ctx.d_Rho[itraj]
                    else
                        dmo_new = 1
                        Rho_new = -sign_dxd * ctx.d_Rho[itraj]
                    end
                elseif dmo == 4
                    if zeta2 <= T(0.5)
                        dmo_new = 3
                        Rho_new = -sign_dxd * ctx.d_Rho[itraj]
                    else
                        dmo_new = 2
                        Rho_new = -sign_dxd * ctx.d_Rho[itraj]
                    end
                end
                ctx.d_Rho[itraj] = Rho_new * branch_weight
            end
        else  # t < t_offset
            dmo_new = ctx.d_dm_on[itraj]
        end
        ctx.d_dm_on[itraj] = dmo_new
end

