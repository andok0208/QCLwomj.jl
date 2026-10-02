# --- symplct4_gpu.jl ---
""" 4th order symplectic propagator """
const cr2 = 2^(1/3)
const s3 = 1 / (2 - cr2)
const symplct4_k = (s3/2,  (1-s3)/2, (1-s3)/2, s3/2)
const symplct4_u = (0.0, s3, 1-2*s3, s3)

@inline function symplct4_step_gpu!(itraj, d_f, dt, current_dmo, ctx, ::Val{NEQ}) where {NEQ}
    T = eltype(d_f)
    u = MVector{NEQ, T}(ntuple(idx -> d_f[idx, itraj], Val(NEQ)))

    @inbounds begin
    for r in 1:4
        du = derivpx_gpu(u, itraj, current_dmo, ctx)
        u_r = T(symplct4_u[r])
        k_r = T(symplct4_k[r])
        for i in 1:ctx.params.nmod
            k_p = 2i - 1  # index for p
            k_x = 2i      # index for x
            u[k_p] += u_r * du[k_p] * dt
            u[k_x] += k_r * du[k_x] * dt
        end
    end

    for idx in 1:NEQ
        d_f[idx, itraj] = u[idx]
    end
    end  # inbounds
    return nothing
end

