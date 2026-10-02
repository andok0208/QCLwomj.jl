# --- output.jl ---
""" Adds output of new trajectory """
function outad!(state::SimulationState, out::SimulationOutput)

    out.econ_sosf .+= out.econ_sos
    out.Pad_sosf  .+= out.Pad_sos
    out.Pd_sosf   .+= out.Pd_sos

    if state.dia
        out.Pdf[:, 2] .+= out.Pd[:, 2]
        if state.nelc > 1
            out.Pdf[:, 1] .+= out.Pd[:, 1]
        end
    end

    if state.adia
        out.Padf[:, 2] .+= out.Pad[:, 2]
        out.Padf[:, 1] .+= out.Pad[:, 1]
        out.c2c1ref    .+= out.c2c1re
        out.c2c1imf    .+= out.c2c1im
        out.econf      .+= out.econ
        out.eoffs      .+= out.eoff
        if state.nelc > 1
            out.Pdf[:, 1] .+= out.Pd[:, 1]
        end
    end

    if state.vib
        out.qf .+= out.q_hist
        out.pf .+= out.p_hist
    end
end

""" output to files """
function output!(state::SimulationState, out::SimulationOutput)

    norm_traj = 1.0 / state.itraj

    norm_var = (state.itraj == 1) ? 1.0 : 1.0/(state.itraj - 1)

    #println("output: norm_traj = ", norm_traj)

    dt = state.deltat

    dir = "outdat/"

    # Output off-diagonal density matrix elements, integrated over all space
    open(dir * "c2c1re.dat", "w") do f_re
    open(dir * "c2c1im.dat", "w") do f_im
        for i in 1:state.ntime
            t = (i - 1) * dt
            write(f_re, "$t $(out.c2c1ref[i] * norm_traj)\n")
            write(f_im, "$t $(out.c2c1imf[i] * norm_traj)\n")
        end
    end
    end

    # Output diabatic population
    if state.dia
        open(dir * "econ.dat", "w") do f_econ
        open(dir * "p2tH.dat", "w") do f_p2t
            for i in 1:state.ntime
                t = (i - 1) * dt
                econf = out.econf[i] * norm_traj
                eoffs = out.eoffs[i] * norm_traj
                write(f_econ, "$t $econf $eoffs $(econf + eoffs)\n")
                p2t = out.Pdf[i, 2] * norm_traj
                write(f_p2t, "$t $p2t\n")
            end
        end
    end
    end

    # Output adiabatic populations
    if state.adia
        open(dir * "p2tadH.dat", "w") do f_p2ad
        open(dir * "p1tadH.dat", "w") do f_p1ad
            for i in 1:state.ntime
                t = (i - 1) * dt
                p2ad = out.Padf[i, 2] * norm_traj
                p1ad = out.Padf[i, 1] * norm_traj
                p2var = norm_var * out.Pad_sosf[i] - (norm_traj * state.itraj) * p2ad^2
                if p2var < 0
                    #println("Warning! negative p2var ($p2var) will be set to 0.0.")
                    p2var = 0.0
                else 
                    p2var = sqrt(p2var / state.itraj)
                end
                write(f_p2ad, "$t $p2ad $p2var\n")
                #write(f_p2ad, "$t $(1.0 - p1ad) $p2var\n")
                write(f_p1ad, "$t $p1ad\n")
            end
        end
        end
    end

    # Output expectation values of momenta and positions
    # j numbers the mode. In case of Spin-Boson model we have just one.
    if state.vib
        for j in 1:state.nmod
            open(dir * "qf_$j.dat", "w") do f_q
            open(dir * "pf_$j.dat", "w") do f_p
                for i in 1:state.ntime
                    t = (i - 1) * dt
                    write(f_q, "$t $(out.qf[i, j] * norm_traj)\n")
                    write(f_p, "$t $(out.pf[i, j] * norm_traj)\n")
                end
            end
            end
        end
    end
end

