module ED

using ..Positions
using ..SimLib: logmsg, geometry_from_density

using JLD2
using LinearAlgebra
using Printf: @sprintf
using XXZNumerics

# simplify type definitions
const _FARRAY{N} = Array{Float64, N} where N

## Data structure

# index order
# [(spin,) eigen_state, shot, h, rho]
struct EDData
    geometry::Symbol
    dim::Int64
    α::Float64
    ρs::_FARRAY{1}
    fields::_FARRAY{1}
    eev::_FARRAY{5} # eigenstate expectation value of x magnetization ⟨i|Sⁱₓ|i⟩
    eon::_FARRAY{4} # eigenstate occupation number |⟨i|ψ⟩|²
    evals::_FARRAY{4} # energy eigenvalues Eᵢ
end

function EDData(geometry, dim, α, ρs, shots, system_size, fields)
    hilbert_space_dim = 2^(system_size-1) # due to symmetry
    hs = length(fields)
    
    EDData(
        geometry,
        dim,
        α,
        sort(ρs),
        fields,
        _FARRAY{5}(undef, system_size, hilbert_space_dim, shots, hs, length(ρs)),
        _FARRAY{4}(undef, hilbert_space_dim, shots, hs, length(ρs)),
        _FARRAY{4}(undef, hilbert_space_dim, shots, hs, length(ρs)),
    )
end

EDData(posdata::PositionData, α, fields) = EDData(Positions.geometry(posdata), Positions.dimension(posdata), α, Positions.ρs(posdata), Positions.shots(posdata), Positions.system_size(posdata), fields)

system_size(eddata::EDData) = size(eddata.eev, 1)
hilbert_space_dimension(eddata::EDData) = size(eddata.eev, 2)
shots(eddata::EDData) = size(eddata.eev, 3)
ρ_values(eddata::EDData) = eddata.ρs
fields(eddata::EDData) = eddata.fields
eev(eddata::EDData) = eddata.eev
eon(eddata::EDData) = eddata.eon
evals(eddata::EDData) = eddata.evals
α(eddata::EDData) = eddata.α
geometry(eddata::EDData) = eddata.geometry
dimension(eddata::EDData) = eddata.dim

## Saving/Loading
ed_datapath(prefix, geometry, N, dim, α) = joinpath(prefix, "data", @sprintf("ed_%s_%id_alpha_%.1f_N_%02i.jld2", geometry, dim, α, N))
ed_datapath(prefix, eddata::EDData) = ed_datapath(prefix, geometry(eddata), system_size(eddata), dimension(eddata), α(eddata))

function save(prefix, eddata::EDData)
    path = ed_datapath(prefix, eddata)
    mkpath(dirname(path))
    JLD2.jldsave(path; eddata)
end

load(path) = JLD2.load(path)["eddata"]
load(prefix, geometry, N, dim, α) = load(ed_datapath(prefix, geometry, N, dim, α))

## main function
function _ensemble_J_mean(interaction, geometry, positions)
    mapreduce(pos->sum(interaction_matrix(interaction, geometry, pos)), +, eachslice(positions; dims=3))/size(positions,2)/size(positions,3)
end

function run_ed!(eddata::EDData, posdata::PositionData; scale_field=:ensemble)
    N = system_size(eddata)
    dim = Positions.dimension(posdata)
    ρs = ρ_values(eddata)
    field_values = fields(eddata)
    nshots = shots(eddata)
    interaction = PowerLaw(α(eddata))
    spin_ops = symmetrize_op.(op_list(σx/2, N))
    field_operator = sum(spin_ops)
    ψ0 = vec(symmetrize_state(foldl(⊗, ((up+down)/√2 for _ in 1:N))))

    logmsg("ToDo: rho=$ρs")
    logmsg("with $nshots realizations and  $(length(field_values)) field values")
    for (i, ρ) in enumerate(ρs)
        logmsg("rho = $ρ")
        geom = geometry_from_density(geometry(eddata), ρ, N, dim)
        ensemble_J_mean = 0
        if scale_field == :ensemble
            # compute the ensembles mean J
            ensemble_J_mean = _ensemble_J_mean(interaction, geom, data(posdata)[:,:,:,i])
            logmsg("Ensemble J mean for rho_$i=$ρ: $ensemble_J_mean")
        end
        Threads.@threads for shot in 1:nshots
            logmsg(@sprintf("%03i/%03i", shot, nshots))
            J = interaction_matrix(interaction, geom, data(posdata)[:,:,shot,i])
            model = symmetrize_op(xxzmodel(J, -0.73))
            normed_field_values = field_values
            if scale_field == :shot
                normed_field_values *= sum(J)/N
            elseif scale_field == :ensemble
                normed_field_values *= ensemble_J_mean
            end

            for (k, h) in enumerate(normed_field_values)
                H = model + h*field_operator
                evals, evecs = eigen!(Hermitian(Matrix(H)))
                for (l, evec) in enumerate(eachcol(evecs))
                    for (spin, op) in enumerate(spin_ops)
                        eev(eddata)[spin, l, shot, k, i] = real(dot(evec, op, evec)) # dot conjugates the first arg
                    end
                end
                eddata.evals[:,shot, k, i] = evals
                eddata.eon[:, shot, k, i] .= abs2.(evecs' * ψ0)
            end
        end
    end
    eddata
end

run_ed(posdata::PositionData, α, fields; scale_field=:ensemble) = run_ed!(EDData(posdata, α, fields), posdata; scale_field)

end #module