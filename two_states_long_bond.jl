using Parameters
using PyPlot

const _TOL = 10^(-13)


struct Alloc
    v::Array{Float64, 1}
    q::Array{Float64, 1}
    repay_prob::Array{Float64, 1}
    b_pol_i::Array{Int64, 1}
end 

Alloc(gridlen::Int64) = Alloc(
        fill(NaN, gridlen),
        fill(NaN, gridlen),
        fill(NaN, gridlen),
        zeros(Int64, gridlen)
    )


function dropnames(namedtuple::NamedTuple, set)
    out = [p for p in pairs(namedtuple) if !(p[1] in set)]
    return (; out...)
end


function get_q̲(;r, δ, λ)
    return (r + δ) / ((r + λ) / (1 - λ) + δ)
end


function get_bS_low(;y, u_inv, β, r, vH)
    return (y - u_inv((1 - β) * vH)) / r
end


function get_bB_high(;y, u_inv, β, r, λ, δ, vH, vL)
    return ((y - u_inv((1 - β) * vL - 
        β * λ * (vH - vL))) / 
     (r + δ * (1 - get_q̲(r=r, δ=δ, λ=λ))))
end


function create_b_grid(;bmin=0.0, bmax, bsaving, npoints_approx)
    b_grid = collect(range(bmin, stop=bmax, length=npoints_approx))
    if !(bsaving in b_grid) 
        loc = findfirst(x -> (x >= bsaving), b_grid)
        insert!(b_grid, loc, bsaving)
    end
    return b_grid
end


TwoStatesPar = @with_kw (
    R = exp(0.06),
    β = exp(-0.1), 
    τₕ = 0.15,   # high cost of default
    τₗ = 0.08,   # low cost of default
    λ = 0.025,  # proba of low cost
    u = (x -> log(x)), 
    u_inv = (x -> exp(x)), 
    δ = 0.25, 
    y = 1.0, 
    npoints_approx = 1000,
    r = R - 1.0,
    vH = u((1 - τₗ) * y) / (1 - β),
    vL = u((1 - τₕ) * y) / (1 - β),
    q̅ = 1.0, 
    q̲ = get_q̲(r=r, δ=δ, λ=λ),
    bS_low = get_bS_low(
        y=y, u_inv=u_inv, β=β, r=r, vH=vH
    ),
    bB_high = get_bB_high(
        y=y, u_inv=u_inv, β=β, r=r, λ=λ, δ=δ,
        vH=vH, vL=vL
    ),
    b_grid = create_b_grid(
        bmin=0.0, bmax=bB_high, bsaving=bS_low, npoints_approx=npoints_approx
    ), 
    gridlen = length(b_grid), 
    bS_low_loc = findfirst(x -> (x == bS_low), b_grid)
)


function q_ss(model, repay_prob)
    @unpack R, r, δ = model
    q_ss = repay_prob * (r + δ) / (R -  repay_prob * (1 - δ)) 
    return q_ss
end


function v_ss(model, b, q, repay_prob)
    @unpack y, β, R, r, δ, u, vH = model
    λ1 = 1 - repay_prob
    return (u(y - (r + δ *(1 - q)) * b) +
            β * λ1 * vH) / (1 - β * (1 - λ1))
end


function c_budget(model; b, b_prime, q)
    @unpack y, δ, r = model
    return y - (r + δ) * b + q * (b_prime - (1 - δ) * b)
end


function v_bellman(model; c, v_prime)
    @unpack u, β, λ, vH, vL = model 
    return u(c) + β * λ * max(v_prime, vH) + 
        β * (1 - λ) * max(v_prime, vL)
end


function q_bellman(model; repay_prob, q_prime)
    @unpack R, r, δ = model
    return repay_prob * ((r + δ) + (1 - δ) * q_prime) / R
end 


function get_repay_prob(model, v)
    @unpack vH, vL, λ = model
    h = if (v >= vH - _TOL) 1.0 else 0.0 end 
    l = if (v >= vL - _TOL) 1.0 else 0.0 end
    return λ * h + (1 - λ) * l
end


function construct_path(model, loc, v_at_loc, Δ, alloc)

    function assign_values(iter, v, q, repay_prob, b_prime)
        # auxiliary assign function 
        alloc.v[iter] = v
        alloc.q[iter] = q
        alloc.repay_prob[iter] = repay_prob
        alloc.b_pol_i[iter] = b_prime
    end

    @unpack b_grid, λ, vL, vH, gridlen = model
    # Δ = 1 is the saving path 
    # Δ = -1 is the borrowing path 
    @assert Δ in [-1, 1]

    iter = loc
    v_max = 0.0
    new_pol = 1 
    prev_pol = 1
    q_prime = 1.0
    valid_until = if (Δ == 1) gridlen else 1 end

    while true
        if iter == loc
            # starting point is stationary
            repay_prob = get_repay_prob(model, v_at_loc)
            assign_values(
                iter, v_at_loc, q_ss(model, repay_prob), repay_prob, iter
            )
            prev_pol = iter
        else
            b = b_grid[iter]
            first_valid = true
            for i in range(prev_pol, step=Δ, stop=iter-Δ)
                b_prime = b_grid[i]
                v_prime = alloc.v[i]
                q = alloc.q[i]
                repay_prob = alloc.repay_prob[i]
                c = c_budget(model, b=b, b_prime=b_prime, q=q)
                if c > 0
                    v_tmp = v_bellman(model; c=c, v_prime=v_prime)
                    if (first_valid) || (v_tmp > v_max)
                        v_max = v_tmp
                        new_pol = i
                        q_prime = q
                    end
                    first_valid = false
                end
            end
            if first_valid
                # nothing is optimal 
                assign_values(iter, NaN, NaN, NaN, -1)
                valid_until = iter - Δ
                break
            else
                repay_prob = get_repay_prob(model, v_max)
                q = q_bellman(
                    model; 
                    repay_prob=repay_prob, 
                    q_prime=q_prime
                )
                assign_values(iter, v_max,  q, repay_prob, new_pol)
                prev_pol = new_pol

                # compute the value of staying put
                v_stay_put = v_bellman(
                    model; 
                    c=c_budget(model; b=b, b_prime=b, q=q), 
                    v_prime=v_max
                )
                if v_stay_put >  v_max 
                    # staying put is better than following prescription. 
                    valid_until = iter - Δ
                    break
                end
            end
        end
        iter += Δ
        (Δ == 1) && (iter > gridlen) && break
        (Δ == -1) && (iter < 1) && break
    end
    return (loc=loc, alloc=alloc, valid_until=valid_until)
end


function construct_sav_path(
        model, 
        loc, 
        v_at_loc; 
        alloc=Alloc(model.gridlen)
    )
    construct_path(model, loc, v_at_loc, 1, alloc)
end 


function construct_bor_path(
        model, 
        loc, 
        v_at_loc; 
        alloc=Alloc(model.gridlen)
    )
    construct_path(model, loc, v_at_loc, -1, alloc)
end 


function create_sav_eqm(model)
    @unpack bS_low_loc, vH, vL, gridlen = model

    safe_region = construct_bor_path(model, bS_low_loc, vH)
    @assert safe_region.valid_until == 1
    
    crisis_saving = construct_sav_path(model, bS_low_loc, vH)
    crisis_borrowing = construct_bor_path(model, gridlen, vL)
    @assert crisis_saving.valid_until >= crisis_borrowing.valid_until
    @assert crisis_borrowing.alloc.v[bS_low_loc] <= vH

    i = bS_low_loc 
    index = gridlen + 1
    while true 
        if crisis_saving.alloc.v[i] < crisis_borrowing.alloc.v[i]
            index = i 
            break
        end 
        (i >= gridlen) && break 
        i += 1
    end 
    # creating the merged arrays with the sav eqm allocation
    return Alloc(
        vcat(
            safe_region.alloc.v[1:bS_low_loc],
            crisis_saving.alloc.v[bS_low_loc + 1:index - 1],
            crisis_borrowing.alloc.v[index:end]
        ),
        vcat(
            safe_region.alloc.q[1:bS_low_loc],
            crisis_saving.alloc.q[bS_low_loc + 1:index - 1],
            crisis_borrowing.alloc.q[index:end]
        ),
        vcat(
            safe_region.alloc.repay_prob[1:bS_low_loc],
            crisis_saving.alloc.repay_prob[bS_low_loc + 1:index - 1],
            crisis_borrowing.alloc.repay_prob[index:end]
        ),
        vcat(
            safe_region.alloc.b_pol_i[1:bS_low_loc],
            crisis_saving.alloc.b_pol_i[bS_low_loc + 1:index - 1],
            crisis_borrowing.alloc.b_pol_i[index:end]
        )
    )
end


function create_bor_eqm(model)
    @unpack vL, gridlen = model

    bor_eqm = construct_bor_path(model, gridlen, vL)
    @assert bor_eqm.valid_until == 1 

    return bor_eqm.alloc
end


function create_hyb_eqm(model)
    @unpack vL, gridlen, bS_low_loc, u, r, β, b_grid, y = model 
    bor = construct_bor_path(model, gridlen, vL)

    loc = bS_low_loc
    @assert  u(y - r * b_grid[loc]) / (1 - β) >= bor.alloc.v[loc]

    hybrid_loc = 0
    while true
        v = u(y - r * b_grid[loc]) / (1 - β)
        if v < bor.alloc.v[loc]
            hybrid_loc = loc
            break
        end
        (loc <= bor.valid_until) && break
        loc+=-1
    end
    @assert 0 < hybrid_loc < bS_low_loc

    safe = construct_bor_path(
        model, 
        hybrid_loc, 
        u(y - r * b_grid[hybrid_loc]) / (1 - β)
    )

    return Alloc(
        vcat(
            safe.alloc.v[1:hybrid_loc],
            bor.alloc.v[hybrid_loc + 1:end]
        ),
        vcat(
            safe.alloc.q[1:hybrid_loc],
            bor.alloc.q[hybrid_loc + 1:end]
        ),
        vcat(
            safe.alloc.repay_prob[1:hybrid_loc],
            bor.alloc.repay_prob[hybrid_loc + 1:end]
        ),
        vcat(
            safe.alloc.b_pol_i[1:hybrid_loc],
            bor.alloc.b_pol_i[hybrid_loc + 1:end]
        )
    )
end 


function iterate_v!(alloc_new, model, alloc)
    # this could be improved using the divide and conquer algorithm 
    @unpack gridlen, b_grid = model
    v_tmp = 0.0
    v_max = 0.0
    b_pol = 0
    prev_pol = 1
    for i in 1:gridlen
        first_valid = true
        v_max = NaN
        b_pol = 0
        for j in prev_pol:gridlen
            c = c_budget(model; b=b_grid[i], b_prime=b_grid[j], q=alloc.q[j])
            if c >= 0
                v_tmp = v_bellman(model; c=c, v_prime=alloc.v[j])
                if (first_valid) || (v_tmp > v_max)
                    v_max = v_tmp
                    b_pol = j
                    first_valid = false
                end
            end
        end
        alloc_new.v[i] = v_max
        alloc_new.b_pol_i[i] = b_pol
        prev_pol = max(b_pol, 1)
    end
end


function iterate_q!(alloc_new, model, alloc)
    @unpack gridlen = model
    for i in 1:gridlen
        repay_prob = get_repay_prob(model, alloc.v[i])
        alloc_new.q[i] = q_bellman(
            model; 
            repay_prob=repay_prob, 
            q_prime=alloc.q[alloc.b_pol_i[i]]
        )
    end
end