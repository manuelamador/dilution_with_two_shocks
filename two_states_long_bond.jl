using Parameters
using PyPlot

const _TOL = 10.0^(-12)
const _MAX_ITERS = 10000 

########################################################################
#
# Structs and constructors
# 
#


# `TwoStatesModel` contains the basic parameters of the model as well 
# as some other values and the debt grid. 
@with_kw struct TwoStatesModel{F1, F2, F3} @deftype Float64
    R = 1.05 
    β = 0.91 
    τH = 0.15
    τL = 0.08
    λ = 0.025
    u::F1 = (x -> log(x))
    u_inv::F2 = (x -> exp(x))
    δ = 0.25
    y = 1.0
    npoints_approx::Int64 = 10000
    # Generated values 
    r = R - 1.0
    vH = u((1 - τL) * y) / (1 - β)
    vL = u((1 - τH) * y) / (1 - β)
    q̅ = 1.0
    q̲ = get_q̲(r=r, δ=δ, λ=λ)
    bS_low = get_bS_low(y=y, u_inv=u_inv, β=β, r=r, vH=vH)
    bB_high = get_bB_high(
        y=y, u_inv=u_inv, β=β, r=r, λ=λ, δ=δ, vH=vH, vL=vL
    )
    b_grid::Array{Float64, 1} = create_b_grid(
        bmin=0.0, bmax=bB_high, bsaving=bS_low, npoints_approx=npoints_approx
    )
    gridlen::Int64 = length(b_grid)
    bS_low_loc::Int64 = findfirst(x -> (x == bS_low), b_grid)
    d_and_c_fun::F3 = get_d_and_c_fun(gridlen)
end


function Base.show(io::IO, model::TwoStatesModel)
    @unpack R, β, τH, τL, λ, δ, y, gridlen = model
    print(
        io, "R=", R, " β=", β, " τH=", τH, " τL=", τL, 
        " λ=", λ, " δ=", δ, " y=", y, " points=", gridlen
    )    
end


# `Alloc` stores an allocation together with a reference to the model that 
# generated. 
struct Alloc{P}
    v::Array{Float64, 1}
    q::Array{Float64, 1}
    repay_prob::Array{Float64, 1}
    b_pol_i::Array{Int64, 1}
    model::P    # reference to the parent model
end 


Alloc(model::TwoStatesModel) = Alloc(
    fill(NaN, model.gridlen),
    fill(NaN, model.gridlen),
    fill(NaN, model.gridlen),
    zeros(Int64, model.gridlen),
    model
)

function Base.show(io::IO, alloc::Alloc)
    @unpack R, β, τH, τL, λ, δ, y, gridlen = alloc.model
    print(io, "Alloc for model: ")
    show(io, alloc.model)   
end

#
# Functions related to the constructor of TwoStateModel
#

function get_d_and_c_fun(gridlen::Int64)

    """
    Generate a bisection tree from array to be used in the "divide and conquer"
    algorithm. 

    Returns a tuple of two arrays. First element is the list of the elements in
    array, excluding the extrema. Second element is an array of tupples with the
    each of the parents.
    """
    function create_tree(array)

        #    Auxiliary function
        function create_subtree!(
            tree_list, 
            parents_list, # modified in place
            array
        )
            length = size(array)[1]
            if length == 2
                return
            else
                parents = (array[1], array[end])
                halflen = (length + 1)÷2
                push!(tree_list, array[halflen])
                push!(parents_list, parents)
                create_subtree!(
                    tree_list, parents_list, @view array[1:halflen]
                )
                create_subtree!(
                    tree_list, parents_list, @view array[halflen:end]
                )
            end 
        end

        tree_list = eltype(array)[]
        parents_list = Tuple{eltype(array), eltype(array)}[]
        create_subtree!(tree_list, parents_list, array)
        return (tree_list, parents_list)
    end

    # Given an index i and a tree returns the current index in the three as well 
    # as the indices of the parents.
    function d_and_c_index_bounds(i, pol_vector, tree)
        if i == 1 
            b_i = 1
            left_bound = 1
            right_bound = gridlen
        elseif i == 2
            b_i = gridlen 
            left_bound = pol_vector[1]
            right_bound = gridlen
        else
            index = i - 2
            b_i = tree[1][index]
            left_bound = pol_vector[tree[2][index][1]]
            right_bound = pol_vector[tree[2][index][2]]
        end 
        return (b_i, left_bound, right_bound)
    end

    tree = create_tree(collect(1:gridlen))
    return ((x, pol) -> d_and_c_index_bounds(x, pol, tree))
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

########################################################################
#
# Helper functions on prices, values, repayments and consumption
#

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


########################################################################
#
# Functions that create the saving and borrowing paths
#


# Generic function that construct a path starting at `loc` with 
# value `v_at_loc` in direction `Δ` and stores the solution in alloc 
# as well as returning a tuple. 
#
# `Δ=-1` corresponds to a borrowing path, and `Δ=1` corresponds to
# a saving path 
function construct_path!(alloc, model, loc, v_at_loc, Δ)

    function assign_values!(alloc, iter, v, q, repay_prob, b_prime)
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
            q =  q_ss(model, repay_prob)
            assign_values!(alloc, iter, v_at_loc, q, repay_prob, iter)
            prev_pol = iter
        else
            b = b_grid[iter]
            first_valid = true
            for i in range(prev_pol, step=Δ, stop=iter-Δ)
                b_prime = b_grid[i]
                v_prime = alloc.v[i]
                q = alloc.q[i]
                repay_prob = alloc.repay_prob[i]
                c = c_budget(model; b=b, b_prime=b_prime, q=q)
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
                # nothing is optimal -- this debt level is not feasible
                assign_values!(alloc, iter, NaN, NaN, NaN, -1)
                valid_until = iter - Δ
                break
            else
                repay_prob = get_repay_prob(model, v_max)
                q = q_bellman(
                    model; 
                    repay_prob=repay_prob, q_prime=q_prime
                )
                assign_values!(alloc, iter, v_max,  q, repay_prob, new_pol)
                prev_pol = new_pol
                # compute the value of staying put
                c = c_budget(model; b=b, b_prime=b, q=q)
                v_stay_put = v_bellman(model; c=c, v_prime=v_max)
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
    model, loc, v_at_loc; 
    alloc=Alloc(model)
)
    construct_path!(alloc, model, loc, v_at_loc, 1)
end 


function construct_bor_path(
    model, loc, v_at_loc; 
    alloc=Alloc(model)
)
    construct_path!(alloc, model, loc, v_at_loc, -1)
end 


# Return a saving equilibrium allocation. Throws error if it can't. 
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
    @views return Alloc(
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
        ),
        model
    )
end


# Returns a borrowing equilibrium allocation. Throws error if it can't. 
function create_bor_eqm(model)
    bor_eqm = construct_bor_path(model, model.gridlen, model.vL)
    @assert bor_eqm.valid_until == 1 
    return bor_eqm.alloc
end


# Returns a hybrid equilibrium allocation. Throws error if it can't. 
function create_hyb_eqm(model)
    @unpack vL, vH, gridlen, bS_low_loc, u, r, β, b_grid, y = model 
    bor = construct_bor_path(model, gridlen, vL)
    loc = findfirst(x -> x < vH, bor.alloc.v) - 1
    println(loc, " ", bS_low_loc)
    @assert  u(y - r * b_grid[loc]) / (1 - β) >= bor.alloc.v[loc]
    # NOTE : this construction may not work when the hybrid is the only
    # equilibrium, as the above condition will fail. 
    hybrid_loc = 0
    while true
        v = u(y - r * b_grid[loc]) / (1 - β)
        if v < bor.alloc.v[loc]
            hybrid_loc = loc + 1 # not sure whether we should add one or not.
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
    @views begin
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
            ),
            model
        )
    end   
end 



# Returns a hybrid equilibrium allocation. Throws error if it can't. 
function create_hyb_eqm(model)
    @unpack vL, vH, gridlen, bS_low_loc, u, r, β, b_grid, y = model 
    bor = construct_bor_path(model, gridlen, vL)
    loc = findfirst(x -> x < vH, bor.alloc.v) - 1
    start_sign = sign(u(y - r * b_grid[loc]) / (1 - β) - bor.alloc.v[loc])
    # get the sign at the first point in the safe region for the borroing eqm
    hybrid_loc = 0
    while true
        v = u(y - r * b_grid[loc]) / (1 - β)
        if sign(v - bor.alloc.v[loc]) != start_sign
            # we switched signs -- so the borrowing equilibrium 
            # and the stationary value at risk free prices have crossed. 
            # This is a potential location for the hybrid stationary point
            hybrid_loc = loc + ((start_sign == 1) ? 1 : 0) 
                    # If we started with the borrowing above, then add one.
                    # This seems to work numerically (sometimes).
            break
        end
        (loc <= bor.valid_until) && break
        loc+=-1
    end
    @assert 0 < hybrid_loc # check that we did found a hybrid alloc
    safe = construct_bor_path(
        model, 
        hybrid_loc, 
        u(y - r * b_grid[hybrid_loc]) / (1 - β)
    )
    @views begin
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
            ),
            model
        )
    end   
end 


########################################################################
#
#  Functions for value and price iteration 
#


# Uses divide and conquer to iterate the value and policy functions. 
function iterate_v_and_pol!(alloc_new, alloc)
    # Iterate the value function and update the 
    @unpack b_grid, d_and_c_fun = alloc.model
    v_tmp = 0.0
    v_max = 0.0
    b_pol = 0
    prev_pol = 1
    for iter in eachindex(b_grid)
        first_valid = true
        v_max = NaN
        i, left_bound, right_bound = d_and_c_fun(iter, alloc_new.b_pol_i)
        b_pol = i # if not feasible -- the policy is the current state
                  # as all higher states are also not feasible 
        for j in left_bound:right_bound
            c = c_budget(
                alloc.model; 
                b=b_grid[i], b_prime=b_grid[j], q=alloc.q[j]
            )
            if c >= 0
                v_tmp = v_bellman(alloc.model; c=c, v_prime=alloc.v[j])
                if (first_valid) || (v_tmp > v_max)
                    v_max = v_tmp
                    b_pol = j
                    first_valid = false
                end
            end
        end
        alloc_new.v[i] = v_max
        alloc_new.repay_prob[i] = get_repay_prob(alloc.model, v_max)
        alloc_new.b_pol_i[i] = b_pol
    end
end


# Uses the values and policies in `alloc_new` together with 
# the price in `alloc` to update the price in `alloc_new`.
function iterate_q!(alloc_new, alloc)
    for i in eachindex(alloc.model.b_grid)
        alloc_new.q[i] = q_bellman(
            alloc.model; 
            repay_prob=alloc_new.repay_prob[i], 
            q_prime=alloc.q[alloc_new.b_pol_i[i]]
        )
    end
end


# Helper distance function 
function distance(new, old)
    error = 0.0
    for i in eachindex(new.v)
        error = max(error, abs(new.v[i] - old.v[i]) + abs(new.q[i] - old.q[i]))
    end 
    return error
end


# Iterate value and price starting from values of zero. This should 
# correspond to iterating backwards from a final finite T.
function iterate_backwards(model; tol=10.0^(-12), max_iters=_MAX_ITERS)
    a_new = Alloc(model)
    a_old = Alloc(
        zero(model.b_grid),
        zero(model.b_grid),
        zero(model.b_grid),
        ones(Int64, model.gridlen),
        model
    )
    counter = 1
    while true
        iterate_v_and_pol!(a_new, a_old)
        iterate_q!(a_new, a_old)
        error = distance(a_new, a_old)
        if  error < tol
            println("Converged.")
            break
        end
        counter+=1
        if mod(counter, 10) == 0 
            println("Iter ", counter, ". Distance=", error) 
        end 
        if counter > max_iters 
            println("Did not converged. Distance=", error)
            break
        end
        a_new, a_old = a_old, a_new
    end
    return a_new
end 

########################################################################
#
# Plotting functions 
#

function plot_pol(alloc; new_figure=true)
    if new_figure
        figure()
    end
    plot(alloc.model.b_grid, alloc.model.b_grid[alloc.b_pol_i])
    plot(
        alloc.model.b_grid[alloc.b_pol_i], 
        alloc.model.b_grid[alloc.b_pol_i], 
        "--"
    )
    loc= findfirst(x -> x < alloc.model.vH, alloc.v) - 1
    axvline(alloc.model.b_grid[loc]; lw=1, color="gray")
end