#Convergence for this module is the final stage - will confirm convergence of the final module - This involves the correct selectio nof all penalty parameters

module INS

using Gridap
using ForwardDiff
using LinearAlgebra
using Test
using GridapODEs.ODETools
using GridapODEs.TransientFETools
using Gridap.FESpaces: get_algebraic_operator
import Gridap: ∇
import GridapODEs.TransientFETools: ∂t
using LineSearches: BackTracking
using Gridap.Algebra: NewtonRaphsonSolver

@law conv(u, ∇u) = (∇u') ⋅ u
@law dconv(du, ∇du, u, ∇u) = conv(u, ∇du) + conv(du, ∇u)

# Physical constants
const u_max = 10 # 150 #cm/s
const L = 1 #cm
const ρ =  1.06e-3 #kg/cm^3 
const μ =  3.50e-5 #kg/cm.s
const ν = μ/ρ 
#const Δt =  0.046  # 0.046  #s \\
dt = 1
Δt=dt

const n_t= 10 #number of timesteps
const tF = n_t*dt

# Manufactured solutions
const θ = 1 
const k=2*pi

t0 = 0.0

h_=1/50 # average of conv meshs
@show  Re = ρ*u_max*L/μ
@show C_t = u_max*dt/h_

u(x,t) = u_max * VectorValue( cos(k*x[1])*sin(k*x[2]), -sin(k*x[1])*cos(k*x[2]) ) * (t/tF)
u(t::Real) = x -> u(x,t)
ud(t) = x -> u(t)(x)

p(x,t) = k* ( sin(k*x[1])-sin(k*x[2]) ) * (t/tF)
p(t::Real) = x -> p(x,t)
q(x) = t -> p(x,t)

f(t) = x ->  ∂t(u)(t)(x) +  conv(u(t)(x),∇(u(t))(x)) - ν * Δ(u(t))(x) + ∇(p(t))(x)
g(t) = x -> (∇⋅u(t))(x)

u_Γn(t) = u(t)
p_Γn(t) = p(t)

function run_test(n)

L=1
domain = (0,L,0,L)
partition = (n,n)
model = CartesianDiscreteModel(domain,partition)
h=L/n

order = 2

V0 = FESpace(
    reffe=:Lagrangian, order=order, valuetype=VectorValue{2,Float64},
    conformity=:H1, model=model, dirichlet_tags="boundary")

Q = TestFESpace(
    model=model,
    order=order-1,
    reffe=:Lagrangian,
    valuetype=Float64,
    conformity=:H1,
    constraint=:zeromean)

U = TransientTrialFESpace(V0,u)

P = TrialFESpace(Q)

trian = Triangulation(model)
degree = 2*order
quad = CellQuadrature(trian,degree)

a(u,v) = ν *inner(∇(u),∇(v))
b(v,t) = inner(v,f(t))
c_Ω(u, v) = v ⊙ conv(u, ∇(u))
dc_Ω(u, du, v) = v ⊙ dconv(du, ∇(du), u, ∇(u))

#SUPG STABILISATION 
#https://doi.org/10.1016/j.advengsoft.2018.02.004
α_τ = 0.1 #Tunable coefficiant (0,1)
@law τ_SUPG(u) = α_τ * ( (2/ Δt )^2 + ( 2 * norm( u.data,2 ) / h )^2 + 9 * ( 4*ν / h^2 )^2 )^(-0.5) #
#SUPG
sp_sΩ(w,p,v)    = τ_SUPG(w)     *  conv(w,∇(v)) ⋅ ∇(p)
st_sΩ(w,ut,v)   = τ_SUPG(w) * ρ *  conv(w,∇(v)) ⋅ ut
sc_sΩ(w,u,v)    = τ_SUPG(w) * ρ *  conv(w,∇(v)) ⋅ conv(u, ∇(u)) 
dsc_sΩ(w,u,du,v)= τ_SUPG(w) * ρ *  conv(w,∇(v)) ⋅ dconv(du, ∇(du), u, ∇(u)) 
ϕ_sΩ(w,v,t)     = τ_SUPG(w)     *  conv(w,∇(v)) ⋅ f(t)  

X = TransientMultiFieldFESpace([U,P])
Y = MultiFieldFESpace([V0,Q])

function res(t,x,xt,y)
  u,p = x
  ut,pt = xt
  v,q = y
  ( a(u,v) + inner(ut,v) - (∇⋅v)*p + q*(∇⋅u) - inner(v,f(t)) - q*g(t) + c_Ω(u, v) #+ 0.5 * (∇⋅u) * u ⊙ v
  - sp_sΩ(u,p,v) - st_sΩ(u,ut,v)  + ϕ_sΩ(u,v,t)    - sc_sΩ(u,u,v) )
end

function jac(t,x,xt,dx,y)
  u, p = x
  du,dp = dx
  v,q = y
  ( a(du,v)- (∇⋅v)*dp + q*(∇⋅du) + dc_Ω(u, du, v) #+ 0.5 * (∇⋅u) * du ⊙ v 
  - sp_sΩ(u,dp,v) - dsc_sΩ(u,u,du,v) )
end

function jac_t(t,x,xt,dxt,y)
  u, p = x   
  dut,dpt = dxt
  v,q = y
  ( inner(dut,v)
  - st_sΩ(u,dut,v) )
end

U0 = U(0.0)
P0 = P(0.0)
X0 = X(0.0)
uh0 = interpolate_everywhere(u(0.0),U0)
ph0 = interpolate_everywhere(p(0.0),P0)
xh0 = interpolate_everywhere([uh0,ph0],X0)

t_Ω = FETerm(res,jac,jac_t,trian,quad)
op = TransientFEOperator(X,Y,t_Ω)

ls = LUSolver()

nls = NLSolver(ls;show_trace=true, method=:newton, linesearch=BackTracking())
#nls = NewtonRaphsonSolver(ls,1e-10,40)
#nls = NewtonRaphsonSolver(ls,1e99,1)

#odes = ForwardEuler(ls,dt)
odes = ThetaMethod(nls,dt,θ)

solver = TransientFESolver(odes)
sol_t = solve(solver,op,xh0,t0,tF)

l2(w) = w⋅w

tol = 1.0e-6
_t_n = t0

result = Base.iterate(sol_t)

eul2=[]
epl2=[]

n_dt = 0
for (xh_tn, tn) in sol_t
    @show n_dt = n_dt + 1 
    _t_n += dt
    uh_tn = xh_tn[1]
    ph_tn = xh_tn[2]
    uh_Ω = restrict(uh_tn,trian)
    ph_Ω = restrict(ph_tn,trian)
    e = u(tn) - uh_Ω
    eul2i = sqrt(sum( integrate(l2(e),trian,quad) ))
    e = p(tn) - ph_Ω
    epl2i = sqrt(sum( integrate(l2(e),trian,quad) ))
    push!(eul2,eul2i)
    push!(epl2,epl2i)
  end

eul2=last(eul2)
epl2=last(epl2)

println(dt)

(eul2, epl2, h)

end

function conv_test(ns)

    eul2s = Float64[]
    epl2s = Float64[]
    hs = Float64[]

    for n in ns

        eul2, epl2, h = run_test(n)

        push!(eul2s,eul2)
        push!(epl2s,epl2)
        push!(hs,h)

    end

    (eul2s, epl2s,  hs)

end

ID = 2
ns = [32,64,96]

global ID = ID+1

eul2s, epl2s, hs = conv_test(ns);
using Plots
plot(hs,[eul2s, epl2s],
    xaxis=:log, yaxis=:log,
    label=["L2U" "L2P"],
    shape=:auto,
    xlabel="h",ylabel="L2 error norm",
    title = "INS_P1P1,Re=$(Re),C_t=$(C_t)")
savefig("INS_Space_ID_$(ID)")

end #module