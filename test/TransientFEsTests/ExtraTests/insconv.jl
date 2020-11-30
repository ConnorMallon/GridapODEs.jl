
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
@law dconv(du, ∇du, u, ∇u) = conv(u, ∇du) #+ conv(du, ∇u)

# Physical constants
const u_max = 150 # 150 #cm/s
const L = 1 #cm
const ρ =  1.06e-3 #kg/cm^3 
const μ =  3.50e-5 #kg/cm.s
const ν = μ/ρ 
#const Δt =  0.046  # 0.046  #s \\
dt = 0.1/u_max
Δt=dt

const n_t= 5 #number of timesteps
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

f(t) = x -> ρ * ∂t(u)(t)(x) - μ * Δ(u(t))(x) + ∇(p(t))(x) + ρ * conv(u(t)(x),∇(u(t))(x))
g(t) = x -> (∇⋅u(t))(x)

u_Γn(t) = u(t)
p_Γn(t) = p(t)

function run_test(n)

L=1
domain = (0,L,0,L)
partition = (n,n)
model = simplexify( CartesianDiscreteModel(domain,partition) )
h=L/n

order = 1

V0 = FESpace(
    reffe=:PLagrangian, order=order, valuetype=VectorValue{2,Float64},
    conformity=:H1, model=model, dirichlet_tags="boundary")

Q = TestFESpace(
    model=model,
    order=order,
    reffe=:PLagrangian,
    valuetype=Float64,
    conformity=:H1,
    constraint=:zeromean)

U = TransientTrialFESpace(V0,u)

P = TrialFESpace(Q)

X = TransientMultiFieldFESpace([U,P])
Y = MultiFieldFESpace([V0,Q])

trian = Triangulation(model)
degree = 2*order
quad = CellQuadrature(trian,degree)

## Weak form terms
#Interior terms
m_Ω(ut,v) = ρ * ut⊙v
a_Ω(u,v) = μ * ∇(u)⊙∇(v) 
b_Ω(v,p) = - (∇⋅v)*p
c_Ω(u, v) = ρ *  v ⊙ conv(u, ∇(u))
dc_Ω(u, du, v) = ρ * v ⊙ dconv(du, ∇(du), u, ∇(u))

#STABILISATION
α_τ = 0.1 #Tunable coefficiant (0,1)
@law τ_SUPG(u) = α_τ * ( (2/ Δt )^2 + ( 2 * norm( u.data,2 ) / h )^2 + 9 * ( 4*ν / h^2 )^2 )^(-0.5) # SUPG Stabilisation - convection stab ( τ_SUPG(u )
@law τ_PSPG(u) = τ_SUPG(u) # PSPG stabilisation - inf-sup stab  ( ρ^-1 * τ_PSPG(u) )

#PSPG 
sp_Ω(w,p,q)    = (ρ^(-1) * τ_PSPG(w))     *  ∇(q) ⋅ ∇(p)
st_Ω(w,ut,q)   = (ρ^(-1) * τ_PSPG(w)) * ρ *  ∇(q) ⋅ ut
sc_Ω(w,u,q)    = (ρ^(-1) * τ_PSPG(w)) * ρ *  ∇(q) ⋅ conv(u, ∇(u))
dsc_Ω(w,u,du,q)= (ρ^(-1) * τ_PSPG(w)) * ρ *  ∇(q) ⋅ dconv(du, ∇(du), u, ∇(u))
ϕ_Ω(w,q,t)     = (ρ^(-1) * τ_PSPG(w))     *  ∇(q) ⋅ f(t)

#SUPG
sp_sΩ(w,p,v)    = τ_SUPG(w)     *  conv(w,∇(v)) ⋅ ∇(p)
st_sΩ(w,ut,v)   = τ_SUPG(w) * ρ *  conv(w,∇(v)) ⋅ ut
sc_sΩ(w,u,v)    = τ_SUPG(w) * ρ *  conv(w,∇(v)) ⋅ conv(u, ∇(u)) 
dsc_sΩ(w,u,du,v)= τ_SUPG(w) * ρ *  conv(w,∇(v)) ⋅ dconv(du, ∇(du), u, ∇(u)) 
ϕ_sΩ(w,v,t)     = τ_SUPG(w)     *  conv(w,∇(v)) ⋅ f(t)  

#NITSCHE
α_γ = 35
@law γ(u) =  α_γ * ( ν / h  +  ρ * maximum(u) / 6 ) # Nitsche Penalty parameter ( γ / h ) 

#Boundary terms 
a_Γ(u,v) = μ* ( - (n_Γ⋅∇(u))⋅v - u⋅(n_Γ⋅∇(v)) ) + ( γ(u)/h )*u⋅v 
b_Γ(v,p) = (n_Γ⋅v)*p

#Interior term collection
function res_Ω(t,x,xt,y)
  u,p = x
  ut,pt = xt
  v,q = y
  ( m_Ω(ut,v) + a_Ω(u,v) + b_Ω(v,p) + b_Ω(u,q) - v⋅f(t) + q*g(t) + c_Ω(u,v)  # + ρ * 0.5 * (∇⋅u) * u ⊙ v  
  - sp_Ω(u,p,q)  -  st_Ω(u,ut,q)   + ϕ_Ω(u,q,t)     - sc_Ω(u,u,q) 
  - sp_sΩ(u,p,v) -  st_sΩ(u,ut,v)  + ϕ_sΩ(u,v,t)    - sc_sΩ(u,u,v) )
end

function jac_Ω(t,x,xt,dx,y)
  u, p = x
  du,dp = dx
  v,q = y
  ( a_Ω(du,v) + b_Ω(v,dp) + b_Ω(du,q)  + dc_Ω(u, du, v) # + ρ * 0.5 * (∇⋅u) * du ⊙ v 
  - sp_Ω(u,dp,q)  - dsc_Ω(u,u,du,q) 
  - sp_sΩ(u,dp,v) - dsc_sΩ(u,u,du,v) )
end

function jac_tΩ(t,x,xt,dxt,y)
  u,p = x 
  dut,dpt = dxt
  v,q = y
  ( m_Ω(dut,v) 
  - st_Ω(u,dut,q) 
  - st_sΩ(u,dut,v) )
end

#Boundary term collection
function res_Γ(t,x,xt,y)
  u,p = x
  ut,pt = xt
  v,q = y
  a_Γ(u,v)+b_Γ(u,q)+b_Γ(v,p) - ud(t) ⊙(  ( γ(u)/h )*v - μ * n_Γ⋅∇(v) + q*n_Γ )
end

function jac_Γ(t,x,xt,dx,y)
  du,dp = dx
  v,q = y
  a_Γ(du,v)+b_Γ(du,q)+b_Γ(v,dp)
end

function jac_tΓ(t,x,xt,dxt,y)
  dut,dpt = dxt
  v,q = y
  0*m_Ω(dut,v)
end

U0 = U(0.0)
P0 = P(0.0)
X0 = X(0.0)
uh0 = interpolate_everywhere(u(0.0),U0)
ph0 = interpolate_everywhere(p(0.0),P0)
xh0 = interpolate_everywhere([uh0,ph0],X0)

t_Ω = FETerm(res_Ω,jac_Ω,jac_tΩ,trian,quad)
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
    e = u(tn) - uh_tn
    eul2i = sqrt(sum( integrate(l2(e),trian,quad) ))
    e = p(tn) - ph_tn
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

ID = 4
ns = [48,64,96,120]

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