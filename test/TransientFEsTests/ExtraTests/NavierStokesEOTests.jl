module StokesEquationTests

# using GridapODEs.ODETools: ThetaMethodLinear
import Gridap: ∇
import GridapODEs.TransientFETools: ∂t

using Gridap
using ForwardDiff
using LinearAlgebra
using Test
using GridapODEs.ODETools
using GridapODEs.TransientFETools
using Gridap.FESpaces: get_algebraic_operator
#using GridapEmbedded
import Gridap: ∇
import GridapODEs.TransientFETools: ∂t
using LineSearches: BackTracking
using Gridap.Algebra: NewtonRaphsonSolver

@law conv(u, ∇u) = (∇u') ⋅ u
@law dconv(du, ∇du, u, ∇u) = conv(u, ∇du) #+ conv(du, ∇u) 

θ = 1.0

# Physical constants
u_max = 150 # 150 #150# 150#  150 #cm/s
L = 1 #cm
ρ =  1.06e-3 #kg/cm^3 
μ =  3.50e-5 #kg/cm.s
ν = μ/ρ 
Δt =  0.046 #/ 100 #/ 1000 # 0.046  #s \\

n=20
h=L/n

@show  Re = ρ*u_max*L/μ
@show C_t = u_max*Δt/h

n_t= 10
t0 = 0.0
tF = Δt * n_t
dt = Δt

u(x,t) = VectorValue(x[1],-x[2]) *t
u(t::Real) = x -> u(x,t)

p(x,t) = (x[1]-x[2]) *t
p(t::Real) = x -> p(x,t)
q(x) = t -> p(x,t)

f(t) = x -> ρ * ∂t(u)(t)(x) - μ * Δ(u(t))(x) + ∇(p(t))(x) + ρ * conv(u(t)(x),∇(u(t))(x)) 
g(t) = x -> (∇⋅u(t))(x)

domain = (0,1,0,1)
partition = (2,2)
model = simplexify(CartesianDiscreteModel(domain,partition))

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

trian = Triangulation(model)
degree = 2*order
quad = CellQuadrature(trian,degree)

X = TransientMultiFieldFESpace([U,P])
Y = MultiFieldFESpace([V0,Q])

#
## Weak form terms
#Interior terms
m_Ω(ut,v) = ρ * ut⊙v
a_Ω(u,v) = μ * ∇(u)⊙∇(v) 
b_Ω(v,p) = - (∇⋅v)*p
c_Ω(u, v) = ρ *  v ⊙ conv(u, ∇(u))
dc_Ω(u, du, v) = ρ * v ⊙ dconv(du, ∇(du), u, ∇(u))

#STABILISATION
α_τ = 1 #Tunable coefficiant (0,1)
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

#Interior term collection
function res_Ω(t,x,xt,y)
  u,p = x
  ut,pt = xt
  v,q = y
  ( m_Ω(ut,v) + a_Ω(u,v) + b_Ω(v,p) + b_Ω(u,q) - v⋅f(t) + q*g(t) + c_Ω(u,v)  # + ρ * 0.5 * (∇⋅u) * u ⊙ v  
  - sp_Ω(u,p,q)  -  st_Ω(u,ut,q)   + ϕ_Ω(u,q,t)     - sc_Ω(u,u,q) 
  - sp_sΩ(u,p,v) - st_sΩ(u,ut,v)  + ϕ_sΩ(u,v,t)    - sc_sΩ(u,u,v) )
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

 
U0 = U(0.0)
P0 = P(0.0)
X0 = X(0.0)
uh0 = interpolate_everywhere(u(0.0),U0)
ph0 = interpolate_everywhere(p(0.0),P0)
xh0 = interpolate_everywhere([uh0,ph0],X0)

t_Ω = FETerm(res_Ω,jac_Ω,jac_tΩ,trian,quad)
op = TransientFEOperator(X,Y,t_Ω)

ls = LUSolver()

#=
nls = NLSolver(
    show_trace = false,
    method = :newton,
    linesearch = BackTracking(),
)

nls = NLSolver(ls;show_trace=true,method=:newton) #linesearch=BackTracking())
=#

nls = NewtonRaphsonSolver(ls,1e-10,40)
#nls = NewtonRaphsonSolver(ls,1e99,1)

#odes = ForwardEuler(ls,dt)


odes = ThetaMethod(nls,dt,θ)


solver = TransientFESolver(odes)
sol_t = solve(solver,op,xh0,t0,tF)

l2(w) = w⋅w

tol = 1.0e-6
_t_n = t0

result = Base.iterate(sol_t)

# #=
uh0 = xh0.single_fe_functions[1]
@show el20 =sqrt(sum( integrate(l2(u(0.0)-uh0),trian,quad) ))

ph0 = xh0.single_fe_functions[2]
@show ep20 =sqrt(sum( integrate(l2(p(0.0)-ph0),trian,quad) ))
#@test el20 < tol
# =#

for (xh_tn, tn) in sol_t
  global _t_n
  _t_n += dt
  uh_tn = xh_tn[1]
  ph_tn = xh_tn[2]
  e = u(tn) - uh_tn
  el2 = sqrt(sum( integrate(l2(e),trian,quad) ))
  e = p(tn) - ph_tn
  el2 = sqrt(sum( integrate(l2(e),trian,quad) ))
  @show el2
  @test el2 < tol
end

end #module