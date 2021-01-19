module inscut

using Gridap
using ForwardDiff
using LinearAlgebra
using Test
using GridapODEs.ODETools
using GridapODEs.TransientFETools
using Gridap.FESpaces: get_algebraic_operator
using GridapEmbedded
import Gridap: ∇
import GridapODEs.TransientFETools: ∂t
using LineSearches: BackTracking
using Gridap.Algebra: NewtonRaphsonSolver
using Gridap.CellData

conv(u, ∇u) = (∇u') ⋅ u
dconv(du, ∇du, u, ∇u) = conv(u, ∇du) #+ conv(du, ∇u) #Semi-Implicit

# Constants
u_max = 150 #cm/s
L = 1 #cm
ρ =  1.06e-3 #kg/cm^3 
μ =  3.50e-5 #kg/cm.s
ν = μ/ρ 
Δt =  0.046 / (u_max)  #s \\

n=10
h=L/n
@show  Re = ρ*u_max*L/μ
@show C_t = u_max*Δt/h
n_t= 10
t0 = 0.0
tF = Δt * n_t
dt = Δt
θ=1

# Manufactured solutions 
k=2*pi
u(x,t) = u_max * VectorValue( x[1] , -x[2] ) * (t/tF)
u(t::Real) = x -> u(x,t)
ud(t) = x -> u(t)(x)
p(x,t) = (x[1]-x[2])* (t/tF)
p(t::Real) = x -> p(x,t)
q(x) = t -> p(x,t)
f(t) = x -> ρ * ∂t(u)(t)(x) - μ * Δ(u(t))(x) + ∇(p(t))(x) + ρ * conv(u(t)(x),∇(u(t))(x)) 
g(t) = x -> (∇⋅u(t))(x)
u_Γn(t) = u(t)
p_Γn(t) = p(t)

order = 2

# Select geometry
n = n
partition = (n,n)
D=length(partition)

# Setup model
domain = (0,L,0,L)
model = simplexify(CartesianDiscreteModel(domain,partition))
Ω = Triangulation(model)
degree = 2*order
dΩ = Measure(Ω,degree)

#FE spaces
reffeᵤ = ReferenceFE(lagrangian,VectorValue{D,Float64},order)

V0 = FESpace(
  model,
  reffeᵤ,
  conformity=:H1,
  dirichlet_tags="boundary"
  )

reffeₚ = ReferenceFE(lagrangian,Float64,order-1)

Q = TestFESpace(
  model,
  reffeₚ,
  conformity=:H1,
  constraint=:zeromean)

U = TransientTrialFESpace(V0,u)
P = TrialFESpace(Q)

X = TransientMultiFieldFESpace([U,P])
Y = MultiFieldFESpace([V0,Q])

X0 = X(0.0)
xh0 = interpolate_everywhere(X0,[u(0.0),p(0.0)])

## Weak form 
#Interior terms
m_Ω(ut,v) = ρ * ut⊙v
a_Ω(u,v) = μ * ∇(u)⊙∇(v) 
b_Ω(v,p) = - (∇⋅v)*p
c_Ω(u, v) = ρ *  v ⊙ conv(u, ∇(u))
dc_Ω(u, du, v) = ρ * v ⊙ dconv(du, ∇(du), u, ∇(u))

#SUPG terms (see https://doi.org/10.1016/0045-7825(92)90141-6)
α_τ = 0.1 #Tunable coefficiant (0,1) 0 for quicker conv, 1 for better stab
τ_SUPG(u) = α_τ * inv(sqrt( (2/ Δt )^2 + ( 2 * normInf(u) / h )*( 2 * normInf(u) / h ) + 9 * ( 4*ν / h^2 )^2 )) # SUPG Stabilisation - convection stab ( τ_SUPG(u )

sp_sΩ(w,p,v)    = τ_SUPG(w)     *  conv(w,∇(v)) ⋅ ∇(p)
st_sΩ(w,ut,v)   = τ_SUPG(w) * ρ *  conv(w,∇(v)) ⋅ ut
sc_sΩ(w,u,v)    = τ_SUPG(w) * ρ *  conv(w,∇(v)) ⋅ conv(u, ∇(u)) 
dsc_sΩ(w,u,du,v)= τ_SUPG(w) * ρ *  conv(w,∇(v)) ⋅ dconv(du, ∇(du), u, ∇(u)) 
ϕ_sΩ(w,v,t)     = τ_SUPG(w)     *  conv(w,∇(v)) ⋅ f(t)  

#Term collection
res(t,(u,p),(ut,pt),(v,q)) = 
∫(  m_Ω(ut,v) + a_Ω(u,v) + b_Ω(v,p) + b_Ω(u,q) - v⋅f(t) + q*g(t) + c_Ω(u,v)  
+(- sp_sΩ(u,p,v) - st_sΩ(u,ut,v)  + ϕ_sΩ(u,v,t)    - sc_sΩ(u,u,v) ) )dΩ 

jac(t,(u,p),(ut,pt),(du,dp),(v,q)) =
∫(  a_Ω(du,v) + b_Ω(v,dp) + b_Ω(du,q)  + dc_Ω(u, du, v) + 0*∇(q)⋅∇(dp) #errors when ∇(q)⋅∇(dp) not included - not sure why
+(- sp_sΩ(u,dp,v) - dsc_sΩ(u,u,du,v) ) )dΩ  

jac_t(t,(u,p),(ut,pt),(dut,dpt),(v,q)) = 
∫(  m_Ω(dut,v) 
+(- st_sΩ(u,dut,v) ) )dΩ

#Solve
nls = NLSolver(
    show_trace = true,
    method = :newton,
    linesearch = BackTracking(),
)
op = TransientFEOperator(res,jac,jac_t,X,Y)
odes = ThetaMethod(nls, dt, θ)
solver = TransientFESolver(odes)
sol_t = solve(solver, op, xh0, t0, tF)

l2(w) = w⋅w
tol = 1.0e-5
_t_n = t0

result = Base.iterate(sol_t)

eul2=[]
epl2=[]

for (xh_tn, tn) in sol_t
  global _t_n += dt
  uh_tn = xh_tn[1]
  ph_tn = xh_tn[2]
  e = u(tn) - uh_tn
  eul2i = sqrt(sum( ∫(l2(e))dΩ ))
  @test eul2i < tol
  e = p(tn) - ph_tn
  epl2i = sqrt(sum( ∫(l2(e))dΩ ))
  @test epl2i < tol
  push!(eul2,eul2i)
  push!(epl2,epl2i)
  writevtk(Ω,"results",cellfields=["uh_Ω"=>uh_tn,"ph_tn"=>ph_tn])
end

end #module