module NavierStokesNeumannTests

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
using WriteVTK

@law conv(u, ∇u) = (∇u') ⋅ u
@law dconv(du, ∇du, u, ∇u) = conv(u, ∇du) #+ conv(du, ∇u) 

θ = 1.0

ν = 1#0.01

u(x,t) = VectorValue(x[1],-x[2])*t
u(t::Real) = x -> u(x,t)

p(x,t) = (x[1]-x[2])*t
p(t::Real) = x -> p(x,t)
q(x) = t -> p(x,t)

f(t) = x -> ∂t(u)(t)(x) - ν * Δ(u(t))(x) + ∇(p(t))(x) + conv(u(t)(x),∇(u(t))(x))
g(t) = x -> (∇⋅u(t))(x)

u_Γ(t) = u(t)
p_Γ(t) = p(t)

domain = (0,1,0,1)
partition = (2,2)
model = CartesianDiscreteModel(domain,partition)

labels = get_face_labeling(model)
add_tag_from_tags!(labels,"dirichlet",[1,2,3,5,6,7])
add_tag_from_tags!(labels,"neumann",[4,6,8])

order = 2

V0 = FESpace(
  reffe=:Lagrangian, order=order, valuetype=VectorValue{2,Float64},
  conformity=:H1, model=model 
  ,dirichlet_tags="dirichlet"
  #,dirichlet_tags="boundary"
  )

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

ntrian = BoundaryTriangulation(model,labels,"neumann")
ndegree = 2*order
nquad = CellQuadrature(ntrian,ndegree)
const n_Γ = get_normal_vector(ntrian)

a(u,v) = ν *inner(∇(u),∇(v))
b(v,t) = inner(v,f(t))
c_Ω(u, v) = v ⊙ conv(u, ∇(u)) 
dc_Ω(u, du, v) = v ⊙ dconv(du, ∇(du), u, ∇(u))

a_Γ(u,v) = - (n_Γ⋅∇(u))⋅v - u⋅(n_Γ⋅∇(v))
b_Γ(v,p) = (n_Γ⋅v)*p

_c_Γ(u)  = min.((n_Γ⋅(u)).array,0*u.array)
c_Γ(u,v) = -_c_Γ(u) ⋅ u ⊙ v

X = TransientMultiFieldFESpace([U,P])
Y = MultiFieldFESpace([V0,Q])

function res(t,x,xt,y)
  u,p = x
  ut,pt = xt
  v,q = y
  a(u,v) + inner(ut,v) - (∇⋅v)*p + q*(∇⋅u) - inner(v,f(t)) - q*g(t) + c_Ω(u, v) + 0.5 * (∇⋅u) * u ⊙ v 
end

function jac(t,x,xt,dx,y)
  u, p = x
  du,dp = dx
  v,q = y
  a(du,v)- (∇⋅v)*dp + q*(∇⋅du) + dc_Ω(u, du, v) + 0.5 * (∇⋅u) * du ⊙ v 
end

function jac_t(t,x,xt,dxt,y)
  dut,dpt = dxt
  v,q = y
  inner(dut,v)
end

function res_Γn(t,x,xt,y)
  u,p = x
  ut,pt = xt
  v,q = y
  - v⋅(n_Γ⋅∇(u_Γ(t))) + (n_Γ⋅v)*p_Γ(t) + c_Γ(u,v)
end

function jac_Γn(t,x,xt,dx,y)
  u, p = x
  du,dp = dx
  v,q = y
  0*a_Γ(du,v)
end

function jac_tΓn(t,x,xt,dxt,y)
  dut,dpt = dxt
  v,q = y
  0*inner(dut,v)
end

U0 = U(0.0)
P0 = P(0.0)
X0 = X(0.0)
uh0 = interpolate_everywhere(u(0.0),U0)

uh0_ = restrict(uh0,trian)
uh0__ = uh0_.array.g.value
fieldnames(typeof(uh0__))

ph0 = interpolate_everywhere(p(0.0),P0)
xh0 = interpolate_everywhere([uh0,ph0],X0)

t_Ω = FETerm(res,jac,jac_t,trian,quad)
t_Γn = FETerm(res_Γn,jac_Γn,jac_tΓn,ntrian,nquad)
op = TransientFEOperator(X,Y,t_Ω,t_Γn)

t0 = 0.0
tF = 1.0
dt = 0.1

ls = LUSolver()
nls = NewtonRaphsonSolver(ls,1e-5,10)

odes = ThetaMethod(nls,dt,θ)

solver = TransientFESolver(odes)
sol_t = solve(solver,op,xh0,t0,tF)

l2(w) = w⋅w

tol = 1.0e-4
_t_n = t0

result = Base.iterate(sol_t)

 #=
uh0 = xh0.single_fe_functions[1]
@show el20 =sqrt(sum( integrate(l2(u(0.0)-uh0),trian,quad) ))

ph0 = xh0.single_fe_functions[2]
@show ep20 =sqrt(sum( integrate(l2(p(0.0)-ph0),trian,quad) ))
#@test el20 < tol
 =#

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