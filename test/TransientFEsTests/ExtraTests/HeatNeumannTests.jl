module HeatEquationTests

using Gridap
using ForwardDiff
using LinearAlgebra
using Test
using GridapODEs.ODETools
using GridapODEs.TransientFETools
using Gridap.FESpaces: get_algebraic_operator

import Gridap: ∇
import GridapODEs.TransientFETools: ∂t

using WriteVTK

θ = 1

u(x,t) = 3*(x[1]+x[2])#*t
u(t::Real) = x -> u(x,t)
f(t) = x -> ∂t(u)(x,t)-Δ(u(t))(x)

domain = (0,1,0,1)
partition = (2,2)
model = CartesianDiscreteModel(domain,partition)

labels = get_face_labeling(model)

add_tag_from_tags!(labels,"dirichlet",[1,2,3,4,5,6,7])
add_tag_from_tags!(labels,"neumann",[8])
writevtk(model,"model")

order = 1

V0 = FESpace(
  reffe=:Lagrangian, order=order, valuetype=Float64,
  conformity=:H1, model=model, dirichlet_tags="dirichlet")

U = TransientTrialFESpace(V0,u)

trian = Triangulation(model)
degree = 2*order
quad = CellQuadrature(trian,degree)

ntrian = BoundaryTriangulation(model,labels,"neumann")
ndegree = 2*order
nquad = CellQuadrature(ntrian,ndegree)
const nn = get_normal_vector(ntrian)

uh(t) = interpolate(u(t),U(t))
uh_Γn(t) = restrict(uh(t),ntrian)

a(u,v) = ∇(v)⊙∇(u)
b(v,t) = v⊙f(t)

l_Γn(v,t) = v⊙(nn⋅∇(uh_Γn(t)))

res(t,u,ut,v) = a(u,v) + ut*v - b(v,t)
jac(t,u,ut,du,v) = a(du,v) 
jac_t(t,u,ut,dut,v) = dut*v

res_Γn(t,u,ut,v) = -l_Γn(v,t)
jac_Γn(t,u,ut,du,v) = 0*a(du,v) 
jac_tΓn(t,u,ut,dut,v) = 0*dut*v

t_Ω = FETerm(res,jac,jac_t,trian,quad)
t_Γn = FETerm(res_Γn,jac_Γn,jac_tΓn,ntrian,nquad)
op = TransientFEOperator(U,V0,t_Ω,t_Γn)

t0 = 0.0
tF = 1.0
dt = 1.0

U0 = U(0.0)
uh0 = interpolate_everywhere(u(0.0),U0)

ls = LUSolver()
odes = ThetaMethod(ls,dt,θ)
solver = TransientFESolver(odes)

sol_t = solve(solver,op,uh0,t0,tF)

# Juno.@enter Base.iterate(sol_t)

l2(w) = w*w

tol = 1.0e-6
_t_n = t0

@show el20 =sqrt(sum( integrate(l2(u(0.0)-uh0),trian,quad) ))

for (uh_tn, tn) in sol_t
  global _t_n
  _t_n += dt
  e = u(tn) - uh_tn
  el2 = sqrt(sum( integrate(l2(e),trian,quad) ))
  #@test el2 < tol
  @show uh_tn.free_values[1] - u((0.5,0.5),tn)
  #push!(es,e)
  #push!(el2s,el2)
end

end #module

