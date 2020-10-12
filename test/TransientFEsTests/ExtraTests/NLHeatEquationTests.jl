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


θ = 1

# Analytical functions
# u(x,t) = (x[1]+x[2])*t
# u(x,t) = (2*x[1]+x[2])*t
u(x,t) = (x[1]-x[2])*t
u(t::Real) = x -> u(x,t)
v(x) = t -> u(x,t)
∂tu(t) = x -> ForwardDiff.derivative(v(x),t)
∂tu(x,t) = ∂tu(t)(x)
∂t(::typeof(u)) = ∂tu
f(t) = x -> ∂t(u)(x,t)-Δ(u(t))(x) + u(t)(x)^2 


domain = (0,1,0,1)
partition = (2,2)
model = CartesianDiscreteModel(domain,partition)

order = 1

V0 = FESpace(
  reffe=:Lagrangian, order=order, valuetype=Float64,
  conformity=:H1, model=model, dirichlet_tags="boundary")
U = TransientTrialFESpace(V0,u)

trian = Triangulation(model)
degree = 2*order
quad = CellQuadrature(trian,degree)

#
a(u,v) = ∇(v)⋅∇(u) 
b(v,t) = v*f(t) 
c(u,v) = v*u*u
dc(du,u,v) = v*2*u*du

res(t,u,ut,v) = a(u,v) + ut*v - b(v,t) + c(u,v)
jac(t,u,ut,du,v) = 0*(a(du,v) + dc(du,u,v))
jac_t(t,u,ut,dut,v) = dut*v

t_Ω = FETerm(res,jac,jac_t,trian,quad)
op = TransientFEOperator(U,V0,t_Ω)

t0 = 0.0
tF = 1.0
dt = 0.1

U0 = U(0.0)
uh0 = interpolate_everywhere(u(0.0),U0)

ls = LUSolver()
using Gridap.Algebra: NewtonRaphsonSolver
nls = NLSolver(ls;show_trace=true,method=:newton) #linesearch=BackTracking())
odes = ThetaMethod(nls,dt,θ)
solver = TransientFESolver(odes)

sol_t = solve(solver,op,uh0,t0,tF)

l2(w) = w*w

tol = 1.0e-4
_t_n = t0

for (uh_tn, tn) in sol_t
  global _t_n
  _t_n += dt
  e = u(tn) - uh_tn
  el2 = sqrt(sum( integrate(l2(e),trian,quad) ))
  @test el2 < tol
end

end #module
