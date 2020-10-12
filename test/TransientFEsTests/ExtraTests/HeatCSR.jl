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
using GridapPardiso

θ = 0.0

u(x,t) = (x[1]-x[2])*t
u(t::Real) = x -> u(x,t)
f(t) = x -> ∂t(u)(x,t)-Δ(u(t))(x)

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

res(t,u,ut,v) = a(u,v) + ut*v - b(v,t)
jac(t,u,ut,du,v) = a(du,v)
jac_t(t,u,ut,dut,v) = dut*v

t_Ω = FETerm(res,jac,jac_t,trian,quad)
#op = TransientFEOperator(U,V0,t_Ω)

using Gridap.FESpaces: SparseMatrixAssembler

matCSR = SparseMatrixCSR{1,Float64,Int}

op = TransientFEOperator(matCSR,U,V0,t_Ω)

t0 = 0.0
tF = 1.0
dt = 1.0

U0 = U(0.0)
uh0 = interpolate_everywhere(u(0.0),U0)

ls = LUSolver()

odes = ThetaMethod(ls,dt,θ)
#odes = RungeKutta(ls,dt,:SIE_2_1)

solver = TransientFESolver(odes)

sol_t = solve(solver,op,uh0,t0,tF)

# Juno.@enter Base.iterate(sol_t)

l2(w) = w*w

tol = 1.0e-6
_t_n = t0

for (uh_tn, tn) in sol_t
  global _t_n
  _t_n += dt
  e = u(tn) - uh_tn
  el2 = sqrt(sum( integrate(l2(e),trian,quad) ))
  @test el2 < tol
  @show uh_tn.free_values[1] - u((0.5,0.5),tn)
end

end #module
