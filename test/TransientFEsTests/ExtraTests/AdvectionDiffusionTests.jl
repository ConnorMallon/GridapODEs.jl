module AdvectionDiffusionTests

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


β(x,t) = VectorValue( x[1] , -x[2] ) 
β(t::Real) = x -> β(x,t)
βd(t) = x -> β(t)(x)

c(x,t)= (x[1]+x[2])
c(t::Real) = x -> c(x,t)
f(t) = x -> ∂t(c)(x,t)-Δ(c(t))(x)  +  β(t)(x)⋅∇(c(t))(x)

domain = (0,1,0,1)
partition = (2,2)
model = CartesianDiscreteModel(domain,partition)

order = 1

reffe = ReferenceFE(lagrangian,Float64,order)
V0 = FESpace(
  model,
  reffe,
  conformity=:H1,
  dirichlet_tags="boundary"
)
U = TransientTrialFESpace(V0,c)

Ω = Triangulation(model)
degree = 2*order
dΩ = Measure(Ω,degree)

#
a(u,v) = ∫(∇(v)⋅∇(u))dΩ
b(v,t) = ∫(v*f(t))dΩ

res(t,c,ct,v) = a(c,v) + ∫(ct*v)dΩ - b(v,t) + ∫( βd(t) ⋅ ∇(c) ⋅ (v) )dΩ
jac(t,c,ct,dc,v) = a(dc,v) + ∫( βd(t) ⋅ ∇(dc) ⋅ (v) )dΩ
jac_t(t,c,ct,dct,v) = ∫(dct*v)dΩ

op = TransientFEOperator(res,jac,jac_t,U,V0)

t0 = 0.0
tF = 1.0
dt = 1.0

U0 = U(0.0)
uh0 = interpolate_everywhere(c(0.0),U0)

ls = LUSolver()
#using Gridap.Algebra: NewtonRaphsonSolver
#nls = NLSolver(ls;show_trace=true,method=:newton) #linesearch=BackTracking())

odes = ThetaMethod(ls,dt,θ)

solver = TransientFESolver(odes)

sol_t = solve(solver,op,uh0,t0,tF)

# Juno.@enter Base.iterate(sol_t)

l2(w) = w*w

tol = 1.0e-6
_t_n = t0

for (ch_tn, tn) in sol_t
  global _t_n
  _t_n += dt
  e = c(tn) - ch_tn
  el2 = sqrt(sum( ∫(l2(e))dΩ ))
  @test el2 < tol
end

end #module
