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

using Gridap.Algebra: NewtonRaphsonSolver

function run_test(n)

θ = 0

u(x,t) = (cos(x[1])*sin(x[2]))*(t)
u(t::Real) = x -> u(x,t)
f(t) = x -> ∂t(u)(x,t)-Δ(u(t))(x)

domain = (0,1,0,1)
partition = (n,n)
h=1/n
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
op = TransientFEOperator(U,V0,t_Ω)

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

el2s=[]

for (uh_tn, tn) in sol_t
  _t_n
  _t_n += dt
  e = u(tn) - uh_tn
  el2 = sqrt(sum( integrate(l2(e),trian,quad) ))
  push!(el2s,el2)
end

eul2=last(el2s)

(eul2, h)

end

function conv_test(ns)

eul2s = Float64[]
hs = Float64[]

for n in ns

    eul2, h = run_test(n)

    push!(eul2s,eul2)
    push!(hs,h)

end

(eul2s,  hs)

end

ID = 1
ns = [16,32,64]

global ID = ID+1
eul2s,  hs = conv_test(ns);
using Plots
plot(hs,[eul2s],
    xaxis=:log, yaxis=:log,
    label=["L2U" "L2P"],
    shape=:auto,
    xlabel="h",ylabel="L2 error norm",
    title = "Heat_SpaceConvergence,ID=$(ID)")
savefig("Heat_SpaceConvergence_$(ID)")


function slope(dts,errors)
x = log10.(dts)
y = log10.(errors)
linreg = hcat(fill!(similar(x), 1), x) \ y
linreg[2]
end

@show slope(hs,eul2s)

end #module