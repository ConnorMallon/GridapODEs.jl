module StokesEquationConvergenceTests

using Gridap
using ForwardDiff
using LinearAlgebra
using Test
using GridapODEs.ODETools
using GridapODEs.TransientFETools
using Gridap.FESpaces: get_algebraic_operator

# using GridapODEs.ODETools: ThetaMethodLinear
import Gridap: ∇
import GridapODEs.TransientFETools: ∂t

θ = 0.5

k=2*pi
u(x,t) = VectorValue(x[1],x[2])*cos(k*t)
u(t::Real) = x -> u(x,t)

p(x,t) = (x[1]-x[2])*cos(k*t)
p(t::Real) = x -> p(x,t)
q(x) = t -> p(x,t)

f(t) = x -> ∂t(u)(t)(x)-Δ(u(t))(x)+ ∇(p(t))(x)
g(t) = x -> (∇⋅u(t))(x)

function run_test(dt)

n=16

  domain = (0,1,0,1)
  partition = (n,n)
  model = CartesianDiscreteModel(domain,partition)
  h=0.5

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

  #
  a(u,v) = inner(∇(u),∇(v))
  b(v,t) = inner(v,f(t))

  X = TransientMultiFieldFESpace([U,P])
  Y = MultiFieldFESpace([V0,Q])

  function res(t,x,xt,y)
    u,p = x
    ut,pt = xt
    v,q = y
    a(u,v) + inner(ut,v) - (∇⋅v)*p + q*(∇⋅u) - inner(v,f(t)) - q*g(t)
  end

  function jac(t,x,xt,dx,y)
    du,dp = dx
    v,q = y
    a(du,v)- (∇⋅v)*dp + q*(∇⋅du)
  end

  function jac_t(t,x,xt,dxt,y)
    dut,dpt = dxt
    v,q = y
    inner(dut,v)
  end

  function b(y)
    v,q = y
    0.0
    v⋅f(0.0) + q*g(0.0)
  end

  function mat(dx,y)
    du1,du2 = dx
    v1,v2 = y
    a(du1,v1)+a(du2,v2)
  end

  X0 = X(0.0)
  xh0 = interpolate_everywhere([u(0.0),p(0.0)],X0)

  t_Ω = FETerm(res,jac,jac_t,trian,quad)
  op = TransientFEOperator(X,Y,t_Ω)

  t0 = 0.0
  tF = 1.0
  dt = dt

  ls = LUSolver()
  odes = ThetaMethod(ls,dt,θ)
  solver = TransientFESolver(odes)

  sol_t = solve(solver,op,xh0,t0,tF)

  l2(w) = w⋅w

  tol = 1.0e-6
  _t_n = t0

  result = Base.iterate(sol_t)
l2(w) = w⋅w

tol = 1.0e-6
_t_n = t0

us = []
eul2 = []
epl2 = []

for (xh_tn, tn) in sol_t
  #global _t_n += dt
  _t_n += dt
  uh_tn = xh_tn[1]
  ph_tn = xh_tn[2]
  e = u(tn) - uh_tn
  eul2i = sqrt(sum( integrate(l2(e),trian,quad) ))
  e = p(tn) - ph_tn
  epl2i = sqrt(sum( integrate(l2(e),trian,quad) ))
  #@test epl2i<tol
  push!(eul2,eul2i)
  push!(epl2,epl2i)
end

eul2=last(eul2)
epl2=last(epl2)

(eul2, epl2, h)

end

function conv_test(dts)

  eul2s = Float64[]
  epl2s = Float64[]
  hs = Float64[]

  for dt in dts

    eul2, epl2, h = run_test(dt)

    push!(eul2s,eul2)
    push!(epl2s,epl2)
    push!(hs,h)

  end

  (eul2s, epl2s,  hs)

end

ID = 7

dts = [0.05,0.02,0.01]

global ID = ID+1
eul2s, epl2s, hs = conv_test(dts);
using Plots
plot(dts,[eul2s, epl2s],
    xaxis=:log, yaxis=:log,
    label=["L2U" "L2P"],
    shape=:auto,
    xlabel="h",ylabel="L2 error norm",
    title = "StokesTimeConvergence,ID=$(ID)")
savefig("StokesTimeConvergence_$(ID)")

function slope(dts,errors)
  x = log10.(dts)
  y = log10.(errors)
  linreg = hcat(fill!(similar(x), 1), x) \ y
  linreg[2]
end

@show slope(dts,eul2s)
@show slope(dts,epl2s)

end #module
