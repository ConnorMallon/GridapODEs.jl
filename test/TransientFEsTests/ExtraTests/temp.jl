module INS_SI_EquationTests

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





#Testing parameters
θ = 1
νs = [1,0.1,0.01]

@law conv(u, ∇u) = (∇u') ⋅ u
@law dconv(du, ∇du, u, ∇u) = conv(u, ∇du) #+ (∇⋅u) #0.5*divergence(u) * du #Changing to using the linear solver

k=2*pi
u(x,t) = VectorValue(-cos(k*x[1])*sin(k*x[2]),sin(k*x[1])*cos(k*x[2]))*(t)
u(t::Real) = x -> u(x,t)

p(x,t) = k*(sin(k*x[1])-sin(k*x[2]))*t
p(t::Real) = x -> p(x,t)
q(x) = t -> p(x,t)


function run_test(n,ν)

f(t) = x -> ∂t(u)(t)(x) - ν * Δ(u(t))(x) + ∇(p(t))(x) + conv(u(t)(x),∇(u(t))(x))
g(t) = x -> (∇⋅u(t))(x)

domain = (0,1,0,1)
partition = (n,n)
h=1/n
model = CartesianDiscreteModel(domain,partition)

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

m(u,v) = u⊙v
a(u,v) = ν *  ∇(u)⊙∇(v)
c_Ω(u, v) = v ⊙ conv(u, ∇(u))
dc_Ω(u, du, v) = v ⊙ dconv(du, ∇(du), u, ∇(u))

X = TransientMultiFieldFESpace([U,P])
Y = MultiFieldFESpace([V0,Q])

function res(t,x,xt,y)
  u,p = x
  ut,pt = xt
  v,q = y
  m(ut,v) + a(u,v) - (∇⋅v)*p + q*(∇⋅u) - v⋅f(t) - q*g(t) + c_Ω(u, v) #+ 0.5 * (∇⋅u) * u ⊙ v
end

function jac(t,x,xt,dx,y)
  u, p = x
  du,dp = dx
  v,q = y
  a(du,v)- (∇⋅v)*dp + q*(∇⋅du) + dc_Ω(u, du, v) #+ 0.5 * (∇⋅u) * du ⊙ v
end

function jac_t(t,x,xt,dxt,y)
  dut,dpt = dxt
  v,q = y
  m(dut,v)
end

X0 = X(0.0)
xh0 = interpolate_everywhere([u(0.0),p(0.0)],X0)

t_Ω = FETerm(res,jac,jac_t,trian,quad)
op = TransientFEOperator(X,Y,t_Ω)

t0 = 0.0
tF = 1.0
dt = 1.0

ls = LUSolver()

nls = NLSolver(
    show_trace = false,
    method = :newton,
    linesearch = BackTracking(),
)
nls = NewtonRaphsonSolver(ls,1e99,1)

odes = ThetaMethod(nls,dt,θ)
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
eul2=[]
epl2=[]

for (xh_tn, tn) in sol_t
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

function conv_test(ns,ν)

  eul2s = Float64[]
  epl2s = Float64[]
  hs = Float64[]

  for n in ns

    eul2, epl2, h = run_test(n,ν)

    push!(eul2s,eul2)
    push!(epl2s,epl2)
    push!(hs,h)

  end

  (eul2s, epl2s,  hs)

end

ID = 0
ns = [8,16,24,48]

using Plots
plot()

for ν in νs
    global ID = ID+1
    eul2s, epl2s, hs = conv_test(ns,ν);
    @show hs

    plot!(hs,
      [eul2s, epl2s],
      #[epl2s],
      
      xaxis=:log, yaxis=:log,
      
      label=["L2U $(ν)" "L2P $(ν)"],
      #label=["L2P $(ν)"],
      
      shape=:auto,
      xlabel="h",ylabel="L2 error norm",
      title = "INS__SI_SpaceConvergemce,ID=$(ID)")
end
  
savefig("INS_SI_SpaceConvergence_$(ID)")

end #module

