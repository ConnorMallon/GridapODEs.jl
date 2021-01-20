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

# Physical constants
u_max = 10 #150 # 150 #150# 150#  150 #cm/s
L = 1 #cm
ρ =  1 # 1.06e-3 #kg/cm^3 
μ =  1# 3.50e-5 #kg/cm.s
ν = μ/ρ 
Δt =  0.001/u_max  #0.046 * 0.1  # / (u_max) #/ 1000 # 0.046  #s \\

n=20
h=L/n

@show  Re = ρ*u_max*L/μ
@show C_t = u_max*Δt/h

# Manufactured solutions
θ = 1 
k=2*pi

u(x,t) = (u_max) * VectorValue( x[2] , - x[1] ) * sin( k * (t/Δt) )
u(t::Real) = x -> u(x,t)
ud(t) = x -> u(t)(x)

p(x,t) = u_max * (x[1]-x[2]) * cos(k* (t/Δt) )
p(t::Real) = x -> p(x,t)
q(x) = t -> p(x,t)

f(t) = x -> ρ * ∂t(u)(t)(x)  - μ * Δ(u(t))(x) + ∇(p(t))(x) #+ ρ * conv(u(t)(x),∇(u(t))(x))
g(t) = x -> (∇⋅u(t))(x)

function run_test(nt)

#Δt = 1
n=20
t0 = 0.0
#tF = Δt
#dt = Δt/n_t
  
order = 1

# Select geometry
L=1
n = n
h=L/n
partition = (n,n)
D=length(partition)
# Setup background model
domain = (0,L,0,L)
bgmodel = simplexify(CartesianDiscreteModel(domain,partition))
#const h = L/n

#Non-embedded geometry 
model=bgmodel

Ω = Triangulation(model)
degree = 2*order
dΩ = Measure(Ω,degree)

Γ = BoundaryTriangulation(model)
dΓ = Measure(Γ,degree)
n_Γ = get_normal_vector(Γ)

reffeᵤ = ReferenceFE(lagrangian,VectorValue{D,Float64},order)

order = 2

reffeᵤ = ReferenceFE(lagrangian,VectorValue{2,Float64},order)

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
  constraint=:zeromean
)

U = TransientTrialFESpace(V0,u)

P = TrialFESpace(Q)

Ω = Triangulation(model)
degree = 2*order
dΩ = Measure(Ω,degree)

Γ = BoundaryTriangulation(model)
dΓ = Measure(Γ,degree)
nb = get_normal_vector(Γ)

a(u,v) = ∫(μ*∇(u)⊙∇(v))dΩ
b((v,q),t) = ∫(v⋅f(t))dΩ + ∫(q*g(t))dΩ
m(ut,v) = ∫(ρ*(ut⋅v))dΩ

X = TransientMultiFieldFESpace([U,P])
Y = MultiFieldFESpace([V0,Q])

res(t,(u,p),(ut,pt),(v,q)) = 
  a(u,v)  +
  m(ut,v) - 
  ∫((∇⋅v)*p)dΩ +
  ∫(q*(∇⋅u))dΩ - 
  b((v,q),t) 

jac(t,(u,p),(ut,pt),(du,dp),(v,q)) = a(du,v) - ∫((∇⋅v)*dp)dΩ + ∫(q*(∇⋅du))dΩ
jac_t(t,(u,p),(ut,pt),(dut,dpt),(v,q)) = m(dut,v)

b((v,q)) = b((v,q),0.0)

mat((du1,du2),(v1,v2)) = a(du1,v1)+a(du2,v2)

U0 = U(0.0)
P0 = P(0.0)
X0 = X(0.0)
uh0 = interpolate_everywhere(u(0.0),U0)
ph0 = interpolate_everywhere(p(0.0),P0)
xh0 = interpolate_everywhere([uh0,ph0],X0)

op = TransientFEOperator(res,jac,jac_t,X,Y)

t0 = 0.0
tF = Δt #1.0
dt = Δt/nt

ls = LUSolver()
odes = ThetaMethod(ls,dt,θ)
solver = TransientFESolver(odes)

sol_t = solve(solver,op,xh0,t0,tF)

l2(w) = w⋅w

tol = 1.0e-6
_t_n = t0

result = Base.iterate(sol_t)

eul2 = []
epl2 = []


for (xh_tn, tn) in sol_t
  _t_n += dt
  uh_tn = xh_tn[1]
  ph_tn = xh_tn[2]
  e_u = u(tn) - uh_tn
  e_p = p(tn) - ph_tn
  u_ex = u(tn) - 0*uh_tn
  p_ex = p(tn) - 0*ph_tn
  #@show eul20 = sqrt(sum( ∫(l2(uh0-u(0)))dΩ ))
  #@show epl20 = sqrt(sum( ∫(l2(ph0-p(0)))dΩ ))
  @show eul2i = sqrt(sum( ∫(l2(e_u))dΩ ))
  @show epl2i = sqrt(sum( ∫(l2(e_p))dΩ ))
  if nt == 8
  writevtk(Ω,"results_t_stokes_$(_t_n)",cellfields=["e_u"=>e_u,"uh_Ω"=>uh_tn,"u_ex"=>u_ex,"e_p"=>e_p,"ph_Ω"=>ph_tn,"p_ex"=>p_ex])
  end
  push!(eul2,eul2i)
  push!(epl2,epl2i)
  @show tn
end

eul2=last(eul2)
epl2=last(epl2)

(eul2, epl2, h)

end

function conv_test(nts)

  eul2s = Float64[]
  epl2s = Float64[]
  hs = Float64[]

  for nt in nts
    eul2, epl2, h = run_test(nt)
    push!(eul2s,eul2)
    push!(epl2s,epl2)
    push!(hs,h)

  end

  (eul2s, epl2s,  hs)

end

ID = 7

nts = [8,16,32]

global ID = ID+1
eul2s, epl2s, hs = conv_test(nts);
using Plots
plot(nts,[eul2s, epl2s],
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

@show slope(nts,eul2s)
@show slope(nts,epl2s)

end #module
