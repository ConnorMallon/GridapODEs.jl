#Convergence for this module is the final stage - will confirm convergence of the final module - This involves the correct selectio nof all penalty parameters

module INS

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
dconv(du, ∇du, u, ∇u) = conv(u, ∇du) #+ conv(du, ∇u)

conv(u, ∇u) = (∇u') ⋅ u
dconv(du, ∇du, u, ∇u) = conv(u, ∇du) #+ conv(du, ∇u) #Changing to using the linear solver

# Physical constants
u_max = 150 # 150 #150# 150#  150 #cm/s
L = 1 #cm
ρ =  1.06e-3 #kg/cm^3 
μ =  3.50e-5 #kg/cm.s
ν = μ/ρ 
Δt =  0.046 * 0.1 / (u_max) # / (u_max) #/ 1000 # 0.046  #s \\

n=20
h=L/n

@show  Re = ρ*u_max*L/μ
@show C_t = u_max*Δt/h

# Manufactured solutions
θ = 1 
k=2*pi

u(x,t) = (u_max) * VectorValue( x[2] , - x[1] ) * cos( k * (t/Δt) )
u(t::Real) = x -> u(x,t)
ud(t) = x -> u(t)(x)

p(x,t) = (x[1]-x[2]) * sin(k* (t/Δt) )
p(t::Real) = x -> p(x,t)
q(x) = t -> p(x,t)

f(t) = x -> ρ * ∂t(u)(t)(x) + ρ * conv(u(t)(x),∇(u(t))(x)) - μ * Δ(u(t))(x) + ∇(p(t))(x)
g(t) = x -> (∇⋅u(t))(x)

u_Γn(t) = u(t)
p_Γn(t) = p(t)

function run_test(n_t)

n=20
t0 = 0.0
tF = Δt
dt = Δt/n_t
  
order = 1

# Select geometry
n = n
partition = (n,n)
D=length(partition)

h=L/n


@show  Re = ρ*u_max*L/μ
@show C_t = u_max*Δt/h

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

#Spaces
V0 = FESpace(
  model,
  reffeᵤ,
  conformity=:H1,
  #dirichlet_tags="boundary"
  )

reffeₚ = ReferenceFE(lagrangian,Float64,order)

Q = TestFESpace(
  model,
  reffeₚ,
  conformity=:H1,
  constraint=:zeromean)

U = TrialFESpace(V0)
P = TrialFESpace(Q)

X = MultiFieldFESpace([U,P])
Y = MultiFieldFESpace([V0,Q])

#NITSCHE
α_γ = 100
#@law 
γ(u) =  α_γ * ( μ / h  )# +  ρ * maximum(u) / 6 ) # Nitsche Penalty parameter ( γ / h ) 

@show VD = ν / h
@show CD = ρ * u_max / 6 
@show TD = h*ρ / (12*θ*Δt)

#STABILISATION
α_τ = 0.1 #Tunable coefficiant (0,1)
#@law 
τ_SUPG(u) = α_τ * inv(sqrt( (2/ Δt )^2 + ( 2 * normInf(u) / h )*( 2 * normInf(u) / h ) + 9 * ( 4*ν / h^2 )^2 )) # SUPG Stabilisation - convection stab ( τ_SUPG(u )
#@law 
τ_PSPG(u) = τ_SUPG(u) # PSPG stabilisation - inf-sup stab  ( ρ^-1 * τ_PSPG(u) )

# Ghost Penalty parameters  
α_B = 0.01 
α_u = 0.01 
α_p = 0.01 

#NS Paper ( DOI 10.1007/s00211-007-0070-5)
γ_B3(u)   = α_B * h^2  * abs(u.⁺ ⋅ n_Γg.⁺ )    #conv
γ_u3      = α_u * h^2  #visc diffusion 
γ_p3      = α_p * h^2  #pressure

## Weak form terms
#Interior terms
m_Ω(ut,v) = ρ * ut⊙v
a_Ω(u,v) = μ * ∇(u)⊙∇(v) 
b_Ω(v,p) = - (∇⋅v)*p
c_Ω(u, v) = ρ *  v ⊙ conv(u, ∇(u))
dc_Ω(u, du, v) = ρ * v ⊙ dconv(du, ∇(du), u, ∇(u))

#Boundary terms 
a_Γ(u,v) = μ* ( - (n_Γ⋅∇(u))⋅v - u⋅(n_Γ⋅∇(v)) ) + ( γ(u)/h )*u⋅v 
b_Γ(v,p) = (n_Γ⋅v)*p

#PSPG 
sp_Ω(w,p,q)    = (ρ^(-1) * τ_PSPG(w))     *  ∇(q) ⋅ ∇(p)
st_Ω(w,ut,q)   = (ρ^(-1) * τ_PSPG(w)) * ρ *  ∇(q) ⋅ ut
sc_Ω(w,u,q)    = (ρ^(-1) * τ_PSPG(w)) * ρ *  ∇(q) ⋅ conv(u, ∇(u))
dsc_Ω(w,u,du,q)= (ρ^(-1) * τ_PSPG(w)) * ρ *  ∇(q) ⋅ dconv(du, ∇(du), u, ∇(u))
ϕ_Ω(w,q,t)     = (ρ^(-1) * τ_PSPG(w))     *  ∇(q) ⋅ f(t)

#SUPG
sp_sΩ(w,p,v)    = τ_SUPG(w)     *  conv(w,∇(v)) ⋅ ∇(p)
st_sΩ(w,ut,v)   = τ_SUPG(w) * ρ *  conv(w,∇(v)) ⋅ ut
sc_sΩ(w,u,v)    = τ_SUPG(w) * ρ *  conv(w,∇(v)) ⋅ conv(u, ∇(u)) 
dsc_sΩ(w,u,du,v)= τ_SUPG(w) * ρ *  conv(w,∇(v)) ⋅ dconv(du, ∇(du), u, ∇(u)) 
ϕ_sΩ(w,v,t)     = τ_SUPG(w)     *  conv(w,∇(v)) ⋅ f(t)  

#Ghost Penalty terms
i_Γg(w,u,v) = ( γ_B3(w) + γ_u3 )*jump(n_Γg⋅∇(u))⋅jump(n_Γg⋅∇(v))
j_Γg(w,p,q) = ( γ_p3 ) * jump(n_Γg⋅∇(p))*jump(n_Γg⋅∇(q))

res(t,(u,p),(ut,pt),(v,q)) = 

∫( ( m_Ω(ut,v) + a_Ω(u,v) + b_Ω(v,p) + b_Ω(u,q) - v⋅f(t) + q*g(t) + c_Ω(u,v)  # + ρ * 0.5 * (∇⋅u) * u ⊙ v  
+1*(- sp_Ω(u,p,q)  -  st_Ω(u,ut,q)   + ϕ_Ω(u,q,t)     - sc_Ω(u,u,q) )
+1*(- sp_sΩ(u,p,v) - st_sΩ(u,ut,v)  + ϕ_sΩ(u,v,t)    - sc_sΩ(u,u,v) )))dΩ +

∫( a_Γ(u,v)+b_Γ(u,q)+b_Γ(v,p) - ud(t) ⊙(  ( γ(u)/h )*v - μ * n_Γ⋅∇(v) + q*n_Γ )  )dΓ #+

#∫(μ * - v⋅(n_Γn⋅∇(u_Γn(t))) + (n_Γn⋅v)*p_Γn(t) )dΓn + 

#∫( i_Γg(u,u,v) - j_Γg(u,p,q) )dΓg


jac(t,(u,p),(ut,pt),(du,dp),(v,q)) =

∫( ( a_Ω(du,v) + b_Ω(v,dp) + b_Ω(du,q)  + dc_Ω(u, du, v) # + ρ * 0.5 * (∇⋅u) * du ⊙ v 
+1* ( - sp_Ω(u,dp,q)  - dsc_Ω(u,u,du,q) )
+1*(- sp_sΩ(u,dp,v) - dsc_sΩ(u,u,du,v) ) ))dΩ + 

∫( a_Γ(du,v)+b_Γ(du,q)+b_Γ(v,dp)  )dΓ #+

#∫( i_Γg(u,du,v) - j_Γg(u,dp,q) )dΓg 


jac_t(t,(u,p),(ut,pt),(dut,dpt),(v,q)) = 
∫(  m_Ω(dut,v) 
+1* ( - st_Ω(u,dut,q) ) 
+1*(- st_sΩ(u,dut,v) ))dΩ



U0 = U(0.0)
P0 = P(0.0)
X0 = X(0.0)
uh0 = interpolate_everywhere(u(0.0),U0)
ph0 = interpolate_everywhere(p(0.0),P0)
xh0 = interpolate_everywhere([uh0,ph0],X0)

ls = LUSolver()

#nls = NewtonRaphsonSolver(ls,1e-10,40)

nls = NLSolver(
    show_trace = true,
    method = :newton,
    linesearch = BackTracking(),
)

op = TransientFEOperator(res,jac,jac_t,X,Y)

#ls=LUSolver()
#nls = NewtonRaphsonSolver(ls,1e-5,2)

odes = ThetaMethod(nls, dt, θ)
solver = TransientFESolver(odes)
sol_t = solve(solver, op, xh0, t0, tF)

l2(w) = w⋅w

tol = 1.0e-3
_t_n = t0

result = Base.iterate(sol_t)

eul2=[]
epl2=[]

for (xh_tn, tn) in sol_t
  _t_n += dt
  uh_tn = xh_tn[1]
  ph_tn = xh_tn[2]
  e_u = u(tn) - uh_tn
  e_p = p(tn) - ph_tn
  u_ex = u(tn) - 0*uh_tn
  p_ex = p(tn) - 0*ph_tn
  @show eul20 = sqrt(sum( ∫(l2(uh0-u(0)))dΩ ))
  @show epl20 = sqrt(sum( ∫(l2(ph0-p(0)))dΩ ))

  @show eul2i = sqrt(sum( ∫(l2(e_u))dΩ ))
  @show epl2i = sqrt(sum( ∫(l2(e_p))dΩ ))
  writevtk(Ω,"results_t_in",cellfields=["e_u"=>e_u,"uh_Ω"=>uh_tn,"u_ex"=>u_ex,"e_p"=>e_p,"ph_Ω"=>ph_tn,"p_ex"=>p_ex])
  push!(eul2,eul2i)
  push!(epl2,epl2i)
  @show tn
end


eul2=last(eul2)
epl2=last(epl2)

println((dt/tF)*n_t)

(eul2, epl2, h)
end

function conv_test(n_ts)

  eul2s = Float64[]
  epl2s = Float64[]
  hs = Float64[]

  for n_t in n_ts

    eul2, epl2, h = run_test(n_t)

    push!(eul2s,eul2)
    push!(epl2s,epl2)
    push!(hs,h)

  end

  (eul2s, epl2s,  hs)

end

ID = 1

n_ts = [8,16,24]

global ID = ID+1
eul2s, epl2s, hs = conv_test(n_ts);
using Plots
plot(n_ts,[eul2s, epl2s],
    xaxis=:log, yaxis=:log,
    label=["L2U" "L2P"],
    shape=:auto,
    xlabel="n_t",ylabel="L2 error norm",
    title = "INS_TimeConvergence,ID=$(ID)")
savefig("INS_TimeConvergence_in$(ID)")

end #module