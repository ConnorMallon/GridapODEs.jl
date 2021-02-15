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

conv(u, ∇u) = (∇u') ⋅ u
dconv(du, ∇du, u, ∇u) = conv(u, ∇du) #+ conv(du, ∇u) #Changing to using the linear solver

#manufactured solutions
u(x,t) = VectorValue(x[2],-x[1])
u(t::Real)= x -> u(x,t)

p(x,t) = (x[1]-x[2]) 
p(t::Real) = x -> p(x,t)

c(x,t)= (x[1]*x[2])*t
c(t::Real) = x -> c(x,t)

m(x,t) = (x[1]*x[2]+x[2])*t  #we can choose m such that ∇c.n-f_c=0 --> m = c-∇c.n on Γ 
m(t::Real) = x -> m(x,t)

fINS_u(t) = x -> ρ * ∂t(u)(t)(x) - μ * Δ(u(t))(x) + ∇(p(t))(x) + ρ * conv(u(t)(x),∇(u(t))(x)) 
fINS_p(t) = x -> (∇⋅u(t))(x)

fAD_c(t) = x -> ∂t(c)(x,t) - Δ(c(t))(x) +  u(t)(x)⋅∇(c(t))(x)
fAD_m(t) = x -> ∂t(m)(x,t) - (c(t)(x) - m(t)(x))

#model
L=1
n=2
domain = (0,L,0,L)
partition = (n,n)
order = 1
D=length(partition)
model = CartesianDiscreteModel(domain,partition)
labels=get_face_labeling(model)
add_tag_from_tags!(labels,"dirichlet",[1,2,3,4,5,6,7])
add_tag_from_tags!(labels,"neumann",[8])

Ω = Triangulation(model)
degree = 2*order
dΩ = Measure(Ω,degree)

Γn = BoundaryTriangulation(model,labels,tags="neumann")
dΓn = Measure(Γn,degree)
n_Γn = get_normal_vector(Γn)

#spaces
reffeu = ReferenceFE(lagrangian,VectorValue{D,Float64},order)
V0 = FESpace(
  model,
  reffeu,
  conformity=:H1,
  dirichlet_tags="boundary"
  )

reffep = ReferenceFE(lagrangian,Float64,order)

Q = TestFESpace(
  model,
  reffep,
  conformity=:H1,
  constraint=:zeromean)

reffew = ReferenceFE(lagrangian,Float64,order)
#reffeᵤ = ReferenceFE(lagrangian,VectorValue{2,Float64},order)

W0 = FESpace(
  model,
  reffew,
  conformity=:H1,
  #dirichlet_tags="boundary",
  dirichlet_tags="dirichlet"
)

reffeg = ReferenceFE(lagrangian,Float64,order)
G0 = TestFESpace(
  model,
  reffeg,
  conformity=:H1,
  #constraint=:zeromean,
  dirichlet_tags=["interior","dirichlet"]
  #dirichlet_tags=["interior","boundary"]
)

U = TrialFESpace(V0)
P = TrialFESpace(Q)
XINS = MultiFieldFESpace([U,P])
YINS = MultiFieldFESpace([V0,Q])
xINSh0 = interpolate_everywhere([u(0.0),p(0.0)],XINS(0.0))

C = TransientTrialFESpace(W0,c)
M = TransientTrialFESpace(G0,m)
XAD = TransientMultiFieldFESpace([C,M])
YAD = MultiFieldFESpace([W0,G0])
xADh0 = interpolate_everywhere([c(0.0),m(0.0)],XAD(0.0))

#Interior terms
m_ΩINS(ut,v) = ρ * ut⊙v
a_ΩINS(u,v) = μ * ∇(u)⊙∇(v) 
b_ΩINS(v,p) = - (∇⋅v)*p
c_ΩINS(u, v) = ρ *  v ⊙ conv(u, ∇(u))
dc_ΩINS(u, du, v) = ρ * v ⊙ dconv(du, ∇(du), u, ∇(u))

#Boundary terms 
a_ΓINS(u,v) = μ* ( - (n_Γ⋅∇(u))⋅v - u⋅(n_Γ⋅∇(v)) ) + ( γ(u)/h )*u⋅v 
b_ΓINS(v,p) = (n_Γ⋅v)*p

#PSPG 
sp_ΩINS(w,p,q)    = (ρ^(-1) * τ_PSPG(w))     *  ∇(q) ⋅ ∇(p)
st_ΩINS(w,ut,q)   = (ρ^(-1) * τ_PSPG(w)) * ρ *  ∇(q) ⋅ ut
sc_ΩINS(w,u,q)    = (ρ^(-1) * τ_PSPG(w)) * ρ *  ∇(q) ⋅ conv(u, ∇(u))
dsc_ΩINS(w,u,du,q)= (ρ^(-1) * τ_PSPG(w)) * ρ *  ∇(q) ⋅ dconv(du, ∇(du), u, ∇(u))
ϕ_ΩINS(w,q,t)     = (ρ^(-1) * τ_PSPG(w))     *  ∇(q) ⋅ f(t)

#SUPG
sp_sΩINS(w,p,v)    = τ_SUPG(w)     *  conv(w,∇(v)) ⋅ ∇(p)
st_sΩINS(w,ut,v)   = τ_SUPG(w) * ρ *  conv(w,∇(v)) ⋅ ut
sc_sΩINS(w,u,v)    = τ_SUPG(w) * ρ *  conv(w,∇(v)) ⋅ conv(u, ∇(u)) 
dsc_sΩINS(w,u,du,v)= τ_SUPG(w) * ρ *  conv(w,∇(v)) ⋅ dconv(du, ∇(du), u, ∇(u)) 
ϕ_sΩINS(w,v,t)     = τ_SUPG(w)     *  conv(w,∇(v)) ⋅ f(t)  

resINS(t,(u,p),(ut,pt),(v,q)) = 
∫( ( m_Ω(ut,v) + a_Ω(u,v) + b_Ω(v,p) + b_Ω(u,q) - v⋅f(t) + q*g(t) + c_Ω(u,v)  # + ρ * 0.5 * (∇⋅u) * u ⊙ v  
+1*(- sp_Ω(u,p,q)  -  st_Ω(u,ut,q)   + ϕ_Ω(u,q,t)     - sc_Ω(u,u,q) )
+1*(- sp_sΩ(u,p,v) - st_sΩ(u,ut,v)  + ϕ_sΩ(u,v,t)    - sc_sΩ(u,u,v) )))dΩ #+
#∫( a_Γ(u,v)+b_Γ(u,q)+b_Γ(v,p) - ud(t) ⊙(  ( γ(u)/h )*v - μ * n_Γ⋅∇(v) + q*n_Γ )  )dΓ #+
#∫(μ * - v⋅(n_Γn⋅∇(u_Γn(t))) + (n_Γn⋅v)*p_Γn(t) )dΓn + 
#∫( i_Γg(u,u,v) - j_Γg(u,p,q) )dΓg

jacINS(t,(u,p),(ut,pt),(du,dp),(v,q)) =
∫( ( a_Ω(du,v) + b_Ω(v,dp) + b_Ω(du,q)  + dc_Ω(u, du, v) # + ρ * 0.5 * (∇⋅u) * du ⊙ v 
+1* ( - sp_Ω(u,dp,q)  - dsc_Ω(u,u,du,q) )
+1*(- sp_sΩ(u,dp,v) - dsc_sΩ(u,u,du,v) ) ))dΩ #+ 
#∫( a_Γ(du,v)+b_Γ(du,q)+b_Γ(v,dp)  )dΓ #+
#∫( i_Γg(u,du,v) - j_Γg(u,dp,q) )dΓg 

jac_tINS(t,(u,p),(ut,pt),(dut,dpt),(v,q)) = 
∫(  m_Ω(dut,v) 
+1* ( - st_Ω(u,dut,q) ) 
+1* (- st_sΩ(u,dut,v) ))dΩ

#interior terms
m_ΩAD(ct,w) = ct * w 
a_ΩAD(c,w) =  ∇(c) ⋅ ∇(w) 
c_ΩAD(β , c , w ) =  β ⋅ ∇(c) ⋅ (w)

#Neumann boundary terms
m_ΓAD(mt,g) = mt * g
f_coupling(c,m)= c - m

#forcing terms
fAD_cw(t,w) = fAD_c(t)*w
fAD_mw(t,w) = (n_Γn⋅∇(c(t)))*w  + ( c(t)*w-m(t)*w )
fAD_mg(t,g) = fAD_m(t)*g

resAD(t,(c,m),(ct,mt),(w,g)) =  
∫(  m_ΩAD(ct,w) + a_ΩAD(c,w) + c_ΩAD(u(t) , c , w )       - fAD_cw(t,w) )dΩ + 
∫(  (f_coupling(c,m))*w  - fAD_mw(t,w)     +      m_ΓAD(mt,g) - f_coupling(c,m)⋅g - fAD_mg(t,g)         )dΓn

jacAD(t,(c,m),(ct,mt),(dc,dm),(w,g))  = 
∫( ( a_ΩAD(dc,w) + c_ΩAD( u(t) , dc , w ) ))dΩ + 
∫( f_coupling(dc,dm)*w     -            f_coupling(dc,dm)*g   )dΓn

jac_tAD(t,(c,m),(ct,mt),(dct,dmt),(w,g)) = 
∫(  m_ΩAD(dct,w) )dΩ + 
∫(  m_ΓAD(dmt,g)   )dΓn


function SolveNavierStokes(op)

ls = PardisoSolver(op.assem_t.matrix_type)
nls = NewtonRaphsonSolver(ls,1e-5,30)
odes = ThetaMethod(nls, dt, θ)
solver = TransientFESolver(odes)
sol_t = solve(solver, op, xh0, t0, tF)

l2(w) = w⋅w

tol = 1.0e-5
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
  @show eul2i = sqrt(sum( ∫(l2(e_u))dΩ ))
  @show epl2i = sqrt(sum( ∫(l2(e_p))dΩ ))
  @test eul2i<tol
  @test epl2i<tol
  #writevtk(Ω,"results_fm",cellfields=["e_u"=>e_u,"uh_Ω"=>uh_tn,"u_ex"=>u_ex,"e_p"=>e_p,"ph_Ω"=>ph_tn,"p_ex"=>p_ex])
  push!(eul2,eul2i)
  push!(epl2,epl2i)
  println("$(tn*100/Δt)% of timesteps complete")
  #@show Int(round(((_t_n-dt)/dt)+1,digits=7))
end

eul2=last(eul2)
epl2=last(epl2)

(eul2, epl2)

end #function INS



op = TransientFEOperator(res,jac,jac_t,X,Y)




















































op = TransientFEOperator(res,jac,jac_t,XAD,YAD)

θ = 1
t0 = 0.0
tF = 1.0
dt = 1.0

ls = LUSolver()
odes = ThetaMethod(ls,dt,θ)
solver = TransientFESolver(odes)
sol_t = solve(solver,op,xADh0,t0,tF)

l2(w) = w⋅ w

tol = 1.0e-6
_t_n = t0

for (xh_tn, tn) in sol_t
  global _t_n
  _t_n += dt
  ec = c(tn) - xh_tn[1]
  ecl2 = sqrt(sum( ∫(l2(ec))dΩ ))
  @test ecl2 < tol
  em = m(tn) - xh_tn[2]
  eml2 = sqrt(sum( ∫(l2(em))dΩ ))
  @test eml2 < tol
end

end #module