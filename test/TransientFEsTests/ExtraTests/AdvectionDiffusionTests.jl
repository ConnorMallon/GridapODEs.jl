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

#=

u(x,t) = (x[1]-x[2])*t
#u(x,t) = VectorValue(-x[2],x[1])*(t)
u(t::Real) = x -> u(x,t)

p(x,t) = (x[1]-x[2])*t
p(t::Real) = x -> p(x,t)
q(x) = t -> p(x,t)

f(t) = x -> ∂t(u)(t)(x) - Δ(u(t))(x) + ∇(p(t))(x)
g(t) = x -> (∇⋅u(t))(x)
=#



β(x,t) = VectorValue( x[1] , -x[2] ) 
β(t::Real) = x -> β(x,t)
βd(t) = x -> β(t)(x)

c(x,t)= (x[1]*x[2])*t
m(x,t) = (x[1]*x[2]+x[2])*t  #we can choose m such that ∇c.n-f_c=0 --> m = c-∇c.n on Γ 

c(t::Real) = x -> c(x,t)
m(t::Real) = x -> m(x,t)

fAD_c(t) = x -> ∂t(c)(x,t) - Δ(c(t))(x) +  β(t)(x)⋅∇(c(t))(x)
fAD_m(t) = x -> ∂t(m)(x,t) - (c(t)(x) - m(t)(x))



L=1
n=2
domain = (0,L,0,L)
partition = (n,n)

order = 1
D=length(partition)

#Bulk space
model = CartesianDiscreteModel(domain,partition)
labels=get_face_labeling(model)
add_tag_from_tags!(labels,"dirichlet",[1,2,3,4,5,6,7])
add_tag_from_tags!(labels,"neumann",[8])

writevtk(model,"model")

order=1
reffeᵤ = ReferenceFE(lagrangian,Float64,order)
#reffeᵤ = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
V0 = FESpace(
  model,
  reffeᵤ,
  conformity=:H1,
  #dirichlet_tags="boundary",
  dirichlet_tags="dirichlet"
)

reffeₚ = ReferenceFE(lagrangian,Float64,order)
Q = TestFESpace(
  model,
  reffeₚ,
  conformity=:H1,
  #constraint=:zeromean,
  dirichlet_tags=["interior","dirichlet"]
  #dirichlet_tags=["interior","boundary"]
  
)

U = TransientTrialFESpace(V0,c)
P = TransientTrialFESpace(Q,m)

X = TransientMultiFieldFESpace([U,P])
Y = MultiFieldFESpace([V0,Q])

XAD=X
YAD=Y

U0 = U(0.0)
P0 = P(0.0)
X0 = X(0.0)
uh0 = interpolate_everywhere(c(0.0),U0)
ph0 = interpolate_everywhere(m(0.0),P0)
xh0 = interpolate_everywhere([uh0,ph0],X0)
xADh0=xh0

######################

Ω = Triangulation(model)
degree = 2*order
dΩ = Measure(Ω,degree)
#Γ = BoundaryTriangulation(model)
#dΓ = Measure(Γ,degree)
#n_Γ = get_normal_vector(Γ)


Γn = BoundaryTriangulation(model,labels,tags="neumann")
Γd = BoundaryTriangulation(model,labels,tags="dirichlet")
dΓn = Measure(Γn,degree)
dΓd = Measure(Γd,degree)
n_Γn = get_normal_vector(Γn)
n_Γd = get_normal_vector(Γd)



h = L / n
γ = order*(order+1)

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


res(t,(c,m),(ct,mt),(w,g)) =  
∫(  m_ΩAD(ct,w) + a_ΩAD(c,w) + c_ΩAD(β(t) , c , w )       - fAD_cw(t,w) )dΩ + 
#∫(  (γ/h)*w⋅c  - w⋅(n_Γd⋅∇(c)) - (n_Γd⋅∇(w))⋅c - (γ/h)*w⋅cd(t) + (n_Γd⋅∇(w))⋅cd(t) )dΓd +
∫(  (f_coupling(c,m))*w  - fAD_mw(t,w)     +      m_ΓAD(mt,g) - f_coupling(c,m)⋅g - fAD_mg(t,g)         )dΓn

jac(t,(c,m),(ct,mt),(dc,dm),(w,g))  = 
∫( ( a_ΩAD(dc,w) + c_ΩAD(β(t) , dc , w ) ))dΩ + 
#∫(  (γ/h)*w⋅dc  - w⋅(n_Γd⋅∇(dc)) - (n_Γd ⋅∇(w))⋅dc   )dΓd + 
∫( f_coupling(dc,dm)*w     -            f_coupling(dc,dm)*g   )dΓn

jac_t(t,(c,m),(ct,mt),(dct,dmt),(w,g)) = 
∫(  m_ΩAD(dct,w) )dΩ + 
∫(  m_ΓAD(dmt,g)   )dΓn

op = TransientFEOperator(res,jac,jac_t,XAD,YAD)

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