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



#manufactured solutions
β(x,t) = VectorValue( x[1] , -x[2] ) 
β(t::Real) = x -> β(x,t)
βd(t) = x -> β(t)(x)

c(x,t)= (x[1]*x[2])*t
c(t::Real) = x -> c(x,t)

m(x,t) = (x[1]*x[2]+x[2])*t  #we can choose m such that ∇c.n-f_c=0 --> m = c-∇c.n on Γ 
m(t::Real) = x -> m(x,t)

fAD_c(t) = x -> ∂t(c)(x,t) - Δ(c(t))(x) +  β(t)(x)⋅∇(c(t))(x)
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
reffeᵤ = ReferenceFE(lagrangian,Float64,order)
#reffeᵤ = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
W0 = FESpace(
  model,
  reffeᵤ,
  conformity=:H1,
  #dirichlet_tags="boundary",
  dirichlet_tags="dirichlet"
)
reffeₚ = ReferenceFE(lagrangian,Float64,order)
G0 = TestFESpace(
  model,
  reffeₚ,
  conformity=:H1,
  #constraint=:zeromean,
  dirichlet_tags=["interior","dirichlet"]
  #dirichlet_tags=["interior","boundary"]
  
)
C = TransientTrialFESpace(W0,c)
M = TransientTrialFESpace(G0,m)
XAD = TransientMultiFieldFESpace([C,M])
YAD = MultiFieldFESpace([W0,G0])
xADh0 = interpolate_everywhere([c(0.0),m(0.0)],XAD(0.0))

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


res(t,((c,m),(ct,mt)),(w,g)) =  
∫(  m_ΩAD(ct,w) + a_ΩAD(c,w) + c_ΩAD(β(t) , c , w )       - fAD_cw(t,w) )dΩ + 
∫(  (f_coupling(c,m))*w  - fAD_mw(t,w)     +      m_ΓAD(mt,g) - f_coupling(c,m)⋅g - fAD_mg(t,g)         )dΓn

jac(t,((c,m),(ct,mt)),(dc,dm),(w,g))  = 
∫( ( a_ΩAD(dc,w) + c_ΩAD(β(t) , dc , w ) ))dΩ + 
∫( f_coupling(dc,dm)*w     -            f_coupling(dc,dm)*g   )dΓn

jac_t(t,((c,m),(ct,mt)),(dct,dmt),(w,g)) = 
∫(  m_ΩAD(dct,w) )dΩ + 
∫(  m_ΓAD(dmt,g)   )dΓn

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