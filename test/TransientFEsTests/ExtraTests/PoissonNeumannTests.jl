module PoissonTests

using Test
using Gridap
import Gridap: ∇
using LinearAlgebra
using WriteVTK

domain = (0,1,0,1)
n=5
partition = (n,n)
#model = simplexify(CartesianDiscreteModel(domain,partition))
model = (CartesianDiscreteModel(domain,partition))

order = 1

labels = get_face_labeling(model)

add_tag_from_tags!(labels,"dirichlet",[1,2,3,4,5,6,7])
add_tag_from_tags!(labels,"neumann",[8])
writevtk(model,"model")

trian = get_triangulation(model)
degree = 2*order
quad = CellQuadrature(trian,degree)

ntrian = BoundaryTriangulation(model,labels,"neumann")
ndegree = order
nquad = CellQuadrature(ntrian,ndegree)
const nn = get_normal_vector(ntrian)

# Using automatic differentiation
u(x) = 3*(x[1] + x[2])
f(x) = - Δ(u)(x)
T = Float64

V = TestFESpace(
model=model,
order=order,
reffe=:Lagrangian,
labels=labels,
valuetype=T,
#dirichlet_tags="boundary")
dirichlet_tags="dirichlet")

U = TrialFESpace(V,u)

uh = interpolate(u, U)
uh_Γn = restrict(uh,ntrian)

a(u,v) = ∇(v)⊙∇(u)
l(v) = v⊙f
t_Ω = AffineFETerm(a,l,trian,quad)

a_Γn(u,v) = 0*a(u,v)
l_Γn(v) = v⊙(nn⋅∇(uh_Γn))
t_Γn = AffineFETerm(a_Γn,l_Γn,ntrian,nquad)

t_Γn2 = FESource(l_Γn,ntrian,nquad)

op = AffineFEOperator(U,V,t_Ω,t_Γn2)

uh = solve(op)

e = u - uh

writevtk(model,"model",)
writevtk(trian,"trian",cellfields=["uh"=>uh,"e"=>e])
l2(u) = inner(u,u)
sh1(u) = a(u,u)
h1(u) = sh1(u) + l2(u)

el2 = sqrt(sum( integrate(l2(e),trian,quad) ))
eh1 = sqrt(sum( integrate(h1(e),trian,quad) ))
ul2 = sqrt(sum( integrate(l2(uh),trian,quad) ))
uh1 = sqrt(sum( integrate(h1(uh),trian,quad) ))

@test el2< 1.e-8

end # module