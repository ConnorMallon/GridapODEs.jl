module StokesTaylorHoodTests


using Test
using Gridap
import Gridap: ∇

using LinearAlgebra: tr, ⋅

# Using automatic differentiation
u(x) = VectorValue( x[1]^2 + 2*x[2]^2, -x[1]^2 )
p(x) = x[1] - x[2]
f(x) = -Δ(u)(x) + ∇(p)(x)
g(x) = (∇⋅u)(x)
∇u(x) = ∇(u)(x)

domain = (0,2,0,2)
partition = (3,3)
model = CartesianDiscreteModel(domain,partition)

labels = get_face_labeling(model)
add_tag_from_tags!(labels,"dirichlet",[1,2,3,4,5,6,7])
add_tag_from_tags!(labels,"neumann",[8])

order = 2

V = TestFESpace(
model=model,
order=order,
reffe=:Lagrangian,
labels=labels,
valuetype=VectorValue{2,Float64},
dirichlet_tags="dirichlet",
conformity=:H1)

Q = TestFESpace(
model=model,
order=order-1,
reffe=:Lagrangian,
valuetype=Float64,
conformity=:H1,
constraint=:zeromean)

U = TrialFESpace(V,u)
P = TrialFESpace(Q)

Y = MultiFieldFESpace([V,Q])
X = MultiFieldFESpace([U,P])

trian = get_triangulation(model)
degree = order
quad = CellQuadrature(trian,degree)

btrian = BoundaryTriangulation(model,labels,"neumann")
bdegree = order
bquad = CellQuadrature(btrian,bdegree)
n = get_normal_vector(btrian)

function a(x,y)
u,p = x
v,q = y
∇(v)⊙∇(u) - (∇⋅v)*p + q*(∇⋅u)
end

function l(y)
v,q = y
v⋅f + q*g
end

function l_Γb(y)
v,q = y
v⋅(n⋅∇u) - (n⋅v)*p
end

t_Ω = AffineFETerm(a,l,trian,quad)
t_Γb = FESource(l_Γb,btrian,bquad)

op = AffineFEOperator(X,Y,t_Ω,t_Γb)

uh, ph = solve(op)

eu = u - uh
ep = p - ph

l2(v) = v⋅v
h1(v) = v⋅v + ∇(v)⊙∇(v)

eu_l2 = sqrt(sum(integrate(l2(eu),trian,quad)))
eu_h1 = sqrt(sum(integrate(h1(eu),trian,quad)))
ep_l2 = sqrt(sum(integrate(l2(ep),trian,quad)))

tol = 1.0e-9
@test eu_l2 < tol
@test eu_h1 < tol
@test ep_l2 < tol


end # module