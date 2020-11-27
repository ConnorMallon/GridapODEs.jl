module Poisson1DTests

using Gridap
using Test

u(x) = x[1] + 1
∇u(x) = ∇(u)(x)
f(x) = -Δ(u)(x)

domain = (0,3)
cells = (3,)
model = CartesianDiscreteModel(domain,cells)

order = 1

labels = get_face_labeling(model)

V = FESpace(
  model=model,
  reffe=:Lagrangian,
  order=order,
  valuetype=Float64,
  conformity=:H1,
  dirichlet_tags=2)

U = TrialFESpace(V,u)

degree = 2*order
trian = Triangulation(model) 
quad = CellQuadrature(trian,degree)

btrian = BoundaryTriangulation(model,1)
bquad = CellQuadrature(btrian,degree)
nb = get_normal_vector(btrian)

a(u,v) = ∇(v)⋅∇(u)
l(v) = v*f
t_Ω = AffineFETerm(a,l,trian,quad)

l_b(v) = v*(nb⋅∇u)
t_b = FESource(l_b,btrian,bquad)

op = AffineFEOperator(U,V,t_Ω,t_b)
uh = solve(op)

e = u - uh
l2(u) = inner(u,u)
el2 = sqrt(sum( integrate(l2(e),trian,quad) ))
@test el2 < 1.e-6

end # module