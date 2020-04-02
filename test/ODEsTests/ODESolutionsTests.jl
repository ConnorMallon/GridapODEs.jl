module ODESolversTests

using GridapTimeStepper.ODETools: GenericODESolution
using GridapTimeStepper.ODETools: BackwardEuler
using GridapTimeStepper.ODETools: BackwardEulerNonLinearOperator
using GridapTimeStepper.ODETools: solve!

using Test
using Gridap
using GridapTimeStepper
using GridapTimeStepper.ODETools


include("ODEOperatorMocks.jl")

op = ODEOperatorMock(1.0,0.0,1.0)

include("ODESolverMocks.jl")

t0 = 0.0
tF = 1.0
dt = 0.1

u0 = ones(2)*2

nls = NLSolverMock()

solver = BackwardEuler(nls,dt)

steps = solve(solver,op,u0,t0,tF)

uf = copy(u0)
uf.=1.0
current, state = Base.iterate(steps)
uf, tf = current
uf, u0, tf, cache = state
cache
@test tf==t0+dt
@test all(uf.≈1+11/9)
# current, state = Base.iterate(steps)
current, state = Base.iterate(steps,state)
uf, tf = current
@test tf≈t0+2*dt
uf, u0, tf, cache = state

_t_n = t0
for (u_n, t_n) in steps
  global _t_n
  _t_n += dt
  @test t_n≈_t_n
end

steps

test_ode_solution(steps)
# println("The solution at time $(t_n) is $(u_n)")

end
