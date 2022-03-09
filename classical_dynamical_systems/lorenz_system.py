# -*- coding: utf-8 -*-
# # Lorenz system

# The Lorenz system, originally intended as a simplified model of atmospheric convection, has instead become a standard example of sensitive dependence on initial conditions; that is, tiny differences in the starting condition for the system rapidly become magnified. The system also exhibits what is known as the "Lorenz attractor", that is, the collection of trajectories for different starting points tends to approach a peculiar butterfly-shaped region.
#
# The Lorenz system includes three ordinary differential equations:
#
# ```{code}
# dx/dt = sigma ( y - x )
# dy/dt = x ( rho - z ) - y
# dz/dt = xy - beta z
# ```
#       
# where the parameters beta, rho and sigma are usually assumed to be positive. The classic case uses the parameter values
#
# ```{code}
# beta = 8 / 3
# rho = 28
# sigma = 10
# ```

import brainpy as bp
import matplotlib.pyplot as plt

assert bp.__version__ >= '2.0.2'

sigma = 10
beta = 8 / 3
rho = 28

dx = lambda x, t, y: sigma * (y - x)
dy = lambda y, t, x, z: x * (rho - z) - y
dz = lambda z, t, x, y: x * y - beta * z

integral = bp.odeint(bp.JointEq([dx, dy, dz]), method='exp_auto')

runner = bp.integrators.IntegratorRunner(integral, monitors=['x', 'y', 'z'],
                                         inits=dict(x=8, y=1, z=1), dt=0.01)
runner.run(100)

fig = plt.figure()
fig.add_subplot(111, projection='3d')
plt.plot(runner.mon.x[100:, 0], runner.mon.y[100:, 0], runner.mon.z[100:, 0])
plt.show()

