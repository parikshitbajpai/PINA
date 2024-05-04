""" Heat Equation """

import torch
from pina.problem import SpatialProblem, TimeDependentProblem
from pina.operators import laplacian, grad, div
from pina import Condition, LabelTensor
from pina.geometry import CartesianDomain
from pina.equation import SystemEquation, Equation

# ===================================================== #
#                                                       #
#  This script implements the two dimensional           #
#  heat equation. The Heat class is defined             #
#  inheriting from SpatialProblem. We denote:           #
#           u --> field variable                        #
#           x,y,z --> spatial variables                 #
#           t --> temporal variables                    #
#                                                       #
# ===================================================== #

class Heat(TimeDependentProblem, SpatialProblem):

    # assign output/ spatial variables
    output_variables = ['u']
    spatial_domain = CartesianDomain({'x': [-2, 2], 'y': [-1, 1]})
    temporal_domain = CartesianDomain({'t': [0, 10]})
    # define the momentum equation
    def heat_equation(input_, output_):
        u_t = grad(output_, input_, components=['u'], d=['t'])
        nabla_u = laplacian(output_, input_, components=['u'], d=['x', 'y', 'z'])
        return nabla_u - u_t

    def continuity(input_, output_):
        return div(output_.extract(['ux', 'uy']), input_)

    # define the inlet velocity
    def inlet(input_, output_):
        value = 2 * (1 - input_.extract(['y'])**2)
        return output_.extract(['ux']) - value

    # define the outlet pressure
    def outlet(input_, output_):
        value = 0.0
        return output_.extract(['p']) - value

    # define the wall condition
    def wall(input_, output_):
        value = 0.0
        return output_.extract(['ux', 'uy']) - value

    # problem condition statement
    conditions = {
        'gamma_top': Condition(location=CartesianDomain({'x': [-2, 2], 'y':  1}), equation=Equation(wall)),
        'gamma_bot': Condition(location=CartesianDomain({'x': [-2, 2], 'y': -1}), equation=Equation(wall)),
        'gamma_out': Condition(location=CartesianDomain({'x':  2, 'y': [-1, 1]}), equation=Equation(outlet)),
        'gamma_in':  Condition(location=CartesianDomain({'x': -2, 'y': [-1, 1]}), equation=Equation(inlet)),
        'D': Condition(location=CartesianDomain({'x': [-2, 2], 'y': [-1, 1]}), equation=SystemEquation([momentum, continuity]))
    }
