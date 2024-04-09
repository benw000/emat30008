 # week15funcs.py
# Week 15 of EMAT30008
# Used in week15exercises.ipynb
# Author: Ben Winstanley, yy23737@bristol.ac.uk


# NO LONGER USED
# Functions below use other functions from week14funcs.py within a parallel folder
# To avoid lots of code calling between folders we put this all within one module
# See scicompfunctions.py



# IMPORTS
import numpy as np
from scipy.optimize import root
from typing import Literal


def limit_cycle_condition(ode_func,
                          params: np.ndarray, 
                          num_loops_needed: int = 10,
                          phase_condition: Literal['constant', 'derivative']='derivative',
                          constant_value: float = None,
                          deltat_max: float = 0.1):
    ''' 
    Computes the pair `G(u0,T), phi(u0)` used to search for limit cycle solutions of ODEs.

    A limit cycle will have initial state `u0` and period `T` such that this function returns zeros.

    -----
    Parameters
    -----
    ode_func : function
        Definition function for the RHS of the ODE.
    params : 1-D Numpy array of floats
        In format `[T, u0]`. Here `T` (float) is the period of the limit cycle, and `u0`
         (array of floats) is the initial point along the limit cycle.
    num_loops_needed : int, default 10
        Determines how many successive loops are checked to conclude the initial state
         `u0` generates a limit cycle.
    phase_condition : 'constant' or 'derivative' 
        Specifies the type of phase condition used
         to select a distinct limit cycle, methods specified below.
    constant_value : float
        Value that the first state variable of `u0` should have. Must be supplied
         if phase_condition == 'constant'.
    deltat_max : float
        Step size used by solve_to numerical ODE solver to compute the solution of the  
         supplied ODE starting from `u0`.

    ------
    Returns
    ------
    1-D Numpy array 
        `[G, phi]`, where the length of `G` is the number of state variables in the 
         ODE function, `phi` is a single float.
    
    -----
    Example
    -----
    >>> import numpy as np
    >>> def shm(x, t):
            return np.array(([x[1], -x[0]]))
    >>> limit_cycle_condition(shm, np.array(([2*np.pi,1,0])), 5)
    array([ 4.32329707e-07, -5.18133545e-06,  8.67617365e-07, -1.03901950e-05,
        1.30414748e-06, -1.56132187e-05,  1.74110099e-06, -2.08422900e-05,
        2.17816753e-06, -2.60732268e-05,  0.00000000e+00])


    ------
    Notes
    ------
    G is the element-wise difference between our starting point u0 and the end of its trajectory
    after time T, uT. When G is zero this means the solution has returned to its starting point
    u0 after time T, and our solution is thus periodic.

    phi is the phase condition, used to set the phase and thus choose a periodic orbit from the
    family of orbits generated when G=0. 

    If phase_condition == 'constant' then we supply a 'constant_value', which the first state
    variable of the ODE must attain at time t=0. We set phi = constant_value - u0[0]

    If phase_condition == 'derivative' then we compute the derivative of the first state 
    variable at time t=0. Every limit cycle should contain a point where the first state variable
    has a turning point (or is constant). We set phi = d/dt[u[t=0]]

    -----
    Raises
    -----
    Exception
        if phase_condition == 'constant' but no constant value is supplied.

    -----
    See also
    -----
    find_limit_cycle
        Function that finds limit cycles by using scipy.optimize.root to find a
         root of limit_cycle_condition.

    '''
    if phase_condition == 'constant' and constant_value == None:
        raise Exception("Error: Please supply a starting value that the first state variable must attain")

    # Extract T, u0 from params
    T, u0 = params[0], params[1:]

    # Establish empty array to hold Gs
    num_variables = len(u0)
    G_collection = np.zeros([num_loops_needed*num_variables])

    # Loop over number of loops checked
    for i in range(num_loops_needed):
        # Compute G by calling the solver to solve until time (i+1)*T. Solve with RK4 and deltat_max supplied
        solution = solve_to(f=ode_func, x_init=u0, t_init=0, 
                            t_final=(i+1)*T, deltat_max=deltat_max, method='RK4')
        uT = solution[-1,1:]
        G = u0 - uT
        G_collection[i*num_variables:(i+1)*num_variables] = G
    
    # Compute phi 
    if phase_condition == 'constant':
        # Compute difference between first state variable at time t=0 and constant_value
        phi = u0[0] - constant_value
    elif phase_condition == 'derivative':
        # Compute derivative of first state variable at time t=0
        u0dot = ode_func(u0, 0)
        phi = u0dot[0]

    return np.append(G_collection,[phi])

def find_limit_cycle(ode_func, 
                     init_point_guess: np.ndarray,
                     init_period_guess: float,
                     num_loops_needed: int = 1,
                     phase_condition: Literal['constant', 'derivative']='derivative',
                     constant_value: float = None,
                     deltat_max: float = 0.1,
                     print_findings: bool = True):
    '''
    Takes in an ODE definition function, and an initial guess for the period and starting state
    of a limit cycle of that ODE. Uses scipy.optimize.root with limit_cycle_condition to converge
    towards a limit cycle starting with the supplied guess. If convergence is successful, returns
    the period and starting state of the limit cycle it located.
    --------------
    INPUTS
    ode_func: Definition function for the RHS of the ODE at position x, time t

    init_point_guess: 1-D Numpy array, initial guess for the starting state of a limit cycle

    init_period_guess: Float, initial guess for the period of the limit cycle

    phase_condition: String, either be 'constant' or 'derivative'

    constant_value: Float value that the first state variable should start from. Must be supplied
    if phase_condition == 'constant'

    deltat_max: Float value step size used by solve_to solver to compute the solution after time T

    print_findings: Boolean value, if True then this function prints out if the convergence was
    successful, and if so the period and starting state of the limit cycle

    RETURNS
    if convergence:
        best_period: float, the period of the found limit cycle
        best_point: float, the starting state of the found limit cycle
    else:
        None
    '''
    # Establish lambda function for use with scipy.optimize.root, means we only vary params
    specific_condition = lambda params: limit_cycle_condition(ode_func=ode_func,
                                                              num_loops_needed=num_loops_needed,
                                                              params = params,
                                                              phase_condition=phase_condition,
                                                              constant_value=constant_value,
                                                              deltat_max=deltat_max)


    # Pack init_point_guess and init_period_guess into params
    init_params = np.concatenate(([init_period_guess], init_point_guess))

    result = root(specific_condition, init_params, method='lm')

    if result.success:
        best_period, best_point = result.x[0], result.x[1:]
        if print_findings:
            print("A limit cycle was found:")
            print("Period:", best_period, ",")
            print("Starting state:", best_point, ".")
        return best_period, best_point
    else:
        if print_findings:
            print("No limit cycle was found (failed to converge).")
        return None, None