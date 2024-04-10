# Functions developed in the University of Bristol module EMAT30008: Scientific Computing 
# Author: Ben Winstanley, yy23737@bristol.ac.uk

# TODO: go through euler_step and solve_to and reformat

# IMPORTS
import numpy as np
from scipy.optimize import root
from typing import Literal

# Week 14
def euler_step(ode_func, x: np.ndarray, t: float, h: float) -> np.ndarray:
    '''Perform single Euler step from `x` with step size `h` for the ODE `x' = ode_func(x,t)`. '''
    x_next = x + h*ode_func(x,t)
    return x_next


def solve_to(ode_func, x_init: np.ndarray, t_init: float, t_final: float, deltat_max: float, method: Literal['Euler', 'RK4', 'RK2'] = 'RK4') -> np.ndarray:
    '''
    Solves ODE IVP problem `x' = f(x,t), x(t_init)=x_init` until time `t_final`.

    Uses either 'Euler' time step method, or 'RK4' 4th order Runge-Kutta method (default)

    -----
    Parameters
    -----
    ode_func : function
        Definition function for the RHS of the ODE `x' = ode_func(x,t)`.
    x_init : 1-D Numpy array of floats
        Initial condition for the ODE, `x(t_init)=x_init`.
    t_init, t_final : floats
        Initial and final times.
    deltat_max : float
        Step size used in numerical timestepping, some final steps may be smaller than this.
    method : either 'Euler','RK4' or 'RK2', default 'RK4'
        Specifies the numerical timestepping method used.

    ------
    Returns
    ------
    2-D Numpy array
        Full timeseries solution of ODE from t_init to t_final.
        Columns `[t, x1, x2, ... ,xn]`, with each row containing the time value after a timestep
        and the values of each state variable in `x=[x1, x2, ... ,xn]`.

    -----
    Example
    -----
    >>> import numpy as np
    >>> def shm(x, t):
            return np.array(([x[1], -x[0]]))
    >>> solve_to(shm, np.array(([5,0])), 0, 5, 0.01, 'RK4')
    array([[ 0.        ,  5.        ,  0.        ],
       [ 0.01      ,  4.99975   , -0.04999917],
       [ 0.02      ,  4.99900003, -0.09999333],
       ...,
       [ 4.98      ,  1.32214124,  4.82202681],
       [ 4.99      ,  1.3702946 ,  4.80856452],
       [ 5.        ,  1.41831093,  4.79462137]])

    -----
    Notes
    -----
    The different methods timestep in the following ways from `x` at time `t` to
    `x_next` with a timestep h <= deltat_max:
    
    Euler timestep 'Euler' : 
        `x_next = x + h * ode_func(x,t)`.

    Fourth Order Runge-Kutta 'RK4' :
        `x_next = x + (h/6) * (k1 + 2k2 + 2k3 + k4)`,

        where `k1 = ode_func(x,t)`, `k2 = ode_func((x+(h/2)*k1),(t+h/2))`,
         `k3 = ode_func((x+(h/2)*k2),(t+h/2))`, `k4 = ode_func((x+h*k3),(t+h))`.

    Second Order Runge-Kutta 'RK2' :
        `x_next = x + h*k2`,

        where `k1 = ode_func(x,t)`, `k2 = ode_func((x+(h/2)*k1),(t+h/2))`.

    -----
    Raises
    -----
    Exception
        If the timestep size `deltat_max` is larger than the total time interval `t_final-t_init`.
    Exception
        If `t_init > t_final`.
    Exception
        If the `method` string supplied isn't in the list provided.

    -----
    See also
    -----
    euler_step
        Performs a single euler timestep
    limit_cycle_condition
        Uses this function to search for limit cycle solutions of an ODE

    '''
    # TODO: could combine some of the code that all timestepping methods share

    if deltat_max >= (t_final - t_init):
        raise Exception("Input Error: Maximum time-step deltat_max >= total time interval.")
    
    if t_init >= t_final:
        raise Exception("Input Error: t_init >= t_final.")

    if method=='Euler': # Euler time-step method

        # Time intervals are constant, use step size h = deltat_max
        h = deltat_max
        # Take steps of h until final time value is just less than t_final
        t_vals = np.arange(t_init, t_final - h/100, h)

        # Initialise x store
        x_store, x = x_init, x_init

        # Loop through each timestep
        for t in t_vals[:-1]:
            # Update x value and x store
            x = euler_step(ode_func, x, t, h)
            x_store = np.vstack((x_store, x))

        # Take final step with h <= deltat_max
        h_final = t_final - t_vals[-1]
        x = euler_step(ode_func, x, t, h_final)
        x_store = np.vstack((x_store, x))
        t_vals = np.append(t_vals, t_final)

        # Combine into one store
        store = np.concatenate((np.array([t_vals]).T, x_store), axis=1)

        return store
    
    elif method=='RK4': # 4th order Runge-Kutta method

        # Time intervals are constant, use step size h = deltat_max
        h = deltat_max
        # Take steps of h until final time value is just less than t_final
        t_vals = np.arange(t_init, t_final - h/100, h)

        # Initialise x store
        x_store, x = x_init, x_init

        # TODO: make below into a rk4_step function
        # Loop through each timestep
        for t in t_vals[:-1]:
            # Calculate k values:
            k1 = ode_func(x,t)
            k2 = ode_func((x+(h/2)*k1),(t+h/2))
            k3 = ode_func((x+(h/2)*k2),(t+h/2))
            k4 = ode_func((x+h*k3),(t+h))

            # Calculate next value
            x = x + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

            # Update store
            x_store = np.vstack((x_store, x))
        
        # Take final step with h <= deltat_max
        h_final = t_final - t_vals[-1]

        k1 = ode_func(x,t)
        k2 = ode_func((x+(h_final/2)*k1),(t+h_final/2))
        k3 = ode_func((x+(h_final/2)*k2),(t+h_final/2))
        k4 = ode_func((x+h_final*k3),(t+h_final))

        x = x + (h_final/6)*(k1 + 2*k2 + 2*k3 + k4)

        # Update store
        x_store = np.vstack((x_store, x))
        t_vals = np.append(t_vals, t_final)

        # Combine into one store
        store = np.concatenate((np.array([t_vals]).T, x_store), axis=1)

        return store
    
    elif method=='RK2': # 2nd order Runge-Kutta method

        # Time intervals are constant, use step size h = deltat_max
        h = deltat_max
        # Take steps of h until final time value is just less than t_final
        t_vals = np.arange(t_init, t_final - h/100, h)

        # Initialise x store
        x_store, x = x_init, x_init

        # TODO: make below into a rk2 step function
        # Loop through each timestep
        for t in t_vals[:-1]:
            # Calculate k values:
            k1 = ode_func(x,t)
            k2 = ode_func((x+(h/2)*k1),(t+h/2))

            # Calculate next value
            x = x + k2*h

            # Update store
            x_store = np.vstack((x_store, x))

        # Take final step with h <= deltat_max
        h_final = t_final - t_vals[-1]
        k1 = ode_func(x,t)
        k2 = ode_func((x+(h_final/2)*k1),(t+h_final/2))
        x = x + k2*h_final

        # Update store
        x_store = np.vstack((x_store, x))

        t_vals = np.append(t_vals, t_final)

        # Combine into one store
        store = np.concatenate((np.array([t_vals]).T, x_store), axis=1)

        return store

    else:
        raise Exception("Not a valid method, please enter 'Euler','RK4' or 'RK2 ")


# Week 15

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
        Definition function for the RHS of the ODE `x' = ode_func(x,t)`.
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
            

        
