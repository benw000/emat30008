# Functions developed in the University of Bristol module EMAT30008: Scientific Computing 
# Author: Ben Winstanley, yy23737@bristol.ac.uk

# TODO: go through euler_step and solve_to and reformat

# IMPORTS
import numpy as np
from scipy.optimize import minimize, root
from typing import Literal

# Week 14
def euler_step(ode_func,
               params: list[float],
               x: np.ndarray, 
               t: float, 
               h: float) -> np.ndarray:
    '''Perform single Euler step from `x` with step size `h` for the ODE `x' = ode_func(params,x,t)`. '''
    x_next = x + h*ode_func(params,x,t)
    return x_next

def rk4_step(ode_func, 
             params: list[float],
             x: np.ndarray, 
             t: float, 
             h: float) -> np.ndarray:
    '''Perform single RK4 step from `x` with step size `h` for the ODE `x' = ode_func(params,x,t)`. '''
    k1 = ode_func(params,x,t,)
    k2 = ode_func(params,(x+(h/2)*k1),(t+h/2))
    k3 = ode_func(params,(x+(h/2)*k2),(t+h/2))
    k4 = ode_func(params,(x+h*k3),(t+h))

    x_next = x + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    return x_next

def rk2_step(ode_func, 
             params: list[float],
             x: np.ndarray, 
             t: float,
             h: float) -> np.ndarray:
    '''Perform single RK2 step from `x` with step size `h` for the ODE `x' = ode_func(params, x,t)`. '''
    k1 = ode_func(params,x,t)
    k2 = ode_func(params,(x+(h/2)*k1),(t+h/2))

    x_next = x + h*k2
    return x_next


def solve_to(ode_func, 
             params: list[float],
             x_init: np.ndarray,
             t_init: float, 
             t_final: float, 
             deltat_max: float, 
             method: Literal['Euler', 'RK4', 'RK2'] = 'RK4') -> np.ndarray:
    '''
    Solves ODE IVP problem `x' = f(params,x,t), x(t_init)=x_init` until time `t_final`.

    Uses either 'Euler' time step method, or 'RK4' 4th order Runge-Kutta method (default)

    -----
    Parameters
    -----
    ode_func : function
        Definition function for the RHS of the ODE `x' = ode_func(params,x,t)`.
    params : list of floats
        Parameters passed into the ODE function `ode_func`.
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
    >>> def shm(params,x, t):
            return np.array(([x[1], -x[0]]))
    >>> solve_to(shm, None, np.array(([5,0])), 0, 5, 0.01, 'RK4')
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
        `x_next = x + h * ode_func(params,x,t)`.

    Fourth Order Runge-Kutta 'RK4' :
        `x_next = x + (h/6) * (k1 + 2k2 + 2k3 + k4)`,

        where `k1 = ode_func(params,x,t)`, `k2 = ode_func(params,(x+(h/2)*k1),(t+h/2))`,
         `k3 = ode_func(params,(x+(h/2)*k2),(t+h/2))`, `k4 = ode_func(params,(x+h*k3),(t+h))`.

    Second Order Runge-Kutta 'RK2' :
        `x_next = x + h*k2`,

        where `k1 = ode_func(params,x,t)`, `k2 = ode_func(params,(x+(h/2)*k1),(t+h/2))`.

    -----
    Raises
    -----
    Exception
        If deltat_max <= 0.
    Exception
        If the timestep size `deltat_max` is larger than the total time interval `t_final-t_init`.
    Exception
        If `t_init > t_final`.
    Exception
        If the `method` string supplied isn't in the list provided.
    Exception
        If the dimension of the initial condition doesn't match the dimension
        of the output of ode_func.

    -----
    See also
    -----
    euler_step, rk4_step, rk2_step
        Performs a single timestep for each method.
    limit_cycle_condition
        Uses this function to search for limit cycle solutions of an ODE.

    '''
    # Error messages
    if deltat_max <= 0:
        raise Exception("Input Error: Please specify a positive step size.")
    if deltat_max >= (t_final - t_init):
        raise Exception("Input Error: Maximum time-step deltat_max >= total time interval.")
    if t_init >= t_final:
        raise Exception("Input Error: t_init >= t_final.")
    if not (method in ['Euler', 'RK4', 'RK2']):
        raise Exception("Invalid Method: Please choose 'Euler', 'RK4' or 'RK2'.")
    if len(ode_func(params,x_init, t_init)) != len(x_init):
        raise Exception("ODE function and initial condition dimensions do not match.")

    # Time intervals are constant, use step size h = deltat_max
    h = deltat_max
    # Take steps of h until final time value is just less than t_final
    t_vals = np.arange(t_init, t_final - h/100, h)

    # Initialise x store
    x_store, x = x_init, x_init

    # Choose timestepping method as lambda function
    if method=='Euler': # Euler time-step method
        time_step = lambda x, t, h: euler_step(ode_func, params, x, t, h)
    elif method=='RK4': # 4th Order Runge-Kutta method
        time_step = lambda x, t, h: rk4_step(ode_func, params, x, t, h)
    elif method=='RK2': # 2th Order Runge-Kutta method
        time_step = lambda x, t, h: rk2_step(ode_func, params, x, t, h)
    
    # Loop through each timestep
    for t in t_vals[:-1]:
        # Update x value and x store
        x = time_step(x, t, h)
        x_store = np.vstack((x_store, x))

    # Take final step with h <= deltat_max
    h_final = t_final - t_vals[-1]
    x = time_step(x, t, h_final)
    
    x_store = np.vstack((x_store, x))
    t_vals = np.append(t_vals, t_final)

    # Combine into one store
    store = np.concatenate((np.array([t_vals]).T, x_store), axis=1)

    return store
    
# Week 15
def limit_cycle_condition(ode_func,
                          params: list[float],
                          pair: np.ndarray, 
                          num_loops_needed: int = 1,
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
    params : list of floats
        Parameters passed into the ODE function `ode_func`.
    pair : 1-D Numpy array of floats
        In format `[T, u0]`. Here `T` (float) is the period of the limit cycle, and `u0`
         (array of floats) is the initial point along the limit cycle.
    num_loops_needed : int, default 1
        The number of consecutive periods checked by our limit_cycle_condition function
    phase_condition : 'constant' or 'derivative', default 'derivative'
        Specifies the type of phase condition used
         to select a distinct limit cycle, methods specified below.
    constant_value : float, default None
        Value that the first state variable of `u0` should have. Must be supplied
         if phase_condition == 'constant'.
    deltat_max : float, default 0.01
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
    >>> def shm(params,x, t):
            return np.array(([x[1], -x[0]]))
    >>> limit_cycle_condition(shm, None, np.array(([2*np.pi,1,0])), 5)
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
    Exception
        if the period < 2*deltat_max, which will lead to problems for our solver

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
    T, u0 = pair[0], pair[1:]

    if T < 2*deltat_max:
        raise Exception("Please choose a different initial period guess, period tested has become smaller than timestep used")

    # Establish empty array to hold Gs
    num_variables = len(u0)
    G_collection = np.zeros([num_loops_needed*num_variables])

    # Loop over number of loops checked
    for i in range(num_loops_needed):
        # Compute G by calling the solver to solve until time (i+1)*T. Solve with RK4 and deltat_max supplied
        solution = solve_to(ode_func=ode_func, params=params, x_init=u0, t_init=0, 
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
        u0dot = ode_func(params,u0, 0)
        phi = u0dot[0]

    return np.append(G_collection,[phi])

def find_limit_cycle(ode_func, 
                     params: list[float],
                     init_point_guess: np.ndarray,
                     init_period_guess: float,
                     num_loops_needed: int = 1,
                     phase_condition: Literal['constant', 'derivative']='derivative',
                     constant_value: float = None,
                     deltat_max: float = 0.1,
                     print_findings: bool = True):
    '''
    Searches for a limit cycle of an ODE given an initial guess.

    Takes in an ODE definition function, and an initial guess for the period and starting state
    of a limit cycle of that ODE. Uses scipy.optimize.root with limit_cycle_condition to converge
    towards a limit cycle starting with the supplied guess. If convergence is successful, returns
    the period and starting state of the limit cycle it located.

    -----
    Parameters
    -----
    ode_func : function
        Definition function for the RHS of the ODE `x' = ode_func(x,t)`.
    params : list of floats
        Parameters passed into the ODE function `ode_func`.
    init_point_guess : 1-D Numpy array
        Initial guess for the starting state of a limit cycle.
    init_period_guess : float
        Initial guess for the period of the limit cycle.
    num_loops_needed : int, default 1
        The number of consecutive periods checked by our limit_cycle_condition function
    phase_condition : 'constant' or 'derivative', default 'derivative'
        Specifies the type of phase condition used
         to select a distinct limit cycle, methods specified below.
    constant_value : float, default None
        Value that the first state variable of `u0` should have. Must be supplied
         if phase_condition == 'constant'.
    deltat_max : float, default 0.01
        Step size used by solve_to numerical ODE solver to compute the solution of the  
         supplied ODE starting from `u0`.
    print_findings: bool, default True
        If True then this function prints out whether the convergence was
         successful, and if so the period and starting state of the limit cycle.

    ------
    Returns
    ------
    If scipy.optimize.root converges:
        best_period : float
            The period of the found limit cycle.
        best_point : 1-D Numpy array of floats
            The starting state of the found limit cycle.
    Else:
        `None`.

    -----
    Example
    -----
    >>> import numpy as np
    >>> def shm(params, x, t):
            return np.array(([x[1], -x[0]]))
    >>> find_limit_cycle_min(shm, None, np.array(([5,1])), 10, 1, 'constant', 4)
    A limit cycle was found:
    Period: 12.566370606193304 ,
    Starting state: [4.00000009 0.44653776] .

    (12.566370606193304, array([4.00000009, 0.44653776]))

    -----
    Notes
    -----
    We minimize the objective function specified in limit_cycle_condition_min, and specify bounds
    so that the solver doesn't try periods that are lower than deltat_max.

    -----
    Raises
    -----
    Exception
        If init_period_guess <= deltat_max.

    -----
    See also
    -----
    limit_cycle_condition
        Describes the function we find the root of to find limit cycles.
    find_limit_cycle_min
        Does the same process but with a minimization objective function instead of root finding.
    
    '''
    # Establish lambda function for use with scipy.optimize.root, means we only vary params
    specific_condition = lambda pair: limit_cycle_condition(ode_func=ode_func,
                                                            params=params,
                                                              pair = pair,
                                                              num_loops_needed=num_loops_needed,
                                                              phase_condition=phase_condition,
                                                              constant_value=constant_value,
                                                              deltat_max=deltat_max)


    # Pack init_point_guess and init_period_guess into init_pair
    init_pair = np.concatenate(([init_period_guess], init_point_guess))

    result = root(specific_condition, init_pair, method='lm')

    if result.success:
        best_period, best_point = result.x[0], result.x[1:]
        if print_findings:
            print("A limit cycle was found:")
            print("Period:", round(best_period,2), ",")
            print("Starting state:", np.round(best_point,2), ".")
        return best_period, best_point
    else:
        if print_findings:
            print("No limit cycle was found (failed to converge).")
        return None, None
    
# SEE odefunctions.py FOR UPDATED VERSIONS OF FUNCTIONS

# Week 16
def limit_cycle_condition_min(ode_func,
                              params: list[float],
                              pair: np.ndarray, 
                              num_loops_needed: int = 1,
                              phase_condition: Literal['constant', 'derivative']='derivative',
                              constant_value: float = None,
                              deltat_max: float = 0.01):
    ''' 
    Computes the value of an objective function used to search for limit cycle solutions of ODEs.

    A limit cycle will have initial state `u0` and period `T` such that this function is minimized.

    -----
    Parameters
    -----
    ode_func : function
        Definition function for the RHS of the ODE `x' = ode_func(x,t)`.
    params : list of floats
        Parameters passed into the ODE function `ode_func`.
    pair : 1-D Numpy array of floats
        In format `[T, u0]`. Here `T` (float) is the period of the limit cycle, and `u0`
         (array of floats) is the initial point along the limit cycle.
    num_loops_needed : int, default 1
        The number of consecutive periods checked by our limit_cycle_condition function
    phase_condition : 'constant' or 'derivative', default 'derivative'
        Specifies the type of phase condition used
         to select a distinct limit cycle, methods specified below.
    constant_value : float, default None
        Value that the first state variable of `u0` should have. Must be supplied
         if phase_condition == 'constant'.
    deltat_max : float, default 0.01
        Step size used by solve_to numerical ODE solver to compute the solution of the  
         supplied ODE starting from `u0`.

    ------
    Returns
    ------
    float
        Value of the objective function `f = (G_collection^2)/size(G_collection) + 10 phi^2 + alpha T` (to be minimized).
    
    -----
    Example
    -----
    >>> import numpy as np
    >>> def shm(params, x, t):
            return np.array(([x[1], -x[0]]))
    >>> objective = limit_cycle_condition_min(shm, None, np.array(([1,3,3])), 10, 'constant', 2)
    >>> print(objective)
    30.551415942882745


    ------
    Notes
    ------
    G is the element-wise difference between our starting point u0 and the end of its trajectory
    after time T, uT. When G is zero this means the solution has returned to its starting point
    u0 after time T, and our solution is thus periodic. We compute G for a range of num_loops_needed
    many final times T, 2T,..., and collect them into G_collection.

    phi is the phase condition, used to set the phase and thus choose a periodic orbit from the
    family of orbits generated when G=0. 

    If phase_condition == 'constant' then we supply a 'constant_value', which the first state
    variable of the ODE must attain at time t=0. We set phi = constant_value - u0[0]

    If phase_condition == 'derivative' then we compute the derivative of the first state 
    variable at time t=0. Every limit cycle should contain a point where the first state variable
    has a turning point (or is constant). We set phi = d/dt[u[t=0]]

    We also minimise the period, to avoid period multiples, by including the regularization alpha T
    in our objective function.

    -----
    Raises
    -----
    Exception
        If num_loops_needed < 1.
    Exception
        If phase_condition != 'constant' or 'derivative'.
    Exception
        if phase_condition == 'constant' but no constant value is supplied.

    -----
    See also
    -----
    find_limit_cycle
        Function that finds limit cycles by using scipy.optimize.root to find a
         root of limit_cycle_condition.

    '''
    # Error Messages
    if num_loops_needed < 1:
        raise Exception("Please specify a positive number of loops needed.")
    if not (phase_condition in ['constant', 'derivative']):
        raise Exception("Input Error: Please supply a valid phase condition.")
    if phase_condition == 'constant' and constant_value == None:
        raise Exception("Input Error: Please supply a starting value that the first state variable must attain.")

    # Extract T, u0 from params
    T, u0 = pair[0], pair[1:]

    # Establish empty array to hold Gs
    num_variables = len(u0)
    G_collection = np.zeros([num_loops_needed*num_variables])

    # Loop over number of loops checked
    for i in range(num_loops_needed):
        # Compute G by calling the solver to solve until time (i+1)*T. Solve with RK4 and deltat_max supplied
        solution = solve_to(ode_func=ode_func, params=params, x_init=u0, t_init=0, 
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
        u0dot = ode_func(params, u0, 0)
        phi = u0dot[0]

    # Introduce a regularisation constant for the period in our objective function
    alpha = 1e-5

    # Return sum of squares of G_collection and phi, plus the period times some weight
    return np.sum(np.square(G_collection))/len(G_collection) + 10*phi**2 + alpha*T



def find_limit_cycle_min(ode_func,
                     params: list[float],
                     init_point_guess: np.ndarray,
                     init_period_guess: float,
                     num_loops_needed: int = 1,
                     phase_condition: Literal['constant', 'derivative']='derivative',
                     constant_value: float = None,
                     deltat_max: float = 0.01,
                     print_findings: bool = True):
    '''
    Searches for a limit cycle of an ODE given an initial guess.

    Takes in an ODE definition function, and an initial guess for the period and starting state
    of a limit cycle of that ODE. Uses scipy.optimize.minimize with limit_cycle_condition_min to converge
    towards a limit cycle starting with the supplied guess. If convergence is successful, returns
    the period and starting state of the limit cycle it located.

    -----
    Parameters
    -----
    ode_func : function
        Definition function for the RHS of the ODE `x' = ode_func(x,t)`.
    params : list of floats
        Parameters passed into the ODE function `ode_func`.
    init_point_guess : 1-D Numpy array
        Initial guess for the starting state of a limit cycle.
    init_period_guess : float
        Initial guess for the period of the limit cycle.
    num_loops_needed : int, default 1
        The number of consecutive periods checked by our limit_cycle_condition function
    phase_condition : 'constant' or 'derivative', default 'derivative'
        Specifies the type of phase condition used
         to select a distinct limit cycle, methods specified below.
    constant_value : float, default None
        Value that the first state variable of `u0` should have. Must be supplied
         if phase_condition == 'constant'.
    deltat_max : float, default 0.01
        Step size used by solve_to numerical ODE solver to compute the solution of the  
         supplied ODE starting from `u0`.
    print_findings: bool, default True
        If True then this function prints out whether the convergence was
         successful, and if so the period and starting state of the limit cycle.

    ------
    Returns
    ------
    If scipy.optimize.root converges:
        best_period : float
            The period of the found limit cycle.
        best_point : 1-D Numpy array of floats
            The starting state of the found limit cycle.
    Else:
        `None`.

    -----
    Example
    -----
    >>> import numpy as np
    >>> def shm(x, t):
            return np.array(([x[1], -x[0]]))
    >>> find_limit_cycle_min(shm, np.array(([5,1])), 10, 1, 'constant', 4)
    A limit cycle was found:
    Period: 12.566370606193304 ,
    Starting state: [4.00000009 0.44653776] .

    (12.566370606193304, array([4.00000009, 0.44653776]))

    -----
    Notes
    -----
    We minimize the objective function specified in limit_cycle_condition_min, and specify bounds
    so that the solver doesn't try periods that are lower than deltat_max.

    -----
    Raises
    -----
    Exception
        If init_period_guess <= deltat_max.

    -----
    See also
    -----
    limit_cycle_condition_min
        Describes the objective function we minimize to find limit cycles.
    find_limit_cycle
        Does the same process but with root finding objective instead of minimisation
    
    '''
    # Error Messages
    if init_period_guess <= deltat_max:
        raise Exception("Initial period guess is less than step size.")
    
    # Establish lambda function for use with scipy.optimize.root, means we only vary params
    objective_function = lambda pair: limit_cycle_condition_min(ode_func=ode_func,
                                                                params=params,
                                                                pair=pair,
                                                                num_loops_needed=num_loops_needed,
                                                                phase_condition=phase_condition,
                                                                constant_value=constant_value,
                                                                deltat_max=deltat_max)


    # Pack init_point_guess and init_period_guess into init_pair
    init_pair = np.concatenate(([init_period_guess], init_point_guess))
    
    # Specify lower bound for limit cycle period to be greater than 10*deltatmax
    bounds = [(10*deltat_max, None)] + [(None,None) for i in range(len(init_point_guess))]

    # Minimize the limit cycle objective function with above bounds and initial guess
    result = minimize(fun=objective_function, x0=init_pair, bounds=bounds)

    # If objective function value too large we deem the result a failure
    # Note if the actual period is larger than 1/alpha in our objective function
    # then any attempt will be deemed a failure.
    if objective_function(result.x) > 1:
        if print_findings:
            print("No limit cycle was found (failed to converge).")
        return None, None

    if result.success:
        best_period, best_point = result.x[0], result.x[1:]
        if print_findings:
            print("A limit cycle was found:")
            print("Period:", round(best_period,2), ",")
            print("Starting state:", [round(i,2) for i in best_point], ".")
        return best_period, best_point
    else:
        if print_findings:
            print("No limit cycle was found (failed to converge).")
        return None, None
            

# Week 17
def limit_cycle_continuation(ode_func,
                             default_params: list[float],
                             vary_param_index: int,
                             vary_param_start: float,
                             vary_param_step: float,
                             init_point_guess: np.ndarray,
                             init_period_guess: float,
                             num_steps: int,
                             method: Literal['NPC','PAC']='NPC',
                             num_loops_needed: int = 1,
                             phase_condition: Literal['constant', 'derivative']='derivative',
                             constant_value: float = None,
                             deltat_max: float = 0.1
                             ):
    '''  
    Searches for a limit cycle of an ODE given an initial guess.

    Takes in an ODE definition function, and an initial guess for the period and starting state
    of a limit cycle of that ODE. At each step uses scipy.optimize.root with limit_cycle_condition
    to converge towards a limit cycle, starting with the supplied guess. 

    -----
    Parameters
    -----
    ode_func : function
        Definition function for the RHS of the ODE `x' = ode_func(x,t)`.
    default_params : list of floats
        Default set of parameters passed into the ODE function `ode_func`.
    vary_param_index : int,
        The index of the parameter to be varied within default_params.
    vary_param_start : float,
        The starting parameter value of the varied parameter.
    vary_param_step : float,
        The size of change in parameter value used by natural-parameter continuation
         and initially in pseudo-arclength continuation.
    init_point_guess : 1-D Numpy array
        Initial guess for the starting state of a limit cycle.
    init_period_guess : float
        Initial guess for the period of the limit cycle.
    num_steps : int,
        The number of steps to conduct in the continuation
    method : 'NPC' or 'PAC', default 'NPC'
        Numerical continuation method, either natural-parameter continuation (NPC),
         or pseudo-arclength continuation (PAC)
    num_loops_needed : int, default 1
        The number of consecutive periods checked by our limit_cycle_condition function
    phase_condition : 'constant' or 'derivative', default 'derivative'
        Specifies the type of phase condition used
         to select a distinct limit cycle, methods specified below.
    constant_value : float, default None
        Value that the first state variable of `u0` should have. Must be supplied
         if phase_condition == 'constant'.
    deltat_max : float, default 0.01
        Step size used by solve_to numerical ODE solver to compute the solution of the  
         supplied ODE starting from `u0`.

    ------
    Returns
    ------
    x_store : 2-D NumPy array,
        Contains the pairs [Period, Starting Point] for each parameter value tried
    c_store : 1-D NumPy array
        Contains the corresponding parameter values for each pair

    -----
    Example
    -----
    >>> import numpy as np
    >>> def predator_prey(params, x: np.ndarray, t: float):
            a,d,b = params
            dxdt = x[0]*(1-x[0]) - (a*x[0]*x[1])/(d+x[0])
            dydt = b*x[1]*(1-(x[1]/x[0]))
            return np.array((dxdt, dydt))
    >>> pair_store, b_store = limit_cycle_continuation(predator_prey,
                                               [1,0.1,0.1], 2, 0.1, 0.01,
                                               np.array(([0.8,0.2])),
                                               30, 20, method='NPC')
    >>> print(b_store)
    [0.1, 0.11, 0.12, 0.13, 0.14, 0.15000000000000002, ... ,0.1663615 ]
    >>> print(pair_store)
    [[31.60389551  0.79912268  0.18061336]
    [29.55628338  0.77832994  0.19469945]
    ...
    [17.92644331  0.27015621  0.27015621]
    [17.92644331  0.27015621  0.27015621]]
    
    -----
    Raises
    -----
    Exception
        If init_period_guess <= deltat_max.
    Exception
        If vary_param_index >= size(default_params)

    -----
    See also
    -----
    limit_cycle_condition
        Describes the function we find the root of to find limit cycles.
    find_limit_cycle
        Finds a single limit cycle for a fixed set of parameters
    
    '''
    if init_period_guess <= deltat_max:
        raise Exception("Initial guess period is smaller than stepsize")
    if vary_param_index > len(default_params)-1:
        raise Exception("vary_param_index supplied is too large")
    
    # Define an inner function that packages information, only taking pair and params as inputs
    def packing_function(vary_param_val: float, pair: np.ndarray):
        
        # Update params list to have the current vary_param value
        params = default_params
        params[vary_param_index] = vary_param_val

        return limit_cycle_condition(ode_func=ode_func,
                                    params = params,
                                    pair = pair,
                                    num_loops_needed=num_loops_needed,
                                    phase_condition=phase_condition,
                                    constant_value=constant_value,
                                    deltat_max=deltat_max)
    
    
    # We now refer to our pairs as 'x' and varied parameter as c, for brevity

    # Pack init_point_guess and init_period_guess into x0guess, rename vary_param_start
    x0guess = np.concatenate(([init_period_guess], init_point_guess))
    c0 = vary_param_start

    # Define specific_condition which only takes the pair x as input
    specific_condition = lambda x: packing_function(c0, x)

    # Solve for the first point of the continuation x0 using our starting guess
    x0 = root(specific_condition, x0guess).x

    # Update c by vary_param_step, 
    # Find our second point x1 via natural parameter continuation
    c1 = c0 + vary_param_step
    specific_condition = lambda x: packing_function(c1, x)
    x1 = root(specific_condition, x0).x

    if method == 'NPC':
        # Construct stores
        x_store = np.vstack((x0,x1))
        c_store = [c0,c1]
        x_current = x1
        c_current = c1
        
        # Loop over number of steps
        for i in range(num_steps-1):
            # Update c by parameter step
            c_next = c_current + vary_param_step

            # Fix the objective function and find the next solution
            specific_condition = lambda x: packing_function(c_next, x)

            # Try the next step of continuation
            try:
                x_next = root(specific_condition, x_current).x
            except:
                print("Numerical divergence encountered, terminating continuation")
                continue
                
            # Update stores and then current values
            x_store = np.vstack((x_store, x_next))
            c_store += [c_next]
            x_current = x_next
            c_current = c_next

    elif method == 'PAC':
        # Construct v0 and v1 vectors
        v0 = np.concatenate(([c0],x0))
        v1 = np.concatenate(([c1],x1))

        # Construct stores
        v_store = np.stack((v0, v1))
        v_last = v0
        v_current = v1

        # Define objective function used for root finding in each step
        def psa_objective(v: np.ndarray, v_next_guess: np.ndarray, secant: np.ndarray):
            original = packing_function(v[0], v[1:])
            pseudo_arclength_eq = np.dot((v - v_next_guess),secant)
            return np.concatenate((original, [pseudo_arclength_eq]))
        
        # Conduct num_steps many steps of pseudo-arclength continuation
        for i in range(num_steps-1):
            # Calculate secant and next guess
            secant = v_current - v_last
            v_next_guess = v_current + secant

            # Define current objective function and find its root for v_next
            current_objective = lambda v: psa_objective(v, v_next_guess, secant)

            # Try the next step of continuation
            try:
                v_next = root(current_objective, v_next_guess).x
            except:
                print("Numerical divergence encountered, terminating continuation")
                continue

            # Update
            v_store = np.vstack((v_store, v_next))
            v_last = v_current
            v_current = v_next
        
        # Unpack v_store into x and c
        c_store, x_store = v_store[:,0], v_store[:,1:]

    return x_store, c_store


# Week 19,20
def bvp_construct_A_and_b(num_grid_points: int,
                          grid_bounds: list[float],
                          left_boundary_vals: list[float],
                          right_boundary_vals: list[float],
                          left_boundary_type: Literal['Dirichlet', 'Neumann', 'Robin'] = 'Dirichlet', 
                          right_boundary_type: Literal['Dirichlet', 'Neumann', 'Robin'] = 'Dirichlet',
                          ):
    '''
    Prepares the matrix A_matrix and vector b_vector for use in finite difference solving for a BVP diffusion problem.
    See finite_diff_bvp_solver function.
    
    Problems are of form:
    D (d^2u/dx2) + q(x,u:mu) = 0
    between a<=x<=b, where D is the diffusivity (scalar), and q is any source function.

    Boundary conditions (at end x=a WLOG):
    - Dirichlet: u(a) = alpha
    - Neumann: du/dx{a} = alpha
    - Robin: du/dx{a} = alpha + beta*u(a)
    where alpha is specified.

    -----
    Parameters
    -----
    num_grid_points : int
        The number of spatial x grid points used in the numerical approximation.
    grid_bounds : [a,b] where a<b are floats
        The bounds a<=x<=b of the problem.
    left_boundary_vals : [alpha, beta] where alpha,beta are floats
        The constant values used to describe the left boundary condition. Beta optional, used for Robin boundary
    right_boundary_vals : [alpha, beta] where alpha,beta are floats
        The constant values used to describe the left boundary condition. Beta optional, used for Robin boundary
    left_boundary_type : 'Dirichlet', 'Neumann', or 'Robin', default 'Dirichlet'
        The string used to specify the type of boundary condition on the left.
    right_boundary_type : 'Dirichlet', 'Neumann', or 'Robin', default 'Dirichlet'
        The string used to specify the type of boundary condition on the right.

    -----
    Returns
    -----
    A_matrix : 2-D tridiagonal matrix array
        The corresponding matrix used to specify the 2nd order diffusion term and boundary values
    b_matrix : 1-D vector array
        The corresponding vector used to specify the 2nd order diffusion term and boundary values
    left_dirichlet_val : float or None
        Value of the left dirichlet boundary u(a) = alpha, None if not a dirichlet boundary
    right_dirichlet_val : float or None
        Value of the right dirichlet boundary u(b) = beta, None if not a dirichlet boundary
    
    -----
    Example
    -----
    >>> A_matrix, b_vec, left_dirichlet_val, right_dirichlet_val = bvp_construct_A_and_b(5,[0,5],[1.5],[3],'Dirichlet','Neumann')
    >>> print(A_matrix)
    [[-2.  1.  0.  0.  0.]
    [ 1. -2.  1.  0.  0.]
    [ 0.  1. -2.  1.  0.]
    [ 0.  0.  1. -2.  1.]
    [ 0.  0.  0.  2. -2.]]
    >>> print(b_vec)
    [1.5 0.  0.  0.  6. ]
    >>> print(left_dirichlet_val)
    1.5
    >>> print(right_dirichlet_val)
    None

    -----
    Notes
    -----
    We use a centred finite difference scheme to approximate the second derivative, and
    we use ghost points for the derivative boundary conditions (Neumann, Robin).
    We construct tridiagonal matrices A_matrix and associated vectors b_vec,
    converting the problem into a linear algebra problem to be solved by numpy's linalg package
    In the case of a nonlinear dependence q(u) we employ root finding instead to find 
    a solution where linear algebra cannot be used.

    -----
    Raises
    -----
    Exception
        If the number of grid points are 3 or less (impractical).
    Exception
        If a,b in the grid bounds are such that b<=a
    Exception
        If the left or right boundary condition is Robin and the 
         associated values isn't length 2.

    -----
    See also
    -----
    .   
    '''
    if num_grid_points < 4:
        raise Exception("Please use more grid points.")
    if grid_bounds[0] >= grid_bounds[1]:
        raise Exception("Grid bounds must be in format [a,b] where a < b.")
    if left_boundary_type == 'Robin':
        if len(left_boundary_vals) != 2:
            raise Exception("Please enter 2 values for left_boundary_vals when using Robin boundary condition.")
    if right_boundary_type == 'Robin':
        if len(right_boundary_vals) != 2:
            raise Exception("Please enter 2 values for right_boundary_vals when using Robin boundary condition.")

    N = num_grid_points
    a, b = grid_bounds

    # Establish delta_x
    delta_x = (b-a)/(N-1)

    # Initiate holders for dirichlet boundary conditions
    left_dirichlet_val, right_dirichlet_val = None, None

    # Calculate number of interior grid points based on boundary conditions
    N_interior = N-2
    if left_boundary_type != 'Dirichlet':
        N_interior += 1  # Include additional ghost point (on left)
    else:
        left_dirichlet_val = left_boundary_vals[0]
    if right_boundary_type != 'Dirichlet':
        N_interior += 1  # Include additional ghost point (on right)
    else:
        right_dirichlet_val = right_boundary_vals[0]
    
    # Initialise b_vec and A_matrix
    b_vec = np.zeros([N_interior])
    A_matrix = -2*np.eye(N_interior,k=0) + np.eye(N_interior,k=1) + np.eye(N_interior,k=-1)

    # Treat left boundary
    if left_boundary_type == 'Dirichlet':
        b_vec[0] = left_boundary_vals[0]
    elif left_boundary_type == 'Neumann':
        b_vec[0] = -left_boundary_vals[0]*2*delta_x
        A_matrix[0,1] = 2
    elif left_boundary_type == 'Robin':
        b_vec[0] = -left_boundary_vals[0]*2*delta_x
        A_matrix[0,1] = 2
        A_matrix[0,0] = -2*(1+delta_x*left_boundary_vals[1])

    # Treat right boundary
    if right_boundary_type == 'Dirichlet':
        b_vec[-1] = right_boundary_vals[0]
    elif right_boundary_type == 'Neumann':
        b_vec[-1] = right_boundary_vals[0]*2*delta_x
        A_matrix[-1,-2] = 2
    elif right_boundary_type == 'Robin':
        b_vec[-1] = right_boundary_vals[0]*2*delta_x
        A_matrix[-1,-2] = 2
        A_matrix[-1,-1] = -2*(1+delta_x*right_boundary_vals[1])

    return A_matrix, b_vec, left_dirichlet_val, right_dirichlet_val

def finite_diff_bvp_solver(num_grid_points: int,
                           diffusivity: float,
                           grid_bounds: list[float],
                           A_matrix: np.ndarray,
                           b_vec: np.ndarray,
                           left_dirichlet_val,
                           right_dirichlet_val,
                           q_func = None,
                           q_nonlinear: bool = False,
                           q_param: float = None,
                           guess_function = None
                           ):
    '''
    Numerically solves a diffusion equation in 1D with given boundary conditions and source function, using the finite difference method.

    Solves problems of form:
    D (d^2u/dx2) + q(x,u:mu) = 0
    between a<=x<=b, where D is the diffusivity (scalar), and q is any source function.

    Boundary conditions (at end x=a WLOG):
    - Dirichlet: u(a) = alpha
    - Neumann: du/dx{a} = alpha
    - Robin: du/dx{a} = alpha + beta*u(a)
    where alpha is specified.

    Boundary conditions are specified via the inputs A_matrix, b_vec and left_dirichlet_val, right_dirichlet_val.
    These are obtained from the bvp_construct_A_and_b function.

    -----
    Parameters
    -----
    num_grid_points : int
        The number of spatial x grid points used in the numerical approximation.
    diffusivity : float
        The diffusivity constant of the system. Higher means more 'flattening'.
    grid_bounds : [a,b] where a<b are floats
        The bounds a<=x<=b of the problem.
    A_matrix : 2-D tridiagonal matrix array
        The corresponding matrix used to specify the 2nd order diffusion term and boundary values
    b_matrix : 1-D vector array
        The corresponding vector used to specify the 2nd order diffusion term and boundary values
    left_dirichlet_val : float or None
        Value of the left dirichlet boundary u(a) = alpha, None if not a dirichlet boundary
    right_dirichlet_val : float or None
        Value of the right dirichlet boundary u(b) = beta, None if not a dirichlet boundary
    q_func : function, default None
        The scalar source term function q(x,u:mu).
    q_nonlinear : bool, default False
        True if the function q has dependence on u
    q_param : float, default None
        The parameter value mu passed to source function q
    guess_function : function, default None
        A function that guesses the solution, used when source term q is nonlinear

    -----
    Returns
    -----
    u_interior : 1-D array, length num_grid_points
        The solution to the differential equation.
    x_vals : 1-D array, length num_grid_points
        The corresponding x grid points
    
    -----
    Example
    -----
    >>> import numpy as np
    >>> num_grid_points = 5
    >>> grid_bounds = [0,5]
    >>> diffusivity = 2
    >>> A_matrix, b_vec, left_dirichlet_val, right_dirichlet_val = bvp_construct_A_and_b(num_grid_points, grid_bounds,[1.5],[3],'Dirichlet','Neumann')
    >>> def source_func(x,u,mu):
            return np.sin(x)
    >>> solution, xvals = finite_diff_bvp_solver(num_grid_points, diffusivity,grid_bounds, A_matrix, b_vec, left_dirichlet_val, right_dirichlet_val, source_func)
    >>> print(solution)
    [ 1.5         4.82781189  7.73488829 10.18731598 12.56918367 15.3294526 ]
    >>> print(xvals)
    [0. 1. 2. 3. 4. 5.]

    -----
    Notes
    -----
    We use a centred finite difference scheme to approximate the second derivative, and
    we use ghost points for the derivative boundary conditions (Neumann, Robin).
    We construct tridiagonal matrices A_matrix and associated vectors b_vec,
    converting the problem into a linear algebra problem to be solved by numpy's linalg package
    In the case of a nonlinear dependence q(u) we employ root finding instead to find 
    a solution where linear algebra cannot be used.

    -----
    Raises
    -----
    Exception
        If the number of grid points are 3 or less (impractical).
    Exception
        If a,b in the grid bounds are such that b<=a
    Exception
        If the left or right boundary condition is Robin and the 
         associated values isn't length 2.

    -----
    See also
    -----
    bvp_construct_A_and_b
        Constructs A_matrix, b_vec, left_dirichlet_val and right_dirichlet_val used for   
    '''
    # Error Messages
    if grid_bounds[0] >= grid_bounds[1]:
        raise Exception("Grid bounds must be in format [a,b] where a < b.")
    
    # Unpack and rename inputs
    D = diffusivity
    N = num_grid_points
    a, b = grid_bounds

    # Establish x grid
    x_vals = np.linspace(a,b,N)
    delta_x = (b-a)/(N-1)

    # Extract interior values
    x_vals_interior = x_vals
    if left_dirichlet_val != None:
        x_vals_interior = x_vals_interior[1:]
    if right_dirichlet_val != None:
        x_vals_interior = x_vals_interior[:-1]
        
    # If no source term q then call linalg to solve
    if q_func == None:
        u_interior = np.linalg.solve(A_matrix, -b_vec)

    # If there is a source term q, then split by linear or nonlinear
    elif not q_nonlinear:
        # q(x,u:mu) = q(x:mu) so calculate fixed values
        q_vals_interior = q_func(x=x_vals_interior, u=None, mu = q_param)

        # Solve with linalg solve
        rhs_vec = -b_vec - ((delta_x**2)/D)*q_vals_interior
        u_interior = np.linalg.solve(A_matrix, rhs_vec)

    elif q_nonlinear:
        # Define objective function to minimise
        def objective(u):
            q_vals_interior = q_func(x=x_vals_interior, u=u, mu= q_param)
            F = np.matmul(A_matrix, u) + b_vec + ((delta_x**2)/D)*q_vals_interior
            return F
        
        # Establish initial guess solution via a function
        # If guess solution function isn't supplied then define one
        if guess_function == None:
            # If both Dirichlet bounds given:
            if (left_dirichlet_val != None) and (right_dirichlet_val != None):
                # Establish function connecting both points
                def guess_function(x):
                    return left_dirichlet_val + ((right_dirichlet_val-left_dirichlet_val)/(b-a))*(x-a)
                
            elif left_dirichlet_val != None:
                # Establish function with constant left boundary value
                def guess_function(x):
                    return left_dirichlet_val*np.ones([len(x)])
                
            elif right_dirichlet_val != None:
                # Establish function with constant right boundary value
                def guess_function(x):
                    return right_dirichlet_val*np.ones([len(x)])
                
            else:
                # Establish function f(x) = 1
                def guess_function(x):
                    return np.ones([len(x)])
        
        # Calculate guess solution
        u_guess = guess_function(x_vals_interior)

        # Solve via scipy root with initial guess solution 
        solution = root(objective, u_guess)
        u_interior = solution.x

    # Add prescribed boundary points if Dirichlet, u_interior becomes full solution
    if left_dirichlet_val != None:
        u_interior = np.concatenate(([left_dirichlet_val], u_interior))
    if right_dirichlet_val != None:
        u_interior = np.concatenate((u_interior, [left_dirichlet_val]))
    
    return u_interior, x_vals

def diffusion_pde_solver(num_grid_points: int,
                          grid_bounds: list[float],
                          diffusivity: float,
                          init_func,
                          t_init: float,
                          t_final: float,
                          deltat_max: float,
                          left_boundary_vals: list[float],
                          right_boundary_vals: list[float],
                          left_boundary_type: Literal['Dirichlet', 'Neumann', 'Robin'] = 'Dirichlet', 
                          right_boundary_type: Literal['Dirichlet', 'Neumann', 'Robin'] = 'Dirichlet',
                          q_func = None,
                          method: Literal['Explicit Euler'] = 'Explicit Euler'):
    '''  
    Solves 1-D diffusion PDEs with the method of lines, using different timestepping methods.

    -----
    Parameters
    -----
    num_grid_points : int
        The number of spatial x grid points used in the numerical approximation.
    grid_bounds : [a,b] where a<b are floats
        The bounds a<=x<=b of the problem.
    diffusivity : float
        The diffusivity constant of the system. Higher means more 'flattening'.
    init_func : function
        The initial spatial condition function u(x,0)
    t_init, t_final : floats
        Initial and final times.
    deltat_max : float
        Step size used in numerical timestepping, some final steps may be smaller than this.
    left_boundary_vals : [alpha, beta] where alpha,beta are floats
        The constant values used to describe the left boundary condition. Beta optional, used for Robin boundary
    right_boundary_vals : [alpha, beta] where alpha,beta are floats
        The constant values used to describe the left boundary condition. Beta optional, used for Robin boundary
    left_boundary_type : 'Dirichlet', 'Neumann', or 'Robin', default 'Dirichlet'
        The string used to specify the type of boundary condition on the left.
    right_boundary_type : 'Dirichlet', 'Neumann', or 'Robin', default 'Dirichlet'
        The string used to specify the type of boundary condition on the right.
    q_func : function
        The source function used in the PDE. Has to be of the form q=q(x)
    method : 'Explicit Euler'
        The type of timestepping method used to simulate the PDE
    
    ------
    Returns
    ------
    2-D Numpy array
        Full timeseries solution of PDE from t_init to t_final.
        Columns `[t, u1(t), u2(t), ... ,un(t)]`, with each row containing the time value after a timestep
        and the values of each state variable in `u=[u1, u2, ... ,un]`.

    -----
    Example
    -----
    >>> import numpy as np
    >>> store, xvals  = diffusion_pde_solver(5, [0,5], 1, np.sin ,0,10,0.1,[2],[3])
    >>> print(xvals)
    [0.   1.25 2.5  3.75 5.  ]
    >>> print(store)
    [[ 0.          1.          0.94898462  0.59847214 -0.57156132  1.        ]
    [ 0.1         1.          0.98973782  0.55942378 -0.28804692  1.        ]
    [ 0.2         1.          1.02397787  0.54321024 -0.04021254  1.        ]
    [ 0.3         1.          1.05369875  0.54479128  0.1774401   1.      
    ....
    [ 9.7         1.          2.20185124  2.43190158  2.70184305  1.        ]
    [ 9.8         1.          2.20362259  2.43440734  2.70361538  1.        ]
    [ 9.9         1.          2.20532881  2.43682091  2.70532247  1.        ]
    [10.          1.          2.20697228  2.43914566  2.7069667   1.        ]]

    -----
    See also
    -----
    bvp_construct_A_and_b
        Used inside this function to construct the matrix A and vector b.
    '''
    A_matrix, b_vec, left_dirichlet_val, right_dirichlet_val = bvp_construct_A_and_b(num_grid_points,
                                                                                     grid_bounds,
                                                                                     left_boundary_vals,
                                                                                     right_boundary_vals,
                                                                                     left_boundary_type,
                                                                                     right_boundary_type)
    
    # Unpack and rename inputs
    D = diffusivity
    N = num_grid_points
    a, b = grid_bounds

    # Establish x grid
    x_vals = np.linspace(a,b,N)
    delta_x = (b-a)/(N-1)

    # Extract interior values
    x_vals_interior = x_vals
    if left_dirichlet_val != None:
        x_vals_interior = x_vals_interior[1:]
    if right_dirichlet_val != None:
        x_vals_interior = x_vals_interior[:-1]

    # Determine initial conditions
    init_dist = init_func(x_vals_interior)
    
    # Define a rhs function depending on q
    if q_func == None:
        def rhs_func(params,x,t):
            return (D/(delta_x**2))*(np.matmul(A_matrix,x)+b_vec)

    else:
        def rhs_func(params,x,t):
            return (D/(delta_x**2))*(np.matmul(A_matrix,x)+b_vec)+q_func(x)
    
    if method == 'Explicit Euler':
        # Timestep via explicit Euler method:
        store = solve_to(rhs_func,None,init_dist,t_init, t_final, deltat_max, 'Euler')

    # Pack the remaining dirichlet boundaries on to the ends
    if left_dirichlet_val != None:
        n_timesteps, _ = np.shape(store)
        left_vals = left_dirichlet_val*np.ones([n_timesteps,1])
        store = np.concatenate((np.array([store[:,0]]).T,left_vals, store[:,1:]), axis=1)
    if right_dirichlet_val != None:
        n_timesteps, _ = np.shape(store)
        right_vals = right_dirichlet_val*np.ones([n_timesteps,1])
        store = np.concatenate((store,right_vals), axis=1)

    return store, x_vals