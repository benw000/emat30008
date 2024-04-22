# Functions developed in the University of Bristol module EMAT30008: Scientific Computing 
# Author: Ben Winstanley, yy23737@bristol.ac.uk

# TODO: go through euler_step and solve_to and reformat

# IMPORTS
import numpy as np
from scipy.optimize import minimize
from typing import Literal

# Week 14
def euler_step(ode_func,
               x: np.ndarray, 
               t: float, 
               h: float) -> np.ndarray:
    '''Perform single Euler step from `x` with step size `h` for the ODE `x' = ode_func(x,t)`. '''
    x_next = x + h*ode_func(x,t)
    return x_next

def rk4_step(ode_func, 
             x: np.ndarray, 
             t: float, 
             h: float) -> np.ndarray:
    '''Perform single RK4 step from `x` with step size `h` for the ODE `x' = ode_func(x,t)`. '''
    k1 = ode_func(x,t)
    k2 = ode_func((x+(h/2)*k1),(t+h/2))
    k3 = ode_func((x+(h/2)*k2),(t+h/2))
    k4 = ode_func((x+h*k3),(t+h))

    x_next = x + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    return x_next

def rk2_step(ode_func, 
             x: np.ndarray, 
             t: float, 
             h: float) -> np.ndarray:
    '''Perform single RK2 step from `x` with step size `h` for the ODE `x' = ode_func(x,t)`. '''
    k1 = ode_func(x,t)
    k2 = ode_func((x+(h/2)*k1),(t+h/2))

    x_next = x + h*k2
    return x_next


def solve_to(ode_func, 
             x_init: np.ndarray,
             t_init: float, 
             t_final: float, 
             deltat_max: float, 
             method: Literal['Euler', 'RK4', 'RK2'] = 'RK4') -> np.ndarray:
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
    if len(ode_func(x_init, t_init)) != len(x_init):
        raise Exception("ODE function and initial condition dimensions do not match.")

    # Time intervals are constant, use step size h = deltat_max
    h = deltat_max
    # Take steps of h until final time value is just less than t_final
    t_vals = np.arange(t_init, t_final - h/100, h)

    # Initialise x store
    x_store, x = x_init, x_init

    # Choose timestepping method as lambda function
    if method=='Euler': # Euler time-step method
        time_step = lambda x, t, h: euler_step(ode_func, x, t, h)
    elif method=='RK4': # 4th Order Runge-Kutta method
        time_step = lambda x, t, h: rk4_step(ode_func, x, t, h)
    elif method=='RK2': # 2th Order Runge-Kutta method
        time_step = lambda x, t, h: rk2_step(ode_func, x, t, h)
    
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
                          params: np.ndarray, 
                          num_loops_needed: int = 10,
                          phase_condition: Literal['constant', 'derivative']='derivative',
                          constant_value: float = None,
                          deltat_max: float = 0.01):
    ''' 
    Computes the value of an objective function used to search for limit cycle solutions of ODEs.

    A limit cycle will have initial state `u0` and period `T` such that this function returns zero.

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
    float
        Value of the objective function `f = (G_collection^2)/size(G_collection) + phi^2` (to be minimized).
    
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
    u0 after time T, and our solution is thus periodic. We compute G for a range of num_loops_needed
    many final times T, 2T,..., and collect them into G_collection

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
    T, u0 = params[0], params[1:]

    # Establish empty array to hold Gs
    num_variables = len(u0)
    G_collection = np.zeros([num_loops_needed*num_variables])

    # Loop over number of loops checked
    for i in range(num_loops_needed):
        # Compute G by calling the solver to solve until time (i+1)*T. Solve with RK4 and deltat_max supplied
        solution = solve_to(ode_func=ode_func, x_init=u0, t_init=0, 
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

    # Introduce a regularisation constant for the period in our objective function
    alpha = 1e-5

    # Return sum of squares of G_collection and phi, plus the period times some weight
    return np.sum(np.square(G_collection))/len(G_collection) + 10*phi**2 + alpha*T



def find_limit_cycle(ode_func, 
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
    of a limit cycle of that ODE. Uses scipy.optimize.root with limit_cycle_condition to converge
    towards a limit cycle starting with the supplied guess. If convergence is successful, returns
    the period and starting state of the limit cycle it located.

    -----
    Parameters
    -----
    ode_func : function
        Definition function for the RHS of the ODE `x' = ode_func(x,t)`.
    init_point_guess : 1-D Numpy array
        Initial guess for the starting state of a limit cycle.
    init_period_guess : float
        Initial guess for the period of the limit cycle.
    phase_condition : 'constant' or 'derivative' 
        Specifies the type of phase condition used
         to select a distinct limit cycle, methods specified below.
    constant_value : float
        Value that the first state variable of `u0` should have. Must be supplied
         if phase_condition == 'constant'.
    deltat_max : float
        Step size used by solve_to numerical ODE solver to compute the solution of the  
         supplied ODE starting from `u0`.
    print_findings: bool
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
    >>> find_limit_cycle(shm, np.array(([5,1])), 10, 1, 'constant', 4)
    A limit cycle was found:
    Period: 12.566370606193304 ,
    Starting state: [4.00000009 0.44653776] .

    (12.566370606193304, array([4.00000009, 0.44653776]))

    -----
    Notes
    -----
    We minimize the objective function specified in limit_cycle_condition, and specify bounds
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
        Describes the objective function we minimize to find limit cycles.
    
    '''
    # Error Messages
    if init_period_guess <= deltat_max:
        raise Exception("Initial period guess is less than step size.")
    
    # Establish lambda function for use with scipy.optimize.root, means we only vary params
    objective_function = lambda params: limit_cycle_condition(ode_func=ode_func,
                                                              num_loops_needed=num_loops_needed,
                                                              params = params,
                                                              phase_condition=phase_condition,
                                                              constant_value=constant_value,
                                                              deltat_max=deltat_max)


    # Pack init_point_guess and init_period_guess into params
    init_params = np.concatenate(([init_period_guess], init_point_guess))
    
    # Specify lower bound for limit cycle period to be greater than 10*deltatmax
    bounds = [(10*deltat_max, None)] + [(None,None) for i in range(len(init_point_guess))]

    # Minimize the limit cycle objective function with above bounds and initial guess
    result = minimize(fun=objective_function, x0=init_params, bounds=bounds)

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
            

# Week 19
def finite_diff_bvp_solver(diffusivity: float,
                           num_grid_points: int,
                           grid_bounds: list[float],
                           left_boundary_vals: list[float],
                           right_boundary_vals: list[float],
                           left_boundary_type: Literal['Dirichlet', 'Neumann', 'Robin'] = 'Dirichlet', 
                           right_boundary_type: Literal['Dirichlet', 'Neumann', 'Robin'] = 'Dirichlet',
                           q_param: float = None,
                           q_func = None,
                           q_nonlinear: bool = False,
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


    Insert extensive docstring here
    boundary types need to be dirichlet, neumann or robin, default dirichlet for both
    do checks for input types
    boundary values is a list of 2 with 2 floats? worth unpacking?
    if robin then check for 2 inputs for each vals
    check for num grid points bigger than 3
    maybe change grid bounds to be seperate, and check b>a
    can we check for q func and guess functions delivering floats?
    do check on q for taking 3 arguments, x, u and mu, raise error if not
    '''
    # Error Messages
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


    D = diffusivity
    N = num_grid_points
    a, b = grid_bounds

    # Establish x grid
    x_vals = np.linspace(a,b,N+1)
    delta_x = (b-a)/N

    # Calculate number of interior grid points based on boundary conditions
    N_interior = N-1
    x_vals_interior = x_vals
    if left_boundary_type != 'Dirichlet':
        N_interior += 1  # Include additional ghost point (on left)
    else:
        x_vals_interior = x_vals_interior[1:]
    if right_boundary_type != 'Dirichlet':
        N_interior += 1  # Include additional ghost point (on right)
    else:
        x_vals_interior = x_vals_interior[:-1]
    
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
            if (left_boundary_type == 'Dirichlet') and (right_boundary_type == 'Dirichlet'):
                # Establish function connecting both points
                def guess_function(x):
                    return left_boundary_vals[0] + ((right_boundary_vals[0]-left_boundary_vals[0])/(b-a))*(x-a)
            elif left_boundary_type == 'Dirichlet':
                # Establish function with constant left boundary value
                def guess_function(x):
                    return left_boundary_vals[0]*np.ones([len(x)])
            elif right_boundary_type == ' Dirichlet':
                # Establish function with constant right boundary value
                def guess_function(x):
                    return right_boundary_vals[0]*np.ones([len(x)])
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
    if left_boundary_type == 'Dirichlet':
        u_interior = np.concatenate(([left_boundary_vals[0]], u_interior))
    if right_boundary_type == 'Dirichlet':
        u_interior = np.concatenate((u_interior, [right_boundary_vals[0]]))
    
    return u_interior, x_vals    
        
