# week14funcs.py
# Assignment Week 14 of EMAT30008
# Used in week14.ipynb
# Author: Ben Winstanley, yy23737@bristol.ac.uk

# SEE odefunctions.py FOR UPDATED VERSIONS OF FUNCTIONS

# Imports
import numpy as np


def euler_step(f, x: np.ndarray, t: float, h: float) -> np.ndarray:
    '''
    ODE is x' = f(x,t)
    Performs single Euler step with step size h
    '''
    x_next = x + h*f(x,t)
    return x_next


def solve_to(f, x_init: np.ndarray, t_init: float, t_final: float, deltat_max: float, method: str ='Euler') -> np.ndarray:
    '''
    Solves initial value ODE problem x' = f(x,t)

    Initial condition is x_init at time t_init.

    Solves until t_final and returns [t, x] with a new row for each new timestep.

    Requires numpy to be imported already.
    Uses either 'Euler' time step method (default), or 'RK4' 4th order Runge-Kutta method.
    '''
    if deltat_max >= (t_final - t_init):
        raise Exception("Maximum time-step deltat_max >= total time interval")

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
            x = euler_step(f, x, t, h)
            x_store = np.vstack((x_store, x))

        # Take final step with h <= deltat_max
        h_final = t_final - t_vals[-1]
        x = euler_step(f, x, t, h_final)
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

        # Loop through each timestep
        for t in t_vals[:-1]:
            # Calculate k values:
            k1 = f(x,t)
            k2 = f((x+(h/2)*k1),(t+h/2))
            k3 = f((x+(h/2)*k2),(t+h/2))
            k4 = f((x+h*k3),(t+h))

            # Calculate next value
            x = x + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

            # Update store
            x_store = np.vstack((x_store, x))
        
        # Take final step with h <= deltat_max
        h_final = t_final - t_vals[-1]

        k1 = f(x,t)
        k2 = f((x+(h_final/2)*k1),(t+h_final/2))
        k3 = f((x+(h_final/2)*k2),(t+h_final/2))
        k4 = f((x+h_final*k3),(t+h_final))

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

        # Loop through each timestep
        for t in t_vals[:-1]:
            # Calculate k values:
            k1 = f(x,t)
            k2 = f((x+(h/2)*k1),(t+h/2))

            # Calculate next value
            x = x + k2*h

            # Update store
            x_store = np.vstack((x_store, x))

        # Take final step with h <= deltat_max
        h_final = t_final - t_vals[-1]
        k1 = f(x,t)
        k2 = f((x+(h_final/2)*k1),(t+h_final/2))
        x = x + k2*h_final

        # Update store
        x_store = np.vstack((x_store, x))

        t_vals = np.append(t_vals, t_final)

        # Combine into one store
        store = np.concatenate((np.array([t_vals]).T, x_store), axis=1)

        return store

    else:
        raise Exception("Not a valid method, please enter 'Euler','RK4' or 'RK2 ")

# SEE odefunctions.py FOR UPDATED VERSIONS OF FUNCTIONS