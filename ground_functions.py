import numpy as np
from scipy.optimize import curve_fit

# # Interval of solution and number of steps
# ri, rf, dr = 0, 20, 0.01 # Change dr to 0.001 for high resolution
# steps = int((rf - ri)//dr)
# rinterv = np.linspace(ri, rf, steps, endpoint=True)

# Right-hand side of ODEs from Eq. (20) in my "Notes on Axion Stars", here X = [\tilde{\Psi}, \chi, \tilde{\phi}, \xi]
def f(r, X):
    f1 = X[1]
    f2 = -2*X[1]/r + 2*X[2]*X[0]
    f3 = X[3]
    f4 = -2*X[3]/r + np.power(X[0],2)
    return np.array([f1, f2, f3, f4])

# Step of RK4
def rk4step(r, X, dr):
    k1 = dr*f(r, X)
    k2 = dr*f(r + dr/2, X + k1/2)
    k3 = dr*f(r + dr/2, X + k2/2)
    k4 = dr*f(r + dr, X + k3)
    return X + (k1 + 2*k2 + 2*k3 + k4)/6

# Full RK4
def rk4(Xbc, rparams): # rparams in the format ri, rf, dr
    ri, rf, dr = rparams
    steps = int((rf - ri)//dr)
    rinterv = np.linspace(ri, rf, steps, endpoint=True)
    
    # Initialize the arrays
    Xs = np.zeros((4, steps))

    # Boundary conditions at r = ri
    Xs[..., 0] = Xbc

    # Perform RK4
    bpoint = 0
    for step in np.arange(1, steps):
        rnow = rinterv[step]
        Xs[..., step] = rk4step(rnow, Xs[..., step - 1], dr)
    
        # The breakpoint 'bpoint' occurs when the solution stops converging to zero at r -> infty
        if Xs[..., step][0] > Xs[..., step - 1][0] or Xs[..., step][0] < 0:
            bpoint = step
            return Xs, bpoint

    return Xs, bpoint

# Step of Monte Carlo for shooting method
def mc_step(phi0, bpointold, rparams):
    Xbc_step = np.array([1, 0, phi0, 0])
    Xs_step, bpoint = rk4(Xbc_step, rparams)

    # If the breakpoint happens for higher values of r, keep the new solution
    if bpoint > bpointold:
        return True, bpoint
    elif bpoint <= bpointold:
        return False, bpointold

# Asymptotic behavior of phi
def fit_asymp(r, C, epsilonb):
        return C/r + epsilonb

def find_epsilonb(rinterv_loc, Xs):
    # Find epsilonb from asymptotic behavior of phi
    tail_ini = 100
    r_asymp, phi_asymp = rinterv_loc[-tail_ini:-1], Xs[2].T[-tail_ini:-1]
    popt, pcov = curve_fit(fit_asymp, r_asymp, phi_asymp)
    # The error from the fit is epsilonb_stdev
    epsilonb, epsilonb_stdev = popt[1], np.sqrt(np.diag(pcov))[1] 
    return epsilonb, epsilonb_stdev
