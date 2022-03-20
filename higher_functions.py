
import numpy as np
from scipy.optimize import curve_fit

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
def rk4(Xbc, nodes_target, rparams):
    ri, rf, dr = rparams
    steps = int((rf - ri)//dr)
    rinterv = np.linspace(ri, rf, steps, endpoint=True)

    # Initialize the arrays
    Xs = np.zeros((4, steps))

    # Boundary conditions at r = ri
    Xs[..., 0] = Xbc

    # Perform RK4
    bpoint, nodes, peaks = 0, -1, -1
    for step in np.arange(1, steps):
        rnow = rinterv[step]
        Xs[..., step] = rk4step(rnow, Xs[..., step - 1], dr)

        if np.isnan(Xs[..., step][0]):
            break 

        # Find number of nodes and peaks (points with zero derivative)
        nodes, peaks = nnodes(Xs[0][:step]), npeaks(Xs[0][:step])
    
        # The breakpoint 'bpoint' occurs when the solution with the right number of nodes stops converging to zero at r -> infty
        sign_nodes = np.power(-1, nodes_target)
        if (sign_nodes*Xs[..., step][0] > sign_nodes*Xs[..., step - 1][0] or sign_nodes*Xs[..., step][0] < 0) and nodes == nodes_target and peaks == nodes:
            bpoint = step
            return Xs, bpoint, nodes

    return Xs, bpoint, nodes

# Step of Monte Carlo for shooting method
def mc_step(phi0, bpointold, nodes_target, rparams):
    Xbc_step = np.array([1, 0, phi0, 0])
    Xs_step, bpoint, nodes = rk4(Xbc_step, nodes_target, rparams)

    # If the breakpoint happens for higher values of r, keep the new solution
    if bpoint > bpointold and nodes == nodes_target:
        return True, bpoint, nodes
    else:
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

# Number of nodes of a 1d function described by an array Xs
def nnodes(Xs):
    Xs_prod = Xs[:len(Xs) - 1]*Xs[1:len(Xs)]
    return (Xs_prod < 0).sum()

# Number of peaks (points with zero derivative) of a 1d function described by an array Xs
def npeaks(Xs):
    Xs_diffs = Xs[1:len(Xs)] - Xs[:len(Xs) - 1]
    Xs_diffs_prod = Xs_diffs[:len(Xs_diffs) - 1]*Xs_diffs[1:len(Xs_diffs)]
    return (Xs_diffs_prod < 0).sum()
