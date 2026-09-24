
import torch, math
from torch import nn

@torch.no_grad()

def _poly_kernel(X, Y):
    # KID uses k(u,v) = ((u·v / d) + 1)^3  (Bińkowski et al., 2018)
    d = X.size(1)
    return ((X @ Y.T) / d + 1.0) ** 3

@torch.no_grad()
def kid(xf, yf):
    m, n = xf.size(0), yf.size(0)
    Kxx = _poly_kernel(xf, xf); Kyy = _poly_kernel(yf, yf); Kxy = _poly_kernel(xf, yf)
    term_x = (Kxx.sum() - Kxx.diag().sum()) / (m * (m - 1))
    term_y = (Kyy.sum() - Kyy.diag().sum()) / (n * (n - 1))
    term_xy = (2.0 * Kxy.mean())
    return (term_x + term_y - term_xy).item()       # unbiased MMD^2




def total_spin_spin_correlation(samples):

    N_samples = samples.shape[0]
    corr_matrix = np.einsum('ki,kj->ij', samples, samples) / N_samples
    return corr_matrix