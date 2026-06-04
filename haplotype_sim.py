import numpy as np

def simulate_haplotype(n_bins, recomb_rate=1e-3):

    H = np.zeros(n_bins, dtype=int)

    current = np.random.randint(0, 2)

    for i in range(n_bins):

        if np.random.rand() < recomb_rate:
            current = 1 - current

        H[i] = current

    return H
