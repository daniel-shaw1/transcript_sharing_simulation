#!/usr/bin/env python3

import numpy as np
import pandas as pd

# -----------------------
# CONFIG
# -----------------------

N_CELLS = 2000
N_GENES = 1000
N_CHROMS = 5
GENES_PER_CHR = N_GENES // N_CHROMS

RECOMB_RATE = 1e-3

# cytoplasmic diffusion strength
ALPHA = 0.35
DIFFUSION_STEPS = 3

P_SNP_DETECT = 0.05
UMI_DEPTH = 0.7

np.random.seed(1)


# -----------------------
# 1. GENE ANNOTATION
# -----------------------

genes = []
for g in range(N_GENES):
    chrom = g // GENES_PER_CHR
    pos = g % GENES_PER_CHR
    genes.append((f"gene_{g}", chrom, pos))

gene_df = pd.DataFrame(genes, columns=["gene", "chrom", "pos"])


# -----------------------
# 2. TRUE GENOINFORMATIVITY
# -----------------------

g_true = np.random.beta(2, 5, size=N_GENES)


# -----------------------
# 3. RECOMBINATION HAPLOTYPES
# -----------------------

def simulate_haplotype(n_bins, recomb_rate):

    h = np.zeros(n_bins, dtype=int)
    state = np.random.randint(0, 2)

    for i in range(n_bins):
        if np.random.rand() < recomb_rate:
            state = 1 - state
        h[i] = state

    return h


haplotypes = np.zeros((N_CELLS, N_GENES), dtype=int)

for c in range(N_CELLS):
    for chrom in range(N_CHROMS):

        start = chrom * GENES_PER_CHR
        end = (chrom + 1) * GENES_PER_CHR

        haplotypes[c, start:end] = simulate_haplotype(
            GENES_PER_CHR,
            RECOMB_RATE
        )


# -----------------------
# 4. BASE EXPRESSION
# -----------------------

mu = np.random.lognormal(mean=1.0, sigma=0.8, size=N_GENES)


# -----------------------
# 5. INITIAL TRANSCRIPTS (pre-diffusion)
# -----------------------

T_B6 = np.zeros((N_CELLS, N_GENES))
T_CAST = np.zeros((N_CELLS, N_GENES))

for c in range(N_CELLS):
    for j in range(N_GENES):

        expr = np.random.poisson(mu[j])
        if expr == 0:
            continue

        H = haplotypes[c, j]

        p = 0.5 + 0.5 * g_true[j] * (2 * H - 1)
        p = np.clip(p, 1e-3, 1 - 1e-3)

        B = np.random.binomial(expr, p)

        T_B6[c, j] = B
        T_CAST[c, j] = expr - B


# -----------------------
# 6. CYTOPLASMIC BRIDGE GRAPH
# -----------------------

def adjacency_matrix(n, window=5):

    A = np.zeros((n, n))

    for i in range(n):
        for j in range(max(0, i-window), min(n, i+window+1)):
            if i != j:
                A[i, j] = 1

    # row normalize
    A = A / A.sum(axis=1, keepdims=True)

    return A


A = adjacency_matrix(N_CELLS, window=5)


# -----------------------
# 7. DIFFUSION PROCESS
# -----------------------

def diffuse(T, A, alpha, steps):

    T = T.copy()

    for _ in range(steps):
        T = (1 - alpha) * T + alpha * A @ T

    return T


T_B6 = diffuse(T_B6, A, ALPHA, DIFFUSION_STEPS)
T_CAST = diffuse(T_CAST, A, ALPHA, DIFFUSION_STEPS)


# -----------------------
# 8. OBSERVATION MODEL (10x)
# -----------------------

rows = []

for c in range(N_CELLS):
    for j in range(N_GENES):

        total = T_B6[c, j] + T_CAST[c, j]
        if total < 1:
            continue

        # SNP detectability
        if np.random.rand() > P_SNP_DETECT:
            continue

        # UMI sampling
        total = np.random.binomial(int(total), UMI_DEPTH)

        if total < 1:
            continue

        p = T_B6[c, j] / (T_B6[c, j] + T_CAST[c, j] + 1e-9)

        B = np.random.binomial(int(total), p)
        C = total - B

        rows.append([
            f"cell_{c}",
            f"gene_{j}",
            B,
            C,
            haplotypes[c, j],
            g_true[j]
        ])


df = pd.DataFrame(rows, columns=[
    "cell", "gene", "B6_umi", "CAST_umi", "haplotype", "g_true"
])


# -----------------------
# 9. OUTPUT
# -----------------------

df[[
    "cell",
    "gene",
    "B6_umi",
    "CAST_umi"
]].to_csv("cell_gene_ase.tsv", sep="\t", index=False)

pd.DataFrame({
    "gene": [f"gene_{i}" for i in range(N_GENES)],
    "g_true": g_true
}).to_csv("ground_truth_g.tsv", sep="\t", index=False)

print("Simulation complete with diffusion model.")
