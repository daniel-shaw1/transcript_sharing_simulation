#!/usr/bin/env python3
"""
simulate_dataset.py

Synthetic generator for genoinformativity pipeline benchmarking.

Simulates:
- recombination-aware haplotypes
- gene expression (NB/Poisson)
- transcript sharing (g parameter)
- allele-specific UMIs (B6 / CAST)
- 10x sparsity (SNP dropout + UMI sampling)
- optional cis-eQTL confounding

Outputs:
- cell_gene_ase.tsv
- ground_truth_g.tsv
- optional haplotype truth (cell_bin_haplotypes.tsv-like)
"""

import numpy as np
import pandas as pd


# -----------------------------
# CONFIG
# -----------------------------

N_CELLS = 3000
N_GENES = 2000
N_CHROMS = 5
GENES_PER_CHR = N_GENES // N_CHROMS

BIN_SIZE = 10  # pseudo-bin resolution per chromosome
RECOMB_RATE = 1e-3

P_SNP_DETECT = 0.05
UMI_DEPTH = 0.7

CIS_EFFECT_SD = 0.15
ADD_CIS_EFFECT = True

np.random.seed(42)


# -----------------------------
# 1. SIMULATE GENE POSITIONS
# -----------------------------

genes = []
for g in range(N_GENES):
    chrom = g // GENES_PER_CHR
    pos = g % GENES_PER_CHR
    genes.append((f"gene_{g}", chrom, pos))

gene_df = pd.DataFrame(genes, columns=["gene", "chrom", "pos"])


# -----------------------------
# 2. TRUE G VALUES
# -----------------------------

g_true = np.random.beta(2, 5, size=N_GENES)


# -----------------------------
# 3. SIMULATE RECOMBINATION HAPLOTYPES
# -----------------------------

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

        hap = simulate_haplotype(GENES_PER_CHR, RECOMB_RATE)
        haplotypes[c, start:end] = hap


# -----------------------------
# 4. BASE EXPRESSION
# -----------------------------

mu = np.random.lognormal(mean=1.0, sigma=0.8, size=N_GENES)


# -----------------------------
# 5. SIMULATE ASE COUNTS
# -----------------------------

rows = []

for c in range(N_CELLS):

    for j in range(N_GENES):

        expr = np.random.poisson(mu[j])

        if expr == 0:
            continue

        H = haplotypes[c, j]

        # base genotype-dependent probability
        p = 0.5 + 0.5 * g_true[j] * (2 * H - 1)

        # cis-eQTL confounding
        if ADD_CIS_EFFECT:
            p += np.random.normal(0, CIS_EFFECT_SD)

        p = np.clip(p, 1e-3, 1 - 1e-3)

        # allele sampling
        B = np.random.binomial(expr, p)
        C = expr - B

        # SNP detectability (10x sparsity)
        if np.random.rand() > P_SNP_DETECT:
            B = 0
            C = 0

        # UMI sampling noise
        B = np.random.binomial(B, UMI_DEPTH)
        C = np.random.binomial(C, UMI_DEPTH)

        if B + C == 0:
            continue

        rows.append([
            f"cell_{c}",
            f"gene_{j}",
            B,
            C,
            H,
            g_true[j]
        ])


df = pd.DataFrame(rows, columns=[
    "cell",
    "gene",
    "B6_umi",
    "CAST_umi",
    "haplotype",
    "g_true"
])


# -----------------------------
# 6. SAVE OUTPUTS
# -----------------------------

df[[
    "cell",
    "gene",
    "B6_umi",
    "CAST_umi"
]].to_csv(
    "cell_gene_ase.tsv",
    sep="\t",
    index=False
)

pd.DataFrame({
    "gene": [f"gene_{i}" for i in range(N_GENES)],
    "g_true": g_true
}).to_csv(
    "ground_truth_g.tsv",
    sep="\t",
    index=False
)

hap_df = pd.DataFrame({
    "cell": [f"cell_{i}" for i in range(N_CELLS)]
})

hap_df.to_csv("simulation_summary.txt", index=False)

print("Simulation complete.")
print("Saved:")
print(" - cell_gene_ase.tsv")
print(" - ground_truth_g.tsv")
