n_cells = 5000
n_genes = 2000

# true genoinformativity
g_true = np.random.beta(2, 5, size=n_genes)

# simulate haplotypes
haplotypes = np.random.randint(0, 2, size=(n_cells, n_genes))

B, C = simulate_gene_expression(
    n_cells,
    n_genes,
    haplotypes,
    g_true
)

B, C = drop_coverage(B, C, p_detect=0.05)
B, C = umi_noise(B, C)
