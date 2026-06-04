def simulate_gene_expression(n_cells, n_genes, haplotypes, g):

    B = np.zeros((n_cells, n_genes))
    C = np.zeros((n_cells, n_genes))

    mu = np.random.lognormal(mean=1, sigma=1, size=n_genes)

    for c in range(n_cells):

        for j in range(n_genes):

            expr = np.random.poisson(mu[j])

            if expr == 0:
                continue

            H = haplotypes[c, j]

            p = 0.5 + 0.5 * g[j] * (2*H - 1)
            p = np.clip(p, 1e-3, 1-1e-3)

            B[c,j] = np.random.binomial(expr, p)
            C[c,j] = expr - B[c,j]

    return B, C
