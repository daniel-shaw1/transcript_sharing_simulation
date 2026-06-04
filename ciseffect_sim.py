cis_effect = np.random.normal(0, 0.2, size=n_genes)

p = p + cis_effect
p = np.clip(p, 0, 1)
