def umi_noise(B, C, depth=0.5):

    B = np.random.binomial(B.astype(int), depth)
    C = np.random.binomial(C.astype(int), depth)

    return B, C
