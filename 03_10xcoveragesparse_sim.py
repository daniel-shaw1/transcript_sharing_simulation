def drop_coverage(B, C, p_detect=0.1):

    mask = np.random.rand(*B.shape) < p_detect

    B = B * mask
    C = C * mask

    return B, C
