import numpy as np
from numpy.random import RandomState
import warnings


def rarefy(x, depth=1000, iterations=1, seed=42):
    """
    Rarefies a count or frequency vector 'x' by randomly subsampling elements.

    Parameters:
    - x (numpy.ndarray): Input count or frequency vector to be rarefied. Meant to represent gene or species counts.
    - depth (int, optional): The desired rarefaction depth, i.e., the number of elements to subsample.
                             Default is 1000.
    - iterations (int, optional): The number of iterations to perform rarefaction.
                                  Default is 1. If > 1, random list of seeds is generated. Overules seed param.
    - seed (int, optional): Seed for reproducibility of random sampling. Default is 42.

    Returns:
    numpy.ndarray: Rarefied vector with the same length as the input vector 'x'.
                  The result is the mean of rarefied counts over multiple iterations.
                  If the number of iterations exceeds 100000, a warning is printed, and the
                  number of iterations is set to 100000.
                  If the rarefaction depth exceeds the total count in 'x', an array of NaNs with the
                  same length as 'x' is returned.
                  If 'x' has zero counts or length zero, an array of NaNs with the same length as 'x' is returned.
    """

    res = None
    noccur = np.sum(x)
    nvar = len(x)
    # Check for invalid count vector
    if noccur == 0 or nvar == 0:
        print("Invalid count vector x")
        return np.array([np.nan]*nvar)

    # Check if the number of iterations is within a reasonable range
    if iterations > 100000:
        warnings.warn(UserWarning(
            'Max number of iterations allowed is 100000, setting to 100000'))
        iterations = 100000

    # Check if the rarefaction depth exceeds the total count in 'x'
    if depth > noccur:
        return np.array([np.nan]*nvar)
    p = x/noccur
    seeds = np.random.choice(
        100000, size=iterations) if iterations > 1 else [seed]
    # Perform rarefaction for each iteration
    for seed in seeds:
        prng = RandomState(seed)
        choice = prng.choice(nvar, size=depth, p=p)

        # Concatenate the results for each iteration
        if res is None:
            res = np.bincount(choice, minlength=nvar)[np.newaxis, :]
        else:
            res = np.concatenate(
                (res, np.bincount(choice, minlength=nvar)[np.newaxis, :]))
    # Return the mean of rarefied counts over multiple iterations
    return np.nanmean(res, axis=0)
