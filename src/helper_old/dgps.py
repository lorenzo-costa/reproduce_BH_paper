"""Data generation classes for simulation studies."""

import numpy as np
from abc import ABC, abstractmethod
from scipy import stats


class DataGenerator(ABC):
    """Abstract base class for data generation.

    All data generators must implement three methods:
    - generate(n): Create n samples from the distribution
    - name(): Return a descriptive name for reporting

    This ensures all generators can be used interchangeably in simulation
    studies without modifying the simulation code.
    """

    @abstractmethod
    def generate(self, n, rng=None):
        """Generate n samples from the distribution.

        Parameters
        ----------
        n : int
            Number of samples to generate
        rng : np.random.Generator, optional
            Random number generator. If None, uses np.random.default_rng (no seed)

        Returns
        -------
        np.ndarray
            Array of n samples from the distribution
        """
        pass

    @property
    @abstractmethod
    def name(self):
        """Return descriptive name of the distribution.

        Returns
        -------
        str
            Human-readable name including parameter values
        """
        pass

    @property
    @abstractmethod
    def null_value(self):
        """Return the true parameter value under H0.

        For testing whether a location parameter equals a specific value,
        this returns that true value. For example, if testing H0: μ = μ0,
        this returns μ0.

        Returns
        -------
        float
            True value of the location parameter
        """
        pass


class NormalGenerator(DataGenerator):
    """Generate data from a normal distribution.

    The normal distribution is symmetric and serves as the baseline case
    for most parametric tests. It is characterized by its mean (loc) and
    standard deviation (scale).

    Parameters
    ----------
    loc : float, default=0
        Mean of the distribution (location parameter)
    scale : float, default=1
        Standard deviation of the distribution (scale parameter)

    Examples
    --------
    >>> gen = NormalGenerator(loc=5, scale=2)
    >>> data = gen.generate(100)
    >>> gen.name()
    'Normal(mu=5, sigma=2)'
    """

    def __init__(self, loc=0, scale=1):
        self.loc = loc
        self.scale = scale

    def generate(self, n=None, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        if n is None:
            n = self.loc.shape[0] if isinstance(self.loc, np.ndarray) else 1
        return rng.normal(self.loc, self.scale, n)

    @property
    def name(self):
        return f"Normal(mu={self.loc}, sigma={self.scale})"

    @property
    def null_value(self):
        return self.loc


def generate_means(m, m0, scheme, L, rng=None, rounding_biase_correction=False):
    """Generate a simulation scenario from a Gaussian sample.

    Parameters
    ----------
    m : int
        Number of hypotheses to test
    m0 : int
        Number of true null hypotheses
    scheme : str
        The testing scheme to use ('E', 'D', or 'I')
    L : int
        Non-zero mean upper bound
    rng : np.random.Generator, optional
        Random number generator. If None, uses np.random.default_rng()
    Returns
    -------
    np.ndarray
        Array of means for the m hypotheses

    Examples
    --------
    >>> means = generate_means(m=4, m0=1, scheme='E', L=4, rng=np.random.default_rng(42))
    >>> means
    array([2., 3., 4., 0.])
    >>> means = generate_means(m=4, m0=4, scheme='D', L=8, rng=np.random.default_rng(42))
    >>> means
    array([0., 0., 0., 0.])
    """

    if not isinstance(m, int) or m <= 0:
        raise ValueError("m must be a positive integer")
    if not isinstance(L, int) or L <= 0:
        raise ValueError("L must be a positive integer")
    if not isinstance(m0, int) or m0 < 0 or m0 > m:
        raise ValueError("m0 must be an integer between 0 and m inclusive")
    if rng is None:
        rng = np.random.default_rng()

    if scheme not in ["E", "D", "I"]:
        raise ValueError("scheme must be one of 'E', 'D', or 'I'")

    means = np.zeros(m)
    if m0 == m:
        return means

    m1 = m - m0
    levels = np.array([L / 4, L / 2, 3 * L / 4, L])

    if scheme == "D":  # Linearly Decreasing
        # more hp closer to 0. divide non nulls as: 4k, 3k, 2k, k
        # sum = 10k = non nulls, so k = non-nulls/10
        base = m1 / 10
        counts = np.array([4 * base, 3 * base, 2 * base, base])
    elif scheme == "E":  # Equal
        counts = np.full(4, m1 / 4)
    else:
        base = m1 / 10
        counts = np.array([base, 2 * base, 3 * base, 4 * base])

    counts = counts.astype(int)

    # Adjust for rounding errors
    if rounding_biase_correction is False:
        diff = m1 - counts.sum()
        if diff > 0:
            for i in range(diff):
                counts[-1 - i] += 1
        elif diff < 0:
            for i in range(-diff):
                counts[i] -= 1
    else:
        # this is what i belive the code should be but it does not reproduce paper
        # implementation. i think the difference is in how they handle the case of m
        # small. e.g. for m=4 and m0=2 my code would give [1, 2, 0, 0] but the code
        # that correctly repoduces the paper results gives [3, 4, 0, 0]
        # the difference is that with my code I get very low power for m=4, with theirs
        # I get power almost 1
        diff = m1 - counts.sum()
        if scheme in ["E", "I"]:
            if diff > 0:
                for i in range(diff):
                    counts[-1 - i] += 1
            elif diff < 0:
                for i in range(-diff):
                    counts[i] -= 1
        elif scheme in "D":
            if diff > 0:
                for i in range(diff):
                    counts[i] += 1
            elif diff < 0:
                for i in range(-diff):
                    counts[-1 - i] -= 1

    idx = 0
    for pos, count in zip(levels, counts):
        means[idx : idx + count] = pos
        idx += count

    return means


def compute_p_values(z_scores):
    return 2 * (1 - stats.norm.cdf(np.abs(z_scores)))
