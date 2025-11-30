"""Multiple hypothesis testing classes for simulation studies.

This module defines an abstract base class for multiple hypothesis testing
and provides concrete implementations for common tests. All tests follow
the same interface, making them interchangeable in simulation studies.
"""

from abc import ABC, abstractmethod
import numpy as np


class MultipleTesting(ABC):
    """Abstract base class for multiple testing procedures.

    All statistical test must implement two methods:
    - __call__(p_values, alpha): Perform multiple testing and return reject True/False
    - name(): Returns descriptive name
    """

    @abstractmethod
    def __call__(self, p_values, alpha):
        """Apply multiple testing correction.

        Parameters
        ----------
        p_values : np.ndarray
            Array of p-values from individual tests
        alpha : float
            Significance level for controlling FWER or FDR

        Returns
        -------
        np.ndarray
            Boolean array indicating which hypotheses are rejected
        """
        pass

    @property
    @abstractmethod
    def name(self):
        """Return descriptive name of the procedure.

        Returns
        -------
        str
            Human-readable name of the procedure
        """
        pass


class Bonferroni(MultipleTesting):
    """Bonferroni correction for multiple testing.

    This method controls the family-wise error rate (FWER) by adjusting
    the significance level for each individual test. A hypothesis is
    rejected if its p-value is less than alpha / m, where m is the
    total number of tests.

    Examples
    --------
    >>> from methods import BonferroniCorrection
    >>> bonf = BonferroniCorrection()
    >>> pvals = np.array([0.01, 0.04, 0.03, 0.20])
    >>> alpha = 0.05
    >>> bonf(pvals, alpha)
    array([ True, False, False, False])
    >>> bonf.name
    'Bonferroni Correction'
    """

    def __call__(self, p_values, alpha):
        assert np.all((p_values >= 0) & (p_values <= 1)), (
            "p-values must be between 0 and 1"
        )
        m = len(p_values)
        threshold = alpha / m
        return p_values <= threshold

    @property
    def name(self):
        return "Bonferroni Correction"


class BonferroniHochberg(MultipleTesting):
    """Bonferroni-Hochberg correction for multiple testing.

    This is a step-up method introduced in Hochberg, 1988, which controls the
    family-wise error rate (FWER) under the assumption of independence of the p-values.
    It works by sorting the p-values, finding the largest k such that
    p_(k) <= k/(m-k+1) * alpha and rejecting all hypothesis H_(i) i=1, ..., k

    Examples
    --------
    >>> from methods import BonferroniHochberg
    >>> bonf = BonferroniHochberg()
    >>> pvals = np.array([0.01, 0.04, 0.03, 0.20])
    >>> alpha = 0.05
    >>> bonf(pvals, alpha)
    array([ True, False, True, False])
    >>> bonf.name
    'Bonferroni Correction'
    """

    def __call__(self, p_values, alpha):
        sorted_pvalues = np.sort(p_values)
        m = len(p_values)
        threshold = -1
        for k in range(m - 1, -1, -1):
            # add one because of zero-based indexing of python
            if sorted_pvalues[k] <= alpha / (m - k):
                threshold = sorted_pvalues[k]
                break

        return p_values <= threshold

    @property
    def name(self):
        return "Bonferroni-Hochberg Correction"


class BenjaminiHochberg(MultipleTesting):
    """Benjamini-Hochberg correction for multiple testing.

    This is a step-up method introduced in Benjamini and Hochberg, 1995, which
    controls the false discovery rate (FDR) under the assumption of independence
    of the test statistics. It works by sorting the p-values, finding the largest k
    such that p_(k) <= k/m * alpha and rejecting all hypothesis with p_(i)<=p_(k)

    Examples
    --------
    >>> from methods import BonferroniHochbergCorrection
    >>> bonf = BenjaminiHochberg()
    >>> pvals = np.array([0.01, 0.04, 0.03, 0.20])
    >>> alpha = 0.05
    >>> bonf(pvals, alpha)
    array([ True, False, True, False])
    >>> bonf.name
    'Benjamini-Hochberg Correction'
    """

    def __call__(self, p_values, alpha):
        sorted_pvalues = np.sort(p_values)
        m = len(p_values)
        threshold = -1
        for k in range(m - 1, -1, -1):
            # add one because of zero-based indexing of python
            if sorted_pvalues[k] <= alpha * (k + 1) / m:
                threshold = sorted_pvalues[k]
                break

        return p_values <= threshold

    @property
    def name(self):
        return "Benjamini-Hochberg Correction"
