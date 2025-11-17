"""Evaluation metrics for simulation studies."""

import numpy as np
from abc import ABC, abstractmethod


class Metric(ABC):
    """Abstract base class for evaluation metrics.

    All metrics must implement two methods:
    - __call__(rejected, true_values): Compute the metric given rejections and true values
    - name(): Return a descriptive name for reporting
    """

    @abstractmethod
    def __call__(self, rejected, true_values):
        """Compute the evaluation metric.

        Parameters
        ----------
        rejected : np.ndarray
            Boolean array indicating which hypotheses are rejected
        true_values : np.ndarray
            Array of true means for each hypothesis; non-zero indicates
            true alternatives

        Returns
        -------
        float
            Computed metric value
        """
        pass

    @property
    @abstractmethod
    def name(self):
        """Return descriptive name of the metric.

        Returns
        -------
        str
            Human-readable name of the metric
        """
        pass


class Power(Metric):
    """Statistical power metric.

    Power is defined as the proportion of true alternative hypotheses
    that are correctly rejected.

    Returns
    -------
    float
        Statistical power

    Examples
    --------
    >>> power_metric = Power()
    >>> rejected = np.array([True, False, True, False])
    >>> true_values = np.array([2.0, 0.0, 3.0, 0.0])
    >>> power = power_metric(rejected, true_values)
    >>> power
    1.0
    """

    @property
    def name(self):
        return "Power"

    def __call__(self, rejected, true_values):
        """Compute the statistical power.

        Power is defined as the proportion of true alternative hypotheses
        that are correctly rejected.

        Parameters
        ----------
        rejected : np.ndarray
            Boolean array indicating which hypotheses are rejected
        true_values : np.ndarray
            Array of true means for each hypothesis; non-zero indicates
            true alternatives

        Returns
        -------
        float
            Statistical power
        """
        truth_mask = true_values != 0
        power = np.mean(rejected[truth_mask]) if np.sum(truth_mask) > 0 else 0.0

        return power


class FalseDiscoveryRate(Metric):
    """False Discovery Rate metric.

    False Discovery Rate is defined as the proportion of false rejections
    among all rejections.

    Returns
    -------
    float
        False Discovery Rate

    Examples
    --------
    >>> fdr_metric = FalseDiscoveryRate()
    >>> rejected = np.array([True, False, True, False])
    >>> true_values = np.array([0.0, 0.0, 3.0, 0.0])
    >>> fdr = fdr_metric(rejected, true_values)
    >>> fdr
    0.5
    """

    @property
    def name(self):
        return "FDR"

    def __call__(self, rejected, true_values):
        """Compute the False Discovery Rate.

        False Discovery Rate is defined as the proportion of false rejections
        among all rejections.

        Parameters
        ----------
        rejected : np.ndarray
            Boolean array indicating which hypotheses are rejected
        true_values : np.ndarray
            Array of true means for each hypothesis; non-zero indicates
            true alternatives

        Returns
        -------
        float
            False Discovery Rate
        """
        if np.sum(rejected) == 0:
            return 0.0

        false_rejections = np.sum(rejected & (true_values == 0))
        fdr = false_rejections / np.sum(rejected)

        return fdr


class TrueRejections(Metric):
    """True Rejections metric.

    True Rejections is defined as the total number of true alternative
    hypotheses that are correctly rejected.

    Returns
    -------
    int
        Number of true rejections

    Examples
    --------
    >>> tr_metric = TrueRejections()
    >>> rejected = np.array([True, False, True, False])
    >>> true_values = np.array([2.0, 0.0, 3.0, 0.0])
    >>> true_rejections = tr_metric(rejected, true_values)
    >>> true_rejections
    2
    """

    @property
    def name(self):
        return "True Rejections"

    def __call__(self, rejected, true_values):
        """Compute the number of true rejections.

        True Rejections is defined as the total number of true alternative
        hypotheses that are correctly rejected.

        Parameters
        ----------
        rejected : np.ndarray
            Boolean array indicating which hypotheses are rejected
        true_values : np.ndarray
            Array of true means for each hypothesis; non-zero indicates
            true alternatives

        Returns
        -------
        int
            Number of true rejections
        """
        truth_mask = true_values != 0
        true_rejections = np.sum(rejected[truth_mask])

        return true_rejections


class RejectionsNumber(Metric):
    """Total Rejections metric.

    Total Rejections is defined as the total number of hypotheses
    that are rejected.

    Returns
    -------
    int
        Total number of rejections

    Examples
    --------
    >>> tr_metric = RejectionsNumber()
    >>> rejected = np.array([True, False, True, False])
    >>> true_values = np.array([2.0, 0.0, 3.0, 0.0])
    >>> total_rejections = tr_metric(rejected, true_values)
    >>> total_rejections
    2
    """

    @property
    def name(self):
        return "Total Rejections"

    def __call__(self, rejected, true_values):
        """Compute the total number of rejections.

        Total Rejections is defined as the total number of hypotheses
        that are rejected.

        Parameters
        ----------
        rejected : np.ndarray
            Boolean array indicating which hypotheses are rejected
        true_values : np.ndarray
            Array of true means for each hypothesis; non-zero indicates
            true alternatives

        Returns
        -------
        int
            Total number of rejections
        """
        total_rejections = np.sum(rejected)

        return total_rejections