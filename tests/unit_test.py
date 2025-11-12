import pytest
from src.helper_functions.dgps import NormalGenerator, compute_p_values, generate_means
from src.helper_functions.methods import (
    BonferroniHochberg,
    BenjaminiHochberg,
    Bonferroni,
)
from src.helper_functions.metrics import (
    Power,
    FalseDiscoveryRate,
    TrueRejections,
    RejectionsNumber,
)
import numpy as np

scheme_dict = {
    "D": 1,
    "E": 2,
    "I": 3,
}


# Test the normal data generator
def test_normal_data_generator():
    loc = 0
    scale = 1
    rng = np.random.default_rng(42)
    gen = NormalGenerator(loc=loc, scale=scale)
    data = gen.generate(10000, rng=rng)
    assert data.mean() == pytest.approx(loc, abs=0.1)
    assert data.std() == pytest.approx(scale, abs=0.1)
    assert gen.name == f"Normal(mu={loc}, sigma={scale})"

    rng2 = np.random.default_rng(42)
    data2 = gen.generate(10000, rng=rng2)
    assert np.array_equal(data, data2)


# Test method names
@pytest.mark.parametrize(
    "method_class, expected_name",
    [
        (Bonferroni, "Bonferroni Correction"),
        (BonferroniHochberg, "Bonferroni-Hochberg Correction"),
        (BenjaminiHochberg, "Benjamini-Hochberg Correction"),
    ],
)
def test_method_names(method_class, expected_name):
    method = method_class()
    assert method.name == expected_name


# Test Bonferroni correction
@pytest.mark.parametrize(
    "pvals, alpha, expected",
    [
        ([0.01], 0.05, [True]),  # single true
        ([0.2], 0.05, [False]),  # sigle false
        ([0.04, 0.5], 0.05, [False, False]),  # douple false
        ([0.06, 0.07, 0.2, 0.5], 0.05, [False, False, False, False]),  # multiple false
        ([0.0001, 0.001, 0.002, 0.003], 0.05, [True, True, True, True]),  # multiple tue
        (
            [0.0005, 0.02, 0.04, 0.06, 0.2],
            0.05,
            [True, False, False, False, False],
        ),  # mixed
        (
            [0.0125, 0.0125, 0.06, 0.06],
            0.05,
            [True, True, False, False],
        ),  # ties at alpha/m
        (
            [0.2, 0.15, 0.04, 0.02, 0.001],
            0.05,
            [False, False, False, False, True],
        ),  # unsorted
        ([0.05, 0.05, 0.05, 0.05], 0.05, [False, False, False, False]),  # all at alpha
        ([0.01, 0.03, 0.08, 0.12], 0.1, [True, False, False, False]),  # alpha 10%
        ([0.0005, 0.002, 0.004, 0.02], 0.01, [True, True, False, False]),  # alpha 1%
    ],
)
def test_bonferroni_correction(pvals, alpha, expected):
    bonf = Bonferroni()
    pvals_array = np.array(pvals)
    expected_array = np.array(expected)
    result = bonf(pvals_array, alpha)
    assert np.array_equal(result, expected_array)


# test Hochberg (1988)
@pytest.mark.parametrize(
    "pvals, alpha, expected",
    [
        ([0.01], 0.05, [True]),
        ([0.2], 0.05, [False]),
        ([0.04, 0.5], 0.05, [False, False]),
        ([0.06, 0.07, 0.2, 0.5], 0.05, [False, False, False, False]),
        ([0.0001, 0.001, 0.002, 0.003], 0.05, [True, True, True, True]),
        ([0.0005, 0.02, 0.04, 0.06, 0.2], 0.05, [True, False, False, False, False]),
        ([0.025, 0.025, 0.06, 0.06], 0.05, [False, False, False, False]),
        ([0.2, 0.15, 0.04, 0.02, 0.001], 0.05, [False, False, False, False, True]),
        ([0.05, 0.05, 0.05, 0.05], 0.05, [True, True, True, True]),
        ([0.001, 0.02, 0.025, 0.03, 0.2], 0.05, [True, False, False, False, False]),
        ([0.009, 0.5, 0.6, 0.7, 0.8], 0.05, [True, False, False, False, False]),
    ],
)
def test_bonferroni_hochberg_correction(pvals, alpha, expected):
    bonf_hoch = BonferroniHochberg()
    pvals_array = np.array(pvals)
    expected_array = np.array(expected)
    result = bonf_hoch(pvals_array, alpha)
    assert np.array_equal(result, expected_array)


# test FDR (Benjamini and Hochberg, 1995)
@pytest.mark.parametrize(
    "pvals, alpha, expected",
    [
        ([0.01], 0.05, [True]),
        ([0.2], 0.05, [False]),
        ([0.01, 0.02], 0.05, [True, True]),
        ([0.04, 0.5], 0.05, [False, False]),
        ([0.06, 0.07, 0.2, 0.5], 0.05, [False, False, False, False]),
        ([0.0001, 0.001, 0.002, 0.003], 0.05, [True, True, True, True]),
        ([0.0005, 0.02, 0.04, 0.06, 0.2], 0.05, [True, True, False, False, False]),
        ([0.025, 0.025, 0.06, 0.06], 0.05, [True, True, False, False]),
        ([0.2, 0.15, 0.04, 0.02, 0.001], 0.05, [False, False, False, True, True]),
        ([0.05, 0.05, 0.05, 0.05], 0.05, [True, True, True, True]),
        ([0.001, 0.02, 0.025, 0.03, 0.2], 0.05, [True, True, True, True, False]),
        ([0.009, 0.5, 0.6, 0.7, 0.8], 0.05, [True, False, False, False, False]),
    ],
)
def test_benjamini_hochberg_correction(pvals, alpha, expected):
    bh = BenjaminiHochberg()
    pvals_array = np.array(pvals)
    expected_array = np.array(expected)
    result = bh(pvals_array, alpha)
    assert np.array_equal(result, expected_array)


# test generate_means function
@pytest.mark.parametrize(
    "m, m0, scheme, L, expected",
    [
        (4, 0, "E", 4, np.array([3.0, 4.0, 1.0, 2.0])),  # all values present
        (4, 0, "I", 4, np.array([3.0, 3.0, 4.0, 4.0])),
        (4, 0, "D", 4, np.array([3.0, 4.0, 1.0, 2.0])),  # all values present
        (4, 2, "E", 4, np.array([3.0, 4.0, 0.0, 0.0])),
        (4, 2, "D", 4, np.array([3.0, 4.0, 0.0, 0.0])),
        (4, 2, "I", 4, np.array([3.0, 4.0, 0.0, 0.0])),
        (8, 7, "E", 4, np.array([4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),  # one high
        (8, 7, "D", 4, np.array([4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),  # one low
        (8, 7, "I", 4, np.array([4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),  # one high
        (8, 0, "E", 4, np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0])),
        (8, 0, "D", 4, np.array([1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0])),
        (8, 0, "I", 4, np.array([2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0])),
        (16, 8, "E", 4, None),
        (16, 8, "D", 4, None),
        (16, 8, "I", 4, None),
        (16, 0, "E", 4, None),
        (16, 0, "D", 4, None),
        (16, 0, "I", 4, None),
        (16, 16, "E", 4, np.zeros(16)),
        (16, 16, "D", 4, np.zeros(16)),
        (16, 16, "I", 4, np.zeros(16)),
    ],
)
def test_generate_means(m, m0, scheme, L, expected):
    means = np.zeros(m)
    means[:m-m0] = generate_means(m, m0, scheme_dict[scheme], L)
    # Check that the means contain the expected values (ignoring order)
    if expected is not None:
        for val in np.unique(expected):
            assert np.sum(means == val) == np.sum(expected == val)
    else:
        assert np.sum(means != 0) == m - m0


# test compute_p_values function
@pytest.mark.parametrize(
    "normal_samples, expected",
    [
        (np.array([0, 0]), np.array([1.0, 1.0])),
        (np.array([1, 1]), np.array([0.3173, 0.3173])),
        (np.array([1.96, -1.96]), np.array([0.05, 0.05])),
    ],
)
def test_compute_p_values(normal_samples, expected):
    pvals = compute_p_values(normal_samples)
    assert np.allclose(pvals, expected, atol=1e-4)


# test Power metric
@pytest.mark.parametrize(
    "rejections, true_alternatives, expected",
    [
        (
            np.array([True, True, False, False]),
            np.array([True, False, True, False]),
            0.5,
        ),
        (
            np.array([True, True, True, False]),
            np.array([True, True, False, False]),
            1.0,
        ),
        (np.array([True, True, True, True]), np.array([True, True, True, False]), 1.0),
        (
            np.array([False, False, False, False]),
            np.array([True, True, False, False]),
            0.0,
        ),
        (
            np.array([True, False, True, False]),
            np.array([False, False, False, False]),
            0.0,
        ),
    ],
)
def test_power_metric(rejections, true_alternatives, expected):
    result = Power()(rejections, true_alternatives)
    assert np.isclose(result, expected, atol=1e-4)


# test False Discovery Rate metric
@pytest.mark.parametrize(
    "rejections, true_alternatives, expected",
    [
        (
            np.array([True, True, False, False]),
            np.array([True, False, True, False]),
            0.5,
        ),
        (
            np.array([True, True, True, False]),
            np.array([True, True, False, False]),
            1 / 3,
        ),
        (np.array([True, True, True, True]), np.array([True, True, True, False]), 0.25),
        (
            np.array([False, False, False, False]),
            np.array([True, True, False, False]),
            0.0,
        ),
        (
            np.array([True, False, True, False]),
            np.array([False, False, False, False]),
            1.0,
        ),
    ],
)
def test_false_discovery_rate_metric(rejections, true_alternatives, expected):
    result = FalseDiscoveryRate()(rejections, true_alternatives)
    assert np.isclose(result, expected, atol=1e-4)


# test True Rejections metric
@pytest.mark.parametrize(
    "rejections, true_alternatives, expected",
    [
        (
            np.array([True, True, False, False]),
            np.array([True, False, True, False]),
            1,
        ),
        (
            np.array([True, True, True, False]),
            np.array([True, True, False, False]),
            2,
        ),
        (np.array([True, True, True, True]), np.array([True, True, True, False]), 3),
        (
            np.array([False, False, False, False]),
            np.array([True, True, False, False]),
            0,
        ),
        (
            np.array([True, False, True, False]),
            np.array([False, False, False, False]),
            0,
        ),
    ],
)
def test_true_rejections_metric(rejections, true_alternatives, expected):
    result = TrueRejections()(rejections, true_alternatives)
    assert result == expected


# test Rejections Number metric
@pytest.mark.parametrize(
    "rejections, true_alternatives, expected",
    [
        (
            np.array([True, True, False, False]),
            np.array([True, False, True, False]),
            2,
        ),
        (
            np.array([True, True, True, False]),
            np.array([True, True, False, False]),
            3,
        ),
        (np.array([True, True, True, True]), np.array([True, True, True, False]), 4),
        (
            np.array([False, False, False, False]),
            np.array([True, True, False, False]),
            0,
        ),
        (
            np.array([True, False, True, False]),
            np.array([False, False, False, False]),
            2,
        ),
    ],
)
def test_rejections_number_metric(rejections, true_alternatives, expected):
    result = RejectionsNumber()(rejections, true_alternatives)
    assert result == expected
