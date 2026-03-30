import pytest

from mcp_numpy._tools import (
    np_abs,
    np_add,
    np_arange,
    np_arccos,
    np_arcsin,
    np_arctan,
    np_argmax,
    np_argmin,
    np_array,
    np_concatenate,
    np_corrcoef,
    np_correlate,
    np_cos,
    np_cosh,
    np_cross,
    np_cumprod,
    np_cumsum,
    np_det,
    np_diag,
    np_diff,
    np_divide,
    np_dot,
    np_dtype,
    np_eig,
    np_exp,
    np_eye,
    np_flatten,
    np_full,
    np_histogram,
    np_inv,
    np_linalg_norm,
    np_linspace,
    np_log,
    np_log10,
    np_matmul,
    np_max,
    np_mean,
    np_min,
    np_mod,
    np_multiply,
    np_ndim,
    np_ones,
    np_percentile,
    np_power,
    np_quantile,
    np_rand,
    np_randint,
    np_randn,
    np_random_choice,
    np_repeat,
    np_reshape,
    np_shape,
    np_shuffle,
    np_sin,
    np_sinh,
    np_size,
    np_solve,
    np_split,
    np_sqrt,
    np_squeeze,
    np_std,
    np_subtract,
    np_sum,
    np_svd,
    np_tan,
    np_tanh,
    np_tile,
    np_trace,
    np_transpose,
    np_var,
    np_zeros,
    npastype,
)


class TestArrayCreation:
    def test_array_basic(self):
        result = np_array([1, 2, 3, 4, 5])
        assert result == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_array_2d(self):
        result = np_array([[1, 2], [3, 4]], dtype="int32")
        assert result == [[1, 2], [3, 4]]

    def test_zeros_1d(self):
        result = np_zeros(5)
        assert result == [0.0, 0.0, 0.0, 0.0, 0.0]

    def test_zeros_2d(self):
        result = np_zeros([2, 3])
        assert result == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    def test_ones_1d(self):
        result = np_ones(5)
        assert result == [1.0, 1.0, 1.0, 1.0, 1.0]

    def test_ones_2d(self):
        result = np_ones([2, 3])
        assert result == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]

    def test_full(self):
        result = np_full(5, 7.0)
        assert result == [7.0, 7.0, 7.0, 7.0, 7.0]

    def test_arange(self):
        result = np_arange(0, 5)
        assert result == [0.0, 1.0, 2.0, 3.0, 4.0]

    def test_arange_with_step(self):
        result = np_arange(0, 10, 2)
        assert result == [0.0, 2.0, 4.0, 6.0, 8.0]

    def test_linspace(self):
        result = np_linspace(0, 1, 5)
        assert result == [0.0, 0.25, 0.5, 0.75, 1.0]

    def test_eye(self):
        result = np_eye(3)
        assert result == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

    def test_diag_from_list(self):
        result = np_diag([1, 2, 3])
        assert result == [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]


class TestArrayManipulation:
    def test_reshape(self):
        result = np_reshape([1, 2, 3, 4], [2, 2])
        assert result == [[1.0, 2.0], [3.0, 4.0]]

    def test_transpose(self):
        result = np_transpose([[1, 2], [3, 4]])
        assert result == [[1.0, 3.0], [2.0, 4.0]]

    def test_concatenate(self):
        result = np_concatenate([[[1, 2], [3, 4]], [[5, 6]]])
        assert result == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

    def test_split(self):
        result = np_split([1, 2, 3, 4, 5, 6], 3)
        assert result == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

    def test_tile(self):
        result = np_tile([1, 2], 3)
        assert result == [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]

    def test_repeat(self):
        result = np_repeat([1, 2], 3)
        assert result == [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]

    def test_squeeze(self):
        result = np_squeeze([[[1], [2], [3]]])
        assert result == [1.0, 2.0, 3.0]

    def test_flatten(self):
        result = np_flatten([[1, 2], [3, 4]])
        assert result == [1.0, 2.0, 3.0, 4.0]


class TestMathOperations:
    def test_sum(self):
        result = np_sum([1, 2, 3, 4])
        assert result == 10.0

    def test_sum_axis(self):
        result = np_sum([[1, 2], [3, 4]], axis=1)
        assert result == [3.0, 7.0]

    def test_mean(self):
        result = np_mean([1, 2, 3, 4, 5])
        assert result == 3.0

    def test_std(self):
        result = np_std([1, 2, 3, 4, 5])
        assert abs(result - 1.4142135623730951) < 1e-10

    def test_var(self):
        result = np_var([1, 2, 3, 4, 5])
        assert result == 2.0

    def test_min(self):
        result = np_min([3, 1, 4, 1, 5])
        assert result == 1.0

    def test_max(self):
        result = np_max([3, 1, 4, 1, 5])
        assert result == 5.0

    def test_argmin(self):
        result = np_argmin([3, 1, 4, 1, 5])
        assert result == 1

    def test_argmax(self):
        result = np_argmax([3, 1, 4, 1, 5])
        assert result == 4

    def test_dot(self):
        result = np_dot([1, 2], [3, 4])
        assert result == 11.0

    def test_matmul(self):
        result = np_matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]])
        assert result == [[19.0, 22.0], [43.0, 50.0]]

    def test_cross(self):
        result = np_cross([1, 2, 3], [4, 5, 6])
        assert result == [-3.0, 6.0, -3.0]

    def test_trace(self):
        result = np_trace([[1, 2], [3, 4]])
        assert result == 5.0

    def test_cumsum(self):
        result = np_cumsum([1, 2, 3])
        assert result == [1.0, 3.0, 6.0]

    def test_cumprod(self):
        result = np_cumprod([1, 2, 3])
        assert result == [1.0, 2.0, 6.0]

    def test_diff(self):
        result = np_diff([1, 4, 9, 16])
        assert result == [3.0, 5.0, 7.0]


class TestLinearAlgebra:
    def test_inv(self):
        result = np_inv([[1, 2], [3, 4]])
        assert result[0][0] == pytest.approx(-2.0)
        assert result[0][1] == pytest.approx(1.0)
        assert result[1][0] == pytest.approx(1.5)
        assert result[1][1] == pytest.approx(-0.5)

    def test_det(self):
        result = np_det([[1, 2], [3, 4]])
        assert result == pytest.approx(-2.0)

    def test_eig(self):
        result = np_eig([[1, 0], [0, 1]])
        assert result["eigenvalues"] == [1.0, 1.0]

    def test_svd(self):
        result = np_svd([[1, 2], [3, 4]])
        assert len(result["U"]) == 2
        assert len(result["singular_values"]) == 2
        assert len(result["Vh"]) == 2

    def test_solve(self):
        result = np_solve([[1, 1], [1, 2]], [3, 5])
        assert result == [1.0, 2.0]

    def test_linalg_norm(self):
        result = np_linalg_norm([3, 4], ord=2)
        assert result == 5.0


class TestRandom:
    def test_rand(self):
        result = np_rand(5)
        assert len(result) == 5
        assert all(0 <= x <= 1 for x in result)

    def test_randn(self):
        result = np_randn(5)
        assert len(result) == 5

    def test_randint(self):
        result = np_randint(0, 10, 5)
        assert len(result) == 5
        assert all(0 <= x < 10 for x in result)

    def test_random_choice(self):
        result = np_random_choice([1, 2, 3, 4, 5], size=3)
        assert len(result) == 3

    def test_shuffle(self):
        original = [1, 2, 3, 4, 5]
        result = np_shuffle(original)
        assert len(result) == 5
        assert set(result) == set(original)


class TestStatistics:
    def test_percentile(self):
        result = np_percentile([1, 2, 3, 4, 5], 50)
        assert result == 3.0

    def test_percentile_list(self):
        result = np_percentile([1, 2, 3, 4, 5], [25, 50, 75])
        assert result[0] == pytest.approx(2.0)
        assert result[1] == pytest.approx(3.0)
        assert result[2] == pytest.approx(4.0)

    def test_quantile(self):
        result = np_quantile([1, 2, 3, 4, 5], 0.5)
        assert result == 3.0

    def test_histogram(self):
        result = np_histogram([1, 1, 2, 2, 3, 3, 4, 4])
        assert len(result["histogram"]) == 10
        assert len(result["bin_edges"]) == 11

    def test_correlate(self):
        result = np_correlate([1, 2, 3], [0, 1, 0.5])
        assert len(result) == 5

    def test_corrcoef(self):
        result = np_corrcoef([[1, 2, 3], [1, 1, 1]])
        assert len(result) == 2


class TestElementWise:
    def test_add(self):
        result = np_add([1, 2, 3], [4, 5, 6])
        assert result == [5.0, 7.0, 9.0]

    def test_subtract(self):
        result = np_subtract([4, 5, 6], [1, 2, 3])
        assert result == [3.0, 3.0, 3.0]

    def test_multiply(self):
        result = np_multiply([1, 2, 3], [4, 5, 6])
        assert result == [4.0, 10.0, 18.0]

    def test_divide(self):
        result = np_divide([6, 12, 18], [2, 3, 6])
        assert result == [3.0, 4.0, 3.0]

    def test_power(self):
        result = np_power([1, 2, 3], 2)
        assert result == [1.0, 4.0, 9.0]

    def test_mod(self):
        result = np_mod([7, 8, 9], [3, 4, 5])
        assert result == [1.0, 0.0, 4.0]

    def test_sqrt(self):
        result = np_sqrt([1, 4, 9, 16])
        assert result == [1.0, 2.0, 3.0, 4.0]

    def test_abs(self):
        result = np_abs([-1, -2, 3])
        assert result == [1.0, 2.0, 3.0]

    def test_exp(self):
        result = np_exp([0, 1])
        assert abs(result[0] - 1.0) < 1e-10
        assert abs(result[1] - 2.718281828459045) < 1e-10

    def test_log(self):
        result = np_log([1, 2.718281828459045])
        assert abs(result[0] - 0.0) < 1e-10
        assert abs(result[1] - 1.0) < 1e-10

    def test_log10(self):
        result = np_log10([1, 10, 100])
        assert result == [0.0, 1.0, 2.0]

    def test_sin(self):
        result = np_sin([0, 1])
        assert abs(result[0] - 0.0) < 1e-10

    def test_cos(self):
        result = np_cos([0, 1])
        assert abs(result[0] - 1.0) < 1e-10

    def test_tan(self):
        result = np_tan([0, 1])
        assert abs(result[0] - 0.0) < 1e-10

    def test_arcsin(self):
        result = np_arcsin([0, 0.5, 1])
        assert abs(result[0] - 0.0) < 1e-10
        assert abs(result[2] - 1.5707963267948966) < 1e-10

    def test_arccos(self):
        result = np_arccos([1, 0.5, 0])
        assert result[0] == 0.0

    def test_arctan(self):
        result = np_arctan([0, 1])
        assert result[0] == 0.0

    def test_sinh(self):
        result = np_sinh([0, 1])
        assert result[0] == 0.0

    def test_cosh(self):
        result = np_cosh([0, 1])
        assert result[0] == 1.0

    def test_tanh(self):
        result = np_tanh([0, 1])
        assert result[0] == 0.0


class TestArrayProperties:
    def test_shape(self):
        result = np_shape([[1, 2, 3], [4, 5, 6]])
        assert result == [2, 3]

    def test_ndim(self):
        result = np_ndim([[1, 2, 3], [4, 5, 6]])
        assert result == 2

    def test_size(self):
        result = np_size([[1, 2, 3], [4, 5, 6]])
        assert result == 6

    def test_dtype(self):
        result = np_dtype([1, 2, 3])
        assert result == "float64"

    def testastype(self):
        result = npastype([1.5, 2.7, 3.9], "int32")
        assert result == [1, 2, 3]
