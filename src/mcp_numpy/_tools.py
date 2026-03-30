import numpy as np
from fastmcp import FastMCP

mcp = FastMCP("mcp-numpy")


def _to_numpy_array(data: list, dtype: str = "float64") -> np.ndarray:
    """Convert a list to a numpy array."""
    return np.array(data, dtype=dtype)


def _from_numpy_array(arr: np.ndarray) -> list:
    """Convert a numpy array to a list."""
    return arr.tolist()


@mcp.tool()
def np_array(data: list, dtype: str = "float64") -> list:
    """Create a NumPy array from a list.

    Args:
        data: A Python list containing the array elements.
        dtype: The data type of the array (default: "float64").
            Common values: "int32", "int64", "float32", "float64", "complex128".

    Returns:
        A list representation of the NumPy array.

    Example:
        >>> np_array([1, 2, 3, 4, 5])
        [1.0, 2.0, 3.0, 4.0, 5.0]

        >>> np_array([[1, 2], [3, 4]], dtype="int32")
        [[1, 2], [3, 4]]
    """
    return _from_numpy_array(_to_numpy_array(data, dtype))


@mcp.tool()
def np_zeros(shape: int | list, dtype: str = "float64") -> list:
    """Create an array of zeros.

    Args:
        shape: An integer for 1D shape, or a list of integers for multi-dimensional.
        dtype: The data type of the array (default: "float64").

    Returns:
        A list representation of the zeros array.

    Example:
        >>> np_zeros(5)
        [0.0, 0.0, 0.0, 0.0, 0.0]

        >>> np_zeros([2, 3])
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    """
    shape = shape if isinstance(shape, (list, tuple)) else shape
    return _from_numpy_array(np.zeros(shape, dtype=dtype))


@mcp.tool()
def np_ones(shape: int | list, dtype: str = "float64") -> list:
    """Create an array of ones.

    Args:
        shape: An integer for 1D shape, or a list of integers for multi-dimensional.
        dtype: The data type of the array (default: "float64").

    Returns:
        A list representation of the ones array.

    Example:
        >>> np_ones(5)
        [1.0, 1.0, 1.0, 1.0, 1.0]

        >>> np_ones([2, 3])
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    """
    shape = shape if isinstance(shape, (list, tuple)) else shape
    return _from_numpy_array(np.ones(shape, dtype=dtype))


@mcp.tool()
def np_full(shape: int | list, fill_value: float, dtype: str = "float64") -> list:
    """Create an array filled with a constant value.

    Args:
        shape: An integer for 1D shape, or a list of integers for multi-dimensional.
        fill_value: The value to fill the array with.
        dtype: The data type of the array (default: "float64").

    Returns:
        A list representation of the filled array.

    Example:
        >>> np_full(5, 7.0)
        [7.0, 7.0, 7.0, 7.0, 7.0]

        >>> np_full([2, 3], 5)
        [[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]]
    """
    shape = shape if isinstance(shape, (list, tuple)) else shape
    return _from_numpy_array(np.full(shape, fill_value, dtype=dtype))


@mcp.tool()
def np_arange(
    start: float, stop: float, step: float = 1, dtype: str = "float64"
) -> list:
    """Create an array with evenly spaced values within a given interval.

    Args:
        start: Start of interval.
        stop: End of interval (exclusive).
        step: Spacing between values (default: 1).
        dtype: The data type of the array (default: "float64").

    Returns:
        A list representation of the array.

    Example:
        >>> np_arange(0, 5)
        [0.0, 1.0, 2.0, 3.0, 4.0]

        >>> np_arange(0, 10, 2)
        [0.0, 2.0, 4.0, 6.0, 8.0]
    """
    return _from_numpy_array(np.arange(start, stop, step, dtype=dtype))


@mcp.tool()
def np_linspace(
    start: float, stop: float, num: int = 50, dtype: str = "float64"
) -> list:
    """Create an array with evenly spaced numbers over a specified interval.

    Args:
        start: Start of interval.
        stop: End of interval.
        num: Number of samples to generate (default: 50).
        dtype: The data type of the array (default: "float64").

    Returns:
        A list representation of the array.

    Example:
        >>> np_linspace(0, 1, 5)
        [0.0, 0.25, 0.5, 0.75, 1.0]
    """
    return _from_numpy_array(np.linspace(start, stop, num, dtype=dtype))


@mcp.tool()
def np_eye(n: int, m: int | None = None, dtype: str = "float64") -> list:
    """Return a 2D identity array.

    Args:
        N: Number of rows.
        M: Number of columns (default: None, equals N).
        dtype: The data type of the array (default: "float64").

    Returns:
        A list representation of the identity matrix.

    Example:
        >>> np_eye(3)
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

        >>> np_eye(2, 3)
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    """
    return _from_numpy_array(np.eye(n, m, dtype=dtype))


@mcp.tool()
def np_diag(k: list | int, dtype: str = "float64") -> list:
    """Create a diagonal array or extract the diagonal of an array.

    Args:
        k: If a list, creates diagonal array from it. If an int, extracts that diagonal.
        dtype: The data type of the array (default: "float64").

    Returns:
        A list representation of the diagonal array.

    Example:
        >>> np_diag([1, 2, 3])
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]

        >>> np_diag(1)
        [[1.0]]
    """
    if isinstance(k, list):
        arr = np.array(k, dtype=dtype)
        return _from_numpy_array(np.diag(arr))
    return _from_numpy_array(np.diag(k))


@mcp.tool()
def np_reshape(array: list, newshape: int | list) -> list:
    """Give a new shape to an array without changing its data.

    Args:
        array: The input array to reshape.
        newshape: The new shape (int or list of ints).

    Returns:
        A list representation of the reshaped array.

    Example:
        >>> np_reshape([1, 2, 3, 4], [2, 2])
        [[1.0, 2.0], [3.0, 4.0]]
    """
    arr = _to_numpy_array(array)
    newshape = newshape if isinstance(newshape, (list, tuple)) else newshape
    return _from_numpy_array(arr.reshape(newshape))


@mcp.tool()
def np_transpose(array: list, axes: list | None = None) -> list:
    """Reverse or permute the axes of an array.

    Args:
        array: The input array.
        axes: By default, reverse the axes. Otherwise, permute the axes.

    Returns:
        A list representation of the transposed array.

    Example:
        >>> np_transpose([[1, 2], [3, 4]])
        [[1.0, 3.0], [2.0, 4.0]]
    """
    arr = _to_numpy_array(array)
    return _from_numpy_array(np.transpose(arr, axes))


@mcp.tool()
def np_concatenate(arrays: list, axis: int = 0) -> list:
    """Join a sequence of arrays along an existing axis.

    Args:
        arrays: A sequence of arrays to concatenate.
        axis: The axis along which to concatenate (default: 0).

    Returns:
        A list representation of the concatenated array.

    Example:
        >>> np_concatenate([[1, 2], [3, 4]], [[5, 6]])
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    """
    arrs = [_to_numpy_array(a) for a in arrays]
    return _from_numpy_array(np.concatenate(arrs, axis=axis))


@mcp.tool()
def np_split(array: list, indices_or_sections: int | list, axis: int = 0) -> list:
    """Split an array into multiple sub-arrays.

    Args:
        array: The array to split.
        indices_or_sections: If an int, the number of equal sections. If a list, the indices at which to split.
        axis: The axis along which to split (default: 0).

    Returns:
        A list of sub-arrays.

    Example:
        >>> np_split([1, 2, 3, 4, 5], 2)
        [[1.0, 2.0], [3.0, 4.0, 5.0]]
    """
    arr = _to_numpy_array(array)
    result = np.split(arr, indices_or_sections, axis=axis)
    return [r.tolist() for r in result]


@mcp.tool()
def np_tile(array: list, reps: int | list) -> list:
    """Construct an array by repeating the input array the given number of times.

    Args:
        array: The input array to tile.
        reps: The number of repetitions along each axis.

    Returns:
        A list representation of the tiled array.

    Example:
        >>> np_tile([1, 2], 3)
        [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
    """
    arr = _to_numpy_array(array)
    reps = reps if isinstance(reps, (list, tuple)) else reps
    return _from_numpy_array(np.tile(arr, reps))


@mcp.tool()
def np_repeat(array: list, repeats: int, axis: int | None = None) -> list:
    """Repeat elements of an array.

    Args:
        array: The input array.
        repeats: The number of repetitions for each element.
        axis: The axis along which to repeat values (default: flattens).

    Returns:
        A list representation of the repeated array.

    Example:
        >>> np_repeat([1, 2], 3)
        [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
    """
    arr = _to_numpy_array(array)
    return _from_numpy_array(np.repeat(arr, repeats, axis=axis))


@mcp.tool()
def np_squeeze(array: list, axis: int | list | None = None) -> list:
    """Remove single-dimensional entries from the shape of an array.

    Args:
        array: The input array.
        axis: Selects a subset of the length-1 dimensions (default: all).

    Returns:
        A list representation of the squeezed array.

    Example:
        >>> np_squeeze([[[1], [2], [3]]])
        [1.0, 2.0, 3.0]
    """
    arr = _to_numpy_array(array)
    if axis is not None:
        axis = axis if isinstance(axis, (list, tuple)) else axis
    return _from_numpy_array(np.squeeze(arr, axis=axis))


@mcp.tool()
def np_flatten(array: list) -> list:
    """Return a flattened copy of the array.

    Args:
        array: The input array.

    Returns:
        A flat list representation.

    Example:
        >>> np_flatten([[1, 2], [3, 4]])
        [1.0, 2.0, 3.0, 4.0]
    """
    arr = _to_numpy_array(array)
    return _from_numpy_array(arr.flatten())


@mcp.tool()
def np_sum(
    array: list, axis: int | list | None = None, dtype: str = "float64"
) -> float | list:
    """Sum of array elements over given axis(es).

    Args:
        array: The input array.
        axis: Axis along which to sum (default: None, sums all).
        dtype: The type of the returned array (default: "float64").

    Returns:
        The sum as a float or list.

    Example:
        >>> np_sum([1, 2, 3, 4])
        10.0

        >>> np_sum([[1, 2], [3, 4]], axis=1)
        [3.0, 7.0]
    """
    arr = _to_numpy_array(array)
    result = np.sum(arr, axis=axis, dtype=dtype)
    if np.isscalar(result):
        return float(result)
    return _from_numpy_array(result)


@mcp.tool()
def np_mean(array: list, axis: int | list | None = None) -> float | list:
    """Compute the arithmetic mean along the specified axis.

    Args:
        array: The input array.
        axis: Axis along which to compute mean (default: None, mean of all).

    Returns:
        The mean as a float or list.

    Example:
        >>> np_mean([1, 2, 3, 4])
        2.5

        >>> np_mean([[1, 2], [3, 4]], axis=0)
        [2.0, 3.0]
    """
    arr = _to_numpy_array(array)
    result = np.mean(arr, axis=axis)
    if np.isscalar(result):
        return float(result)
    return _from_numpy_array(result)


@mcp.tool()
def np_std(array: list, axis: int | list | None = None, ddof: int = 0) -> float | list:
    """Compute the standard deviation along the specified axis.

    Args:
        array: The input array.
        axis: Axis along which to compute std (default: None, std of all).
        ddof: Delta degrees of freedom for normalization (default: 0).

    Returns:
        The std as a float or list.

    Example:
        >>> np_std([1, 2, 3, 4, 5])
        1.4142135623730951
    """
    arr = _to_numpy_array(array)
    result = np.std(arr, axis=axis, ddof=ddof)
    if np.isscalar(result):
        return float(result)
    return _from_numpy_array(result)


@mcp.tool()
def np_var(array: list, axis: int | list | None = None, ddof: int = 0) -> float | list:
    """Compute the variance along the specified axis.

    Args:
        array: The input array.
        axis: Axis along which to compute variance (default: None, variance of all).
        ddof: Delta degrees of freedom for normalization (default: 0).

    Returns:
        The variance as a float or list.

    Example:
        >>> np_var([1, 2, 3, 4, 5])
        2.0
    """
    arr = _to_numpy_array(array)
    result = np.var(arr, axis=axis, ddof=ddof)
    if np.isscalar(result):
        return float(result)
    return _from_numpy_array(result)


@mcp.tool()
def np_min(array: list, axis: int | list | None = None) -> float | list:
    """Return the minimum of an array or minimum along an axis.

    Args:
        array: The input array.
        axis: Axis along which to find minimum (default: None, min of all).

    Returns:
        The minimum as a float or list.

    Example:
        >>> np_min([3, 1, 4, 1, 5])
        1.0
    """
    arr = _to_numpy_array(array)
    result = np.min(arr, axis=axis)
    if np.isscalar(result):
        return float(result)
    return _from_numpy_array(result)


@mcp.tool()
def np_max(array: list, axis: int | list | None = None) -> float | list:
    """Return the maximum of an array or maximum along an axis.

    Args:
        array: The input array.
        axis: Axis along which to find maximum (default: None, max of all).

    Returns:
        The maximum as a float or list.

    Example:
        >>> np_max([3, 1, 4, 1, 5])
        5.0
    """
    arr = _to_numpy_array(array)
    result = np.max(arr, axis=axis)
    if np.isscalar(result):
        return float(result)
    return _from_numpy_array(result)


@mcp.tool()
def np_argmin(array: list, axis: int | None = None) -> int | list:
    """Return the indices of the minimum values along an axis.

    Args:
        array: The input array.
        axis: Axis along which to find argmin (default: None, flattened).

    Returns:
        The index of the minimum as an int or list.

    Example:
        >>> np_argmin([3, 1, 4, 1, 5])
        1
    """
    arr = _to_numpy_array(array)
    result = np.argmin(arr, axis=axis)
    if np.isscalar(result):
        return int(result)
    return _from_numpy_array(result)


@mcp.tool()
def np_argmax(array: list, axis: int | None = None) -> int | list:
    """Return the indices of the maximum values along an axis.

    Args:
        array: The input array.
        axis: Axis along which to find argmax (default: None, flattened).

    Returns:
        The index of the maximum as an int or list.

    Example:
        >>> np_argmax([3, 1, 4, 1, 5])
        4
    """
    arr = _to_numpy_array(array)
    result = np.argmax(arr, axis=axis)
    if np.isscalar(result):
        return int(result)
    return _from_numpy_array(result)


@mcp.tool()
def np_dot(a: list, b: list) -> list | float:
    """Compute the dot product of two arrays.

    Args:
        a: First input array.
        b: Second input array.

    Returns:
        The dot product as a float or list.

    Example:
        >>> np_dot([1, 2], [3, 4])
        11.0
    """
    arr_a = _to_numpy_array(a)
    arr_b = _to_numpy_array(b)
    result = np.dot(arr_a, arr_b)
    if np.isscalar(result):
        return float(result)
    return _from_numpy_array(result)


@mcp.tool()
def np_matmul(a: list, b: list) -> list:
    """Matrix product of two arrays.

    Args:
        a: First input array (2D).
        b: Second input array (2D).

    Returns:
        The matrix product.

    Example:
        >>> np_matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]])
        [[19.0, 22.0], [43.0, 50.0]]
    """
    arr_a = _to_numpy_array(a)
    arr_b = _to_numpy_array(b)
    return _from_numpy_array(np.matmul(arr_a, arr_b))


@mcp.tool()
def np_cross(a: list, b: list) -> list:
    """Compute the cross product of two arrays.

    Args:
        a: First input array.
        b: Second input array.

    Returns:
        The cross product.

    Example:
        >>> np_cross([1, 2, 3], [4, 5, 6])
        [-3.0, 6.0, -3.0]
    """
    arr_a = _to_numpy_array(a)
    arr_b = _to_numpy_array(b)
    return _from_numpy_array(np.cross(arr_a, arr_b))


@mcp.tool()
def np_trace(array: list, offset: int = 0) -> float:
    """Return the sum along the main diagonal of the array.

    Args:
        array: The input array (must be at least 2D).
        offset: The diagonal offset (default: 0, main diagonal).

    Returns:
        The trace as a float.

    Example:
        >>> np_trace([[1, 2], [3, 4]])
        5.0
    """
    arr = _to_numpy_array(array)
    return float(np.trace(arr, offset=offset))


@mcp.tool()
def np_cumsum(array: list, axis: int | None = None) -> list:
    """Return the cumulative sum of the array along a given axis.

    Args:
        array: The input array.
        axis: The axis along which to compute cumsum (default: None, flattened).

    Returns:
        The cumulative sum.

    Example:
        >>> np_cumsum([1, 2, 3])
        [1.0, 3.0, 6.0]
    """
    arr = _to_numpy_array(array)
    return _from_numpy_array(np.cumsum(arr, axis=axis))


@mcp.tool()
def np_cumprod(array: list, axis: int | None = None) -> list:
    """Return the cumulative product of the array along a given axis.

    Args:
        array: The input array.
        axis: The axis along which to compute cumprod (default: None, flattened).

    Returns:
        The cumulative product.

    Example:
        >>> np_cumprod([1, 2, 3])
        [1.0, 2.0, 6.0]
    """
    arr = _to_numpy_array(array)
    return _from_numpy_array(np.cumprod(arr, axis=axis))


@mcp.tool()
def np_diff(array: list, n: int = 1, axis: int = 0) -> list:
    """Calculate the n-th discrete difference along the given axis.

    Args:
        array: The input array.
        n: The number of times values are differenced (default: 1).
        axis: The axis along which to difference (default: 0).

    Returns:
        The n-th difference.

    Example:
        >>> np_diff([1, 4, 9, 16])
        [3.0, 5.0, 7.0]
    """
    arr = _to_numpy_array(array)
    return _from_numpy_array(np.diff(arr, n=n, axis=axis))


@mcp.tool()
def np_inv(array: list) -> list:
    """Compute the (multiplicative) inverse of a matrix.

    Args:
        array: The input matrix (must be square and invertible).

    Returns:
        The inverse matrix.

    Example:
        >>> np_inv([[1, 2], [3, 4]])
        [[-2.0, 1.0], [1.5, -0.5]]
    """
    arr = _to_numpy_array(array)
    return _from_numpy_array(np.linalg.inv(arr))


@mcp.tool()
def np_det(array: list) -> float:
    """Compute the determinant of an array.

    Args:
        array: The input matrix (must be square).

    Returns:
        The determinant as a float.

    Example:
        >>> np_det([[1, 2], [3, 4]])
        -2.0
    """
    arr = _to_numpy_array(array)
    return float(np.linalg.det(arr))


@mcp.tool()
def np_eig(array: list) -> dict:
    """Compute the eigenvalues and eigenvectors of a square array.

    Args:
        array: The input square matrix.

    Returns:
        A dict with 'eigenvalues' and 'eigenvectors'.

    Example:
        >>> np_eig([[1, 0], [0, 1]])
        {'eigenvalues': [1.0, 1.0], 'eigenvectors': [[1.0, 0.0], [0.0, 1.0]]}
    """
    arr = _to_numpy_array(array)
    eigenvalues, eigenvectors = np.linalg.eig(arr)
    return {
        "eigenvalues": _from_numpy_array(eigenvalues),
        "eigenvectors": _from_numpy_array(eigenvectors),
    }


@mcp.tool()
def np_svd(array: list, full_matrices: bool = False) -> dict:
    """Singular Value Decomposition.

    Args:
        array: The input array.
        full_matrices: Whether to compute full SVD (default: False).

    Returns:
        A dict with 'U', 'singular_values', and 'Vh'.

    Example:
        >>> result = np_svd([[1, 2], [3, 4]])
        >>> result['singular_values']
        [5.464985704219029, 0.36596618969622733]
    """
    arr = _to_numpy_array(array)
    U, s, Vh = np.linalg.svd(arr, full_matrices=full_matrices)  # noqa: N806
    return {
        "U": _from_numpy_array(U),
        "singular_values": _from_numpy_array(s),
        "Vh": _from_numpy_array(Vh),
    }


@mcp.tool()
def np_solve(a: list, b: list) -> list:
    """Solve a linear matrix equation, or system of linear equations.

    Args:
        a: Coefficient matrix.
        b: Ordinate or "dependent variable" values.

    Returns:
        Solution to the system.

    Example:
        >>> np_solve([[1, 1], [1, 2]], [3, 5])
        [1.0, 2.0]
    """
    arr_a = _to_numpy_array(a)
    arr_b = _to_numpy_array(b)
    return _from_numpy_array(np.linalg.solve(arr_a, arr_b))


@mcp.tool()
def np_linalg_norm(array: list, ord: str = "fro") -> float:
    """Matrix or vector norm.

    Args:
        array: The input array.
        ord: The order of the norm (default: "fro" for matrices, "2" for vectors).
            Common values: "fro", "nuc", "inf", "-inf", "0", "1", "2".

    Returns:
        The norm as a float.

    Example:
        >>> np_linalg_norm([3, 4])
        5.0
    """
    arr = _to_numpy_array(array)
    return float(np.linalg.norm(arr, ord=ord))


@mcp.tool()
def np_rand(shape: int | list) -> list:
    """Random values in a given shape.

    Args:
        shape: The shape of the output (int or list of ints).

    Returns:
        Array of random values.

    Example:
        >>> len(np_rand(5))
        5

        >>> len(np_rand([2, 3]))
        2
    """
    shape = shape if isinstance(shape, (list, tuple)) else shape
    return _from_numpy_array(
        np.random.rand(*(shape if isinstance(shape, list) else [shape]))
    )


@mcp.tool()
def np_randn(shape: int | list) -> list:
    """Return a sample (or samples) from the "standard normal" distribution.

    Args:
        shape: The shape of the output (int or list of ints).

    Returns:
        Array of random normal values.

    Example:
        >>> len(np_randn(5))
        5
    """
    if isinstance(shape, int):
        return _from_numpy_array(np.random.randn(shape))
    return _from_numpy_array(np.random.randn(*shape))


@mcp.tool()
def np_randint(
    low: int, high: int | None = None, size: int | list | None = None
) -> list:
    """Return random integers from low (inclusive) to high (exclusive).

    Args:
        low: Lowest integers to be drawn (inclusive). If high is None, this is the upper bound.
        high: Upper bound (exclusive). If None, low=0 and this becomes high.
        size: Output shape (int or tuple of ints).

    Returns:
        Array of random integers.

    Example:
        >>> np_randint(0, 10, 5)
        [3, 5, 7, 2, 9]
    """
    if high is None:
        high = low
        low = 0
    if size is None:
        size = 1
    size = size if isinstance(size, (list, tuple)) else size
    result = np.random.randint(low, high, size=size)
    return _from_numpy_array(result)


@mcp.tool()
def np_random_choice(a: list, size: int | None = None, replace: bool = True) -> list:
    """Generates a random sample from a given array.

    Args:
        a: 1-D array-like object from which to sample.
        size: Output shape (default: None, returns single value).
        replace: Whether sampling with replacement (default: True).

    Returns:
        Random sample(s).

    Example:
        >>> np_random_choice([1, 2, 3, 4, 5], size=3)
        [2, 5, 1]
    """
    result = np.random.choice(a, size=size, replace=replace)
    if size is None:
        return [float(result)]
    return _from_numpy_array(result)


@mcp.tool()
def np_shuffle(array: list) -> list:
    """Modify a sequence in-place by shuffling its contents.

    Args:
        array: The array to shuffle.

    Returns:
        The shuffled array.

    Example:
        >>> np_shuffle([1, 2, 3, 4, 5])
        [3, 1, 5, 2, 4]
    """
    arr = _to_numpy_array(array).copy()
    np.random.shuffle(arr)
    return _from_numpy_array(arr)


@mcp.tool()
def np_percentile(array: list, q: float | list) -> float | list:
    """Compute the q-th percentile of the array elements.

    Args:
        array: The input array.
        q: Percentile(s) to compute (0-100). Can be a float or list.

    Returns:
        The percentile value(s).

    Example:
        >>> np_percentile([1, 2, 3, 4, 5], 50)
        3.0

        >>> np_percentile([1, 2, 3, 4, 5], [25, 50, 75])
        [1.5, 3.0, 4.5]
    """
    arr = _to_numpy_array(array)
    result = np.percentile(arr, q)
    if np.isscalar(result):
        return float(result)
    return _from_numpy_array(result)


@mcp.tool()
def np_quantile(array: list, q: float | list) -> float | list:
    """Compute the q-th quantile of the array elements.

    Args:
        array: The input array.
        q: Quantile(s) to compute (0-1). Can be a float or list.

    Returns:
        The quantile value(s).

    Example:
        >>> np_quantile([1, 2, 3, 4, 5], 0.5)
        3.0
    """
    arr = _to_numpy_array(array)
    result = np.quantile(arr, q)
    if np.isscalar(result):
        return float(result)
    return _from_numpy_array(result)


@mcp.tool()
def np_histogram(array: list, bins: int | list = 10, range: list | None = None) -> dict:
    """Compute the histogram of a set of data.

    Args:
        array: Input data.
        bins: Number of bins or bin edges (default: 10).
        range: The lower and upper range of the bins (default: [min, max]).

    Returns:
        A dict with 'histogram' (counts) and 'bin_edges'.

    Example:
        >>> result = np_histogram([1, 1, 2, 2, 3, 3, 4, 4])
        >>> result['histogram']
        [2, 2, 2, 2]
    """
    arr = _to_numpy_array(array)
    result = np.histogram(arr, bins=bins, range=range)
    return {
        "histogram": _from_numpy_array(result[0]),
        "bin_edges": _from_numpy_array(result[1]),
    }


@mcp.tool()
def np_correlate(a: list, b: list, mode: str = "full") -> list:
    """Cross-correlation of two 1-dimensional sequences.

    Args:
        a: First input sequence.
        b: Second input sequence.
        mode: Computation mode (default: "full"). Options: "full", "same", "valid".

    Returns:
        The cross-correlation array.

    Example:
        >>> np_correlate([1, 2, 3], [0, 1, 0.5])
        [0.5, 2.0, 4.0, 3.0, 0.0]
    """
    arr_a = _to_numpy_array(a)
    arr_b = _to_numpy_array(b)
    return _from_numpy_array(np.correlate(arr_a, arr_b, mode=mode))


@mcp.tool()
def np_corrcoef(array: list, rowvar: bool = True) -> list:
    """Return Pearson product-moment correlation coefficients.

    Args:
        array: A 1-D or 2-D array containing multiple variables and observations.
        rowvar: If True, each row represents a variable (default: True).

    Returns:
        The correlation coefficient matrix.

    Example:
        >>> np_corrcoef([[1, 2, 3], [1, 1, 1]])
        [[1.0, 0.0], [0.0, nan]]
    """
    arr = _to_numpy_array(array)
    return _from_numpy_array(np.corrcoef(arr, rowvar=rowvar))


@mcp.tool()
def np_add(a: list, b: list) -> list:
    """Element-wise addition of two arrays.

    Args:
        a: First input array.
        b: Second input array.

    Returns:
        The element-wise sum.

    Example:
        >>> np_add([1, 2, 3], [4, 5, 6])
        [5.0, 7.0, 9.0]
    """
    arr_a = _to_numpy_array(a)
    arr_b = _to_numpy_array(b)
    return _from_numpy_array(np.add(arr_a, arr_b))


@mcp.tool()
def np_subtract(a: list, b: list) -> list:
    """Element-wise subtraction of two arrays.

    Args:
        a: First input array.
        b: Second input array.

    Returns:
        The element-wise difference.

    Example:
        >>> np_subtract([4, 5, 6], [1, 2, 3])
        [3.0, 3.0, 3.0]
    """
    arr_a = _to_numpy_array(a)
    arr_b = _to_numpy_array(b)
    return _from_numpy_array(np.subtract(arr_a, arr_b))


@mcp.tool()
def np_multiply(a: list, b: list) -> list:
    """Element-wise multiplication of two arrays.

    Args:
        a: First input array.
        b: Second input array.

    Returns:
        The element-wise product.

    Example:
        >>> np_multiply([1, 2, 3], [4, 5, 6])
        [4.0, 10.0, 18.0]
    """
    arr_a = _to_numpy_array(a)
    arr_b = _to_numpy_array(b)
    return _from_numpy_array(np.multiply(arr_a, arr_b))


@mcp.tool()
def np_divide(a: list, b: list) -> list:
    """Element-wise division of two arrays.

    Args:
        a: First input array (dividend).
        b: Second input array (divisor).

    Returns:
        The element-wise quotient.

    Example:
        >>> np_divide([6, 12, 18], [2, 3, 6])
        [3.0, 4.0, 3.0]
    """
    arr_a = _to_numpy_array(a)
    arr_b = _to_numpy_array(b)
    return _from_numpy_array(np.divide(arr_a, arr_b))


@mcp.tool()
def np_power(a: list, b: list | float) -> list:
    """Element-wise exponentiation of array elements.

    Args:
        a: The base array.
        b: The exponent (can be array or scalar).

    Returns:
        The element-wise result.

    Example:
        >>> np_power([1, 2, 3], 2)
        [1.0, 4.0, 9.0]
    """
    arr_a = _to_numpy_array(a)
    arr_b = _to_numpy_array(b) if isinstance(b, list) else b
    return _from_numpy_array(np.power(arr_a, arr_b))


@mcp.tool()
def np_mod(a: list, b: list) -> list:
    """Element-wise modulo of two arrays.

    Args:
        a: First input array (dividend).
        b: Second input array (divisor).

    Returns:
        The element-wise remainder.

    Example:
        >>> np_mod([7, 8, 9], [3, 4, 5])
        [1.0, 0.0, 4.0]
    """
    arr_a = _to_numpy_array(a)
    arr_b = _to_numpy_array(b)
    return _from_numpy_array(np.mod(arr_a, arr_b))


@mcp.tool()
def np_sqrt(array: list) -> list:
    """Return the non-negative square root of an array element-wise.

    Args:
        array: The input array.

    Returns:
        The square root of each element.

    Example:
        >>> np_sqrt([1, 4, 9, 16])
        [1.0, 2.0, 3.0, 4.0]
    """
    arr = _to_numpy_array(array)
    return _from_numpy_array(np.sqrt(arr))


@mcp.tool()
def np_abs(array: list) -> list:
    """Calculate the absolute value of array elements.

    Args:
        array: The input array.

    Returns:
        The absolute values.

    Example:
        >>> np_abs([-1, -2, 3])
        [1.0, 2.0, 3.0]
    """
    arr = _to_numpy_array(array)
    return _from_numpy_array(np.abs(arr))


@mcp.tool()
def np_exp(array: list) -> list:
    """Calculate the exponential of all elements in the array.

    Args:
        array: The input array.

    Returns:
        The exponential of each element.

    Example:
        >>> np_exp([0, 1, 2])
        [1.0, 2.718281828459045, 7.38905609893065]
    """
    arr = _to_numpy_array(array)
    return _from_numpy_array(np.exp(arr))


@mcp.tool()
def np_log(array: list) -> list:
    """Natural logarithm, element-wise.

    Args:
        array: The input array.

    Returns:
        The natural logarithm of each element.

    Example:
        >>> np_log([1, np.e, np.e**2])
        [0.0, 1.0, 2.0]
    """
    arr = _to_numpy_array(array)
    return _from_numpy_array(np.log(arr))


@mcp.tool()
def np_log10(array: list) -> list:
    """Base-10 logarithm, element-wise.

    Args:
        array: The input array.

    Returns:
        The base-10 logarithm of each element.

    Example:
        >>> np_log10([1, 10, 100])
        [0.0, 1.0, 2.0]
    """
    arr = _to_numpy_array(array)
    return _from_numpy_array(np.log10(arr))


@mcp.tool()
def np_sin(array: list) -> list:
    """Trigonometric sine, element-wise.

    Args:
        array: The input array (in radians).

    Returns:
        The sine of each element.

    Example:
        >>> np_sin([0, np.pi/2, np.pi])
        [0.0, 1.0, 1.2246467991473532e-16]
    """
    arr = _to_numpy_array(array)
    return _from_numpy_array(np.sin(arr))


@mcp.tool()
def np_cos(array: list) -> list:
    """Trigonometric cosine, element-wise.

    Args:
        array: The input array (in radians).

    Returns:
        The cosine of each element.

    Example:
        >>> np_cos([0, np.pi/2, np.pi])
        [1.0, 6.123233995736766e-17, -1.0]
    """
    arr = _to_numpy_array(array)
    return _from_numpy_array(np.cos(arr))


@mcp.tool()
def np_tan(array: list) -> list:
    """Trigonometric tangent, element-wise.

    Args:
        array: The input array (in radians).

    Returns:
        The tangent of each element.

    Example:
        >>> np_tan([0, np.pi/4, np.pi])
        [0.0, 0.9999999999999999, -1.2246467991473532e-16]
    """
    arr = _to_numpy_array(array)
    return _from_numpy_array(np.tan(arr))


@mcp.tool()
def np_arcsin(array: list) -> list:
    """Inverse sine, element-wise.

    Args:
        array: The input array (must be in [-1, 1]).

    Returns:
        The inverse sine of each element (in radians).

    Example:
        >>> np_arcsin([0, 0.5, 1])
        [0.0, 0.5235987755982988, 1.5707963267948966]
    """
    arr = _to_numpy_array(array)
    return _from_numpy_array(np.arcsin(arr))


@mcp.tool()
def np_arccos(array: list) -> list:
    """Inverse cosine, element-wise.

    Args:
        array: The input array (must be in [-1, 1]).

    Returns:
        The inverse cosine of each element (in radians).

    Example:
        >>> np_arccos([1, 0.5, 0])
        [0.0, 1.0471975511965976, 1.5707963267948966]
    """
    arr = _to_numpy_array(array)
    return _from_numpy_array(np.arccos(arr))


@mcp.tool()
def np_arctan(array: list) -> list:
    """Inverse tangent, element-wise.

    Args:
        array: The input array.

    Returns:
        The inverse tangent of each element (in radians).

    Example:
        >>> np_arctan([0, 1, np.inf])
        [0.0, 0.7853981633974483, 1.5707963267948966]
    """
    arr = _to_numpy_array(array)
    return _from_numpy_array(np.arctan(arr))


@mcp.tool()
def np_sinh(array: list) -> list:
    """Hyperbolic sine, element-wise.

    Args:
        array: The input array.

    Returns:
        The hyperbolic sine of each element.

    Example:
        >>> np_sinh([0, 1])
        [0.0, 1.1752011936438014]
    """
    arr = _to_numpy_array(array)
    return _from_numpy_array(np.sinh(arr))


@mcp.tool()
def np_cosh(array: list) -> list:
    """Hyperbolic cosine, element-wise.

    Args:
        array: The input array.

    Returns:
        The hyperbolic cosine of each element.

    Example:
        >>> np_cosh([0, 1])
        [1.0, 1.5430806348152437]
    """
    arr = _to_numpy_array(array)
    return _from_numpy_array(np.cosh(arr))


@mcp.tool()
def np_tanh(array: list) -> list:
    """Hyperbolic tangent, element-wise.

    Args:
        array: The input array.

    Returns:
        The hyperbolic tangent of each element.

    Example:
        >>> np_tanh([0, 1])
        [0.0, 0.7615941559557649]
    """
    arr = _to_numpy_array(array)
    return _from_numpy_array(np.tanh(arr))


@mcp.tool()
def np_shape(array: list) -> list:
    """Return the shape of an array.

    Args:
        array: The input array.

    Returns:
        The shape as a list of integers.

    Example:
        >>> np_shape([[1, 2, 3], [4, 5, 6]])
        [2, 3]
    """
    arr = _to_numpy_array(array)
    return list(arr.shape)


@mcp.tool()
def np_ndim(array: list) -> int:
    """Return the number of dimensions of an array.

    Args:
        array: The input array.

    Returns:
        The number of dimensions.

    Example:
        >>> np_ndim([[1, 2, 3], [4, 5, 6]])
        2
    """
    arr = _to_numpy_array(array)
    return int(arr.ndim)


@mcp.tool()
def np_size(array: list) -> int:
    """Return the total number of elements in an array.

    Args:
        array: The input array.

    Returns:
        The total number of elements.

    Example:
        >>> np_size([[1, 2, 3], [4, 5, 6]])
        6
    """
    arr = _to_numpy_array(array)
    return int(arr.size)


@mcp.tool()
def np_dtype(array: list) -> str:
    """Return the dtype of an array.

    Args:
        array: The input array.

    Returns:
        The dtype as a string.

    Example:
        >>> np_dtype([1, 2, 3])
        'float64'

        >>> np_dtype([1, 2, 3], dtype="int32")
        'int32'
    """
    arr = _to_numpy_array(array)
    return str(arr.dtype)


@mcp.tool()
def npastype(array: list, dtype: str) -> list:
    """Copy of the array, cast to a specified type.

    Args:
        array: The input array.
        dtype: The target dtype.

    Returns:
        The array with the specified dtype.

    Example:
        >>> npastype([1.5, 2.7, 3.9], "int32")
        [1, 2, 3]
    """
    arr = _to_numpy_array(array)
    return _from_numpy_array(arr.astype(dtype))
