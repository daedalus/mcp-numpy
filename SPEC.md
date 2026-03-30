# SPEC.md — mcp-numpy

## Purpose

An MCP (Model Context Protocol) server that exposes NumPy functionality as tools, enabling AI assistants to perform numerical computations, array manipulations, and mathematical operations.

## Scope

### What IS in scope
- Core array creation functions (array, zeros, ones, arange, linspace, etc.)
- Array manipulation functions (reshape, transpose, concatenate, split, etc.)
- Mathematical operations (sum, mean, std, min, max, dot, matmul, etc.)
- Linear algebra functions (inv, det, eig, svd, etc.)
- Random number generation (rand, randn, randint, choice, etc.)
- Statistical functions (percentile, quantile, histogram, etc.)
- Array indexing and slicing
- Type conversion and dtype handling

### What is NOT in scope
- File I/O operations (save, load, etc.)
- Polynomial functions
- FFT functions
- Window functions
- Date/time handling
- NumPy C API bindings
- Low-level memory views

## Public API / Interface

### MCP Tools

All tools accept JSON-serializable inputs and return JSON-serializable outputs. Arrays are converted to lists.

#### Array Creation Tools

| Tool | Description | Signature |
|------|-------------|-----------|
| `np_array` | Create NumPy array | `(data: list, dtype: str = "float64") -> list` |
| `np_zeros` | Create zeros array | `(shape: int \| list, dtype: str = "float64") -> list` |
| `np_ones` | Create ones array | `(shape: int \| list, dtype: str = "float64") -> list |
| `np_full` | Create array filled with value | `(shape: int \| list, fill_value: float, dtype: str = "float64") -> list` |
| `np_arange` | Create array with range | `(start: float, stop: float, step: float = 1, dtype: str = "float64") -> list` |
| `np_linspace` | Create evenly spaced array | `(start: float, stop: float, num: int = 50, dtype: str = "float64") -> list` |
| `np_eye` | Create identity matrix | `(N: int, M: int = None, dtype: str = "float64") -> list` |
| `np_diag` | Create diagonal array | `(k: list \| int, dtype: str = "float64") -> list` |

#### Array Manipulation Tools

| Tool | Description | Signature |
|------|-------------|-----------|
| `np_reshape` | Reshape array | `(array: list, newshape: int \| list) -> list` |
| `np_transpose` | Transpose array | `(array: list, axes: list = None) -> list` |
| `np_concatenate` | Concatenate arrays | `(arrays: list, axis: int = 0) -> list` |
| `np_split` | Split array | `(array: list, indices_or_sections: int \| list, axis: int = 0) -> list` |
| `np_tile` | Tile array | `(array: list, reps: int \| list) -> list` |
| `np_repeat` | Repeat elements | `(array: list, repeats: int, axis: int = None) -> list` |
| `np_squeeze` | Remove single-dimensional entries | `(array: list, axis: int \| list = None) -> list` |
| `np_flatten` | Flatten array | `(array: list) -> list` |

#### Mathematical Operations Tools

| Tool | Description | Signature |
|------|-------------|-----------|
| `np_sum` | Sum of elements | `(array: list, axis: int \| list = None, dtype: str = "float64") -> float \| list` |
| `np_mean` | Mean of elements | `(array: list, axis: int \| list = None) -> float \| list` |
| `np_std` | Standard deviation | `(array: list, axis: int \| list = None, ddof: int = 0) -> float \| list` |
| `np_var` | Variance | `(array: list, axis: int \| list = None, ddof: int = 0) -> float \| list` |
| `np_min` | Minimum value | `(array: list, axis: int \| list = None) -> float \| list` |
| `np_max` | Maximum value | `(array: list, axis: int \| list = None) -> float \| list` |
| `np_argmin` | Index of minimum | `(array: list, axis: int = None) -> int \| list` |
| `np_argmax` | Index of maximum | `(array: list, axis: int = None) -> int \| list` |
| `np_dot` | Dot product | `(a: list, b: list) -> list \| float` |
| `np_matmul` | Matrix multiplication | `(a: list, b: list) -> list` |
| `np_cross` | Cross product | `(a: list, b: list) -> list` |
| `np_trace` | Sum of diagonal | `(array: list, offset: int = 0) -> float` |
| `np_cumsum` | Cumulative sum | `(array: list, axis: int = None) -> list` |
| `np_cumprod` | Cumulative product | `(array: list, axis: int = None) -> list` |
| `np_diff` | Differences | `(array: list, n: int = 1, axis: int = 0) -> list` |

#### Linear Algebra Tools

| Tool | Description | Signature |
|------|-------------|-----------|
| `np_inv` | Matrix inverse | `(array: list) -> list` |
| `np_det` | Matrix determinant | `(array: list) -> float` |
| `np_eig` | Eigenvalues and eigenvectors | `(array: list) -> dict` |
| `np_svd` | Singular value decomposition | `(array: list, full_matrices: bool = False) -> dict` |
| `np_solve` | Solve linear system | `(a: list, b: list) -> list` |
| `np_linalg_norm` | Matrix/vector norm | `(array: list, ord: str = "fro") -> float` |

#### Random Tools

| Tool | Description | Signature |
|------|-------------|-----------|
| `np_rand` | Random floats | `(shape: int \| list) -> list` |
| `np_randn` | Random normal | `(shape: int \| list) -> list` |
| `np_randint` | Random integers | `(low: int, high: int = None, size: int \| list = None) -> list` |
| `np_random_choice` | Random choice | `(a: list, size: int = None, replace: bool = True) -> list` |
| `np_shuffle` | Shuffle array | `(array: list) -> list` |

#### Statistical Tools

| Tool | Description | Signature |
|------|-------------|-----------|
| `np_percentile` | Percentiles | `(array: list, q: float \| list) -> float \| list` |
| `np_quantile` | Quantiles | `(array: list, q: float \| list) -> float \| list` |
| `np_histogram` | Histogram | `(array: list, bins: int \| list = 10, range: list = None) -> dict` |
| `np_correlate` | Cross-correlation | `(a: list, b: list, mode: str = "full") -> list` |
| `np_corrcoef` | Correlation coefficient | `(array: list, rowvar: bool = True) -> list` |

#### Element-wise Math Tools

| Tool | Description | Signature |
|------|-------------|-----------|
| `np_add` | Element-wise add | `(a: list, b: list) -> list` |
| `np_subtract` | Element-wise subtract | `(a: list, b: list) -> list` |
| `np_multiply` | Element-wise multiply | `(a: list, b: list) -> list` |
| `np_divide` | Element-wise divide | `(a: list, b: list) -> list` |
| `np_power` | Element-wise power | `(a: list, b: list \| float) -> list` |
| `np_mod` | Element-wise mod | `(a: list, b: list) -> list` |
| `np_sqrt` | Square root | `(array: list) -> list` |
| `np_abs` | Absolute value | `(array: list) -> list` |
| `np_exp` | Exponential | `(array: list) -> list` |
| `np_log` | Natural log | `(array: list) -> list` |
| `np_log10` | Log base 10 | `(array: list) -> list` |
| `np_sin` | Sine | `(array: list) -> list` |
| `np_cos` | Cosine | `(array: list) -> list` |
| `np_tan` | Tangent | `(array: list) -> list` |
| `np_arcsin` | Arc sine | `(array: list) -> list` |
| `np_arccos` | Arc cosine | `(array: list) -> list` |
| `np_arctan` | Arc tangent | `(array: list) -> list` |
| `np_sinh` | Hyperbolic sine | `(array: list) -> list` |
| `np_cosh` | Hyperbolic cosine | `(array: list) -> list` |
| `np_tanh` | Hyperbolic tangent | `(array: list) -> list` |

#### Array Properties Tools

| Tool | Description | Signature |
|------|-------------|-----------|
| `np_shape` | Get array shape | `(array: list) -> list` |
| `np_ndim` | Get number of dimensions | `(array: list) -> int` |
| `np_size` | Get total size | `(array: list) -> int` |
| `np_dtype` | Get dtype | `(array: list) -> str` |
| `npastype` | Convert dtype | `(array: list, dtype: str) -> list` |

## Data Formats

- Input arrays: JSON lists (nested for multidimensional)
- Output arrays: JSON lists
- Shapes: integers or lists of integers
- dtypes: NumPy dtype strings ("float64", "int32", "complex128", etc.)
- Returns include metadata where helpful (shape, dtype, eigenvalues, etc.)

## Edge Cases

1. **Empty arrays**: Handle gracefully, return empty list
2. **Single element arrays**: Return scalar when appropriate
3. **Mismatched shapes**: Raise descriptive error
4. **Invalid dtypes**: Raise descriptive error
5. **Non-numeric data**: Handle with clear error messages
6. **Singular matrices**: For inv/det operations, handle with error
7. **NaN/Inf values**: Propagate as per NumPy behavior
8. **Large arrays**: No explicit limit, but warn about memory

## Performance & Constraints

- All operations use NumPy's optimized C backend
- JSON serialization may be slow for very large arrays (>1M elements)
- No parallelization needed - NumPy handles internally
- Memory usage is NumPy's responsibility
