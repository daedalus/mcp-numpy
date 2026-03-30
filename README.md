# mcp-numpy

> An MCP server that exposes NumPy functionality

[![PyPI](https://img.shields.io/pypi/v/mcp-numpy.svg)](https://pypi.org/project/mcp-numpy/)
[![Python](https://img.shields.io/pypi/pyversions/mcp-numpy.svg)](https://pypi.org/project/mcp-numpy/)
[![Coverage](https://codecov.io/gh/daedalus/mcp-numpy/branch/main/graph/badge.svg)](https://codecov.io/gh/daedalus/mcp-numpy)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Install

```bash
pip install mcp-numpy
```

## Usage

### As an MCP Server

To use with Claude Desktop or other MCP clients, add to your `mcp.json`:

```json
{
  "mcpServers": {
    "mcp-numpy": {
      "command": "mcp-numpy"
    }
  }
}
```

### Available Tools

The server exposes the following NumPy functionality as MCP tools:

#### Array Creation
- `np_array` - Create a NumPy array
- `np_zeros` - Create zeros array
- `np_ones` - Create ones array
- `np_full` - Create array filled with value
- `np_arange` - Create array with range
- `np_linspace` - Create evenly spaced array
- `np_eye` - Create identity matrix
- `np_diag` - Create diagonal array

#### Array Manipulation
- `np_reshape` - Reshape array
- `np_transpose` - Transpose array
- `np_concatenate` - Concatenate arrays
- `np_split` - Split array
- `np_tile` - Tile array
- `np_repeat` - Repeat elements
- `np_squeeze` - Remove single-dimensional entries
- `np_flatten` - Flatten array

#### Mathematical Operations
- `np_sum`, `np_mean`, `np_std`, `np_var` - Summary statistics
- `np_min`, `np_max`, `np_argmin`, `np_argmax` - Min/max operations
- `np_dot`, `np_matmul`, `np_cross` - Matrix operations
- `np_trace`, `np_cumsum`, `np_cumprod`, `np_diff` - Array operations

#### Linear Algebra
- `np_inv` - Matrix inverse
- `np_det` - Matrix determinant
- `np_eig` - Eigenvalues and eigenvectors
- `np_svd` - Singular value decomposition
- `np_solve` - Solve linear system
- `np_linalg_norm` - Matrix/vector norm

#### Random
- `np_rand` - Random floats
- `np_randn` - Random normal
- `np_randint` - Random integers
- `np_random_choice` - Random choice
- `np_shuffle` - Shuffle array

#### Statistics
- `np_percentile`, `np_quantile` - Percentiles/quantiles
- `np_histogram` - Histogram
- `np_correlate`, `np_corrcoef` - Correlation

#### Element-wise Math
- `np_add`, `np_subtract`, `np_multiply`, `np_divide` - Arithmetic
- `np_power`, `np_mod` - Power and modulo
- `np_sqrt`, `np_abs` - Basic math
- `np_exp`, `np_log`, `np_log10` - Logarithms
- `np_sin`, `np_cos`, `np_tan` - Trigonometry
- `np_arcsin`, `np_arccos`, `np_arctan` - Inverse trig
- `np_sinh`, `np_cosh`, `np_tanh` - Hyperbolic

#### Array Properties
- `np_shape`, `np_ndim`, `np_size`, `np_dtype` - Properties
- `npastype` - Type conversion

## Development

```bash
git clone https://github.com/daedalus/mcp-numpy.git
cd mcp-numpy
pip install -e ".[test]"

# run tests
pytest

# format
ruff format src/ tests/

# lint
ruff check src/ tests/

# type check
mypy src/
```

mcp-name: io.github.daedalus/mcp-numpy
