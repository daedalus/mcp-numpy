
import pytest
from hypothesis import Verbosity, settings

settings.register_profile("ci", verbosity=Verbosity.verbose)


@pytest.fixture
def sample_array_1d() -> list[float]:
    return [1.0, 2.0, 3.0, 4.0, 5.0]


@pytest.fixture
def sample_array_2d() -> list[list[float]]:
    return [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]


@pytest.fixture
def sample_matrix() -> list[list[float]]:
    return [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]


@pytest.fixture
def identity_matrix_3x3() -> list[list[float]]:
    return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
