import pytest


@pytest.fixture
def assert_almost_equal():
    """Keep the original decimal rounding and absolute delta checks."""
    def compare(actual, expected, places=7, delta=None):
        if actual == expected:
            return
        difference = abs(actual - expected)
        if delta is not None:
            assert difference <= delta
        else:
            assert round(difference, places) == 0

    return compare
