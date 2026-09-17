"""Local hosting remains independent of optional model engines."""

from exp.optimize.claas.backends import local


def test_package_import() -> None:
    """The package exports no eager runtime or provider initialization."""
    assert local.__doc__
