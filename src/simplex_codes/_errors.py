"""Typed solver failures, retaining the public exception base classes."""


class InfeasibleError(ValueError):
    """The model has no feasible solution."""


class UnboundedError(ValueError):
    """The objective can improve without limit."""


class IterationLimitError(RuntimeError):
    """The pivot budget was exhausted."""


class NumericalError(RuntimeError):
    """The solver could not verify a numerical result."""
