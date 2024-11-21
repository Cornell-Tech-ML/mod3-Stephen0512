from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Apply the scalar function to the given values.

        This method processes the input values, handles both Scalar and non-Scalar inputs,
        applies the forward function, and creates a new Scalar with the result.

        Args:
        ----
            *vals: Variable number of input values, which can be Scalars or floats.

        Returns:
        -------
            A new Scalar instance containing the result of the forward function and its history needed.

        """
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute the sum of two input float numbers.

        Args:
        ----
            ctx: The context object used to save values for the backward method.
            a: The first input float number.
            b: The second input float number.

        Returns:
        -------
            The sum of a and b.

        """
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Compute the local derivatives with respect to the inputs.

        Args:
        ----
            ctx: The context object with saved values from the forward method.
            d_output: The local derivative of the higher-order function.

        Returns:
        -------
            A tuple containing the local derivatives with respect to each input.

        """
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the natural logarithm of the input value.

        Args:
        ----
            ctx: The context object used to save values for the backward method.
            a: The input float number.

        Returns:
        -------
            The natural logarithm of a.

        """
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the local derivatives with respect to the inputs.

        Args:
        ----
            ctx: The context object with saved values from the forward method.
            d_output: The local derivative of the higher-order function.

        Returns:
        -------
            The local derivative with respect to the input.

        """
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


# ## Task 1.2

# Implementation of forward and backward methods for a set of basic mathematical functions.

# Mathematical functions:
# - mul
# - inv
# - neg
# - sigmoid
# - relu
# - exp
# - lt
# - eq


class Mul(ScalarFunction):
    """Multiplication function $f(x, y) = x * y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute the product of two input float numbers.

        Args:
        ----
            ctx: The context object used to save values for the backward method.
            a: The first input float number.
            b: The second input float number.

        Returns:
        -------
            The product of a and b.

        """
        ctx.save_for_backward(a, b)
        return float(operators.mul(a, b))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Compute the local derivatives with respect to the inputs.

        Args:
        ----
            ctx: The context object with saved values from the forward method.
            d_output: The local derivative of the higher-order function.

        Returns:
        -------
            A tuple containing the local derivatives with respect to each input.

        """
        (a, b) = ctx.saved_values
        return float(b * d_output), float(a * d_output)


class Inv(ScalarFunction):
    """Inverse function $f(x) = 1/x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the inverse of the input float number.

        Args:
        ----
            ctx: The context object used to save values for the backward method.
            a: The input float number.

        Returns:
        -------
            The inverse of a.

        """
        ctx.save_for_backward(a)
        return float(operators.inv(a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the local derivative with respect to the input.

        Args:
        ----
            ctx: The context object with saved values from the forward method.
            d_output: The local derivative of the higher-order function.

        Returns:
        -------
            The local derivative with respect to the input.

        """
        (a,) = ctx.saved_values
        return float(operators.inv_back(a, d_output))


class Neg(ScalarFunction):
    """Negation function $f(x) = -x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the negation of the input float number.

        Args:
        ----
            ctx: The context object used to save values for the backward method.
            a: The input float number.

        Returns:
        -------
            The negation of a.

        """
        return float(operators.neg(a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the local derivative with respect to the input.

        Args:
        ----
            ctx: The context object with saved values from the forward method.
            d_output: The local derivative of the higher-order function.

        Returns:
        -------
            The local derivative with respect to the input.

        """
        return float(-d_output)


class Sigmoid(ScalarFunction):
    """Sigmoid function $f(x) = 1 / (1 + e^{-x})$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the sigmoid of the input float number.

        Args:
        ----
            ctx: The context object used to save values for the backward method.
            a: The input float number.

        Returns:
        -------
            The sigmoid of a.

        """
        ctx.save_for_backward(a)
        return float(operators.sigmoid(a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the local derivative with respect to the input.

        Args:
        ----
            ctx: The context object with saved values from the forward method.
            d_output: The local derivative of the higher-order function.

        Returns:
        -------
            The local derivative with respect to the input.

        """
        (a,) = ctx.saved_values
        sigmoid_value = float(operators.sigmoid(a))
        return (
            sigmoid_value * (1 - sigmoid_value) * d_output
        )  # $f'(x) = f(x) * (1 - f(x))$


class ReLU(ScalarFunction):
    """ReLU function $f(x) = max(0, x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the ReLU of the input float number.

        Args:
        ----
            ctx: The context object used to save values for the backward method.
            a: The input float number.

        Returns:
        -------
            The ReLU of a.

        """
        ctx.save_for_backward(a)
        return float(operators.relu(a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the local derivative with respect to the input.

        Args:
        ----
            ctx: The context object with saved values from the forward method.
            d_output: The local derivative of the higher-order function.

        Returns:
        -------
            The local derivative with respect to the input.

        """
        (a,) = ctx.saved_values
        return float(operators.relu_back(a, d_output))


class Exp(ScalarFunction):
    """Exponential function $f(x) = e^x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the exponential of the input float number.

        Args:
        ----
            ctx: The context object used to save values for the backward method.
            a: The input float number.

        Returns:
        -------
            The exponential of a.

        """
        ctx.save_for_backward(a)
        return float(operators.exp(a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the local derivative with respect to the input.

        Args:
        ----
            ctx: The context object with saved values from the forward method.
            d_output: The local derivative of the higher-order function.

        Returns:
        -------
            The local derivative with respect to the input.

        """
        (a,) = ctx.saved_values
        exp_value = float(operators.exp(a))
        return exp_value * d_output  # $f'(x) = f(x)$


class LT(ScalarFunction):
    """Less than function $f(x, y) = 1 if x < y else 0$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute the less than comparison of two input float numbers.

        Args:
        ----
            ctx: The context object used to save values for the backward method.
            a: The first input float number.
            b: The second input float number.

        Returns:
        -------
            1.0 if a < b, else 0.0.

        """
        return float(operators.lt(a, b))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Compute the local derivatives with respect to the inputs.

        Args:
        ----
            ctx: The context object with saved values from the forward method.
            d_output: The local derivative of the higher-order function.

        Returns:
        -------
            A tuple containing the local derivatives with respect to each input.

        """
        return 0.0, 0.0


class EQ(ScalarFunction):
    """Equal function $f(x, y) = 1.0 if x == y else 0.0$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute the equality comparison of two input float numbers.

        Args:
        ----
            ctx: The context object used to save values for the backward method.
            a: The first input float number.
            b: The second input float number.

        Returns:
        -------
            1.0 if a == b, else 0.0.

        """
        return float(operators.eq(a, b))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Compute the local derivatives with respect to the inputs.

        Args:
        ----
            ctx: The context object with saved values from the forward method.
            d_output: The local derivative of the higher-order function.

        Returns:
        -------
            A tuple containing the local derivatives with respect to each input.

        """
        return 0.0, 0.0
