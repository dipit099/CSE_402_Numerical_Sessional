import cmath
import math


def bisection_method(func, x_l, x_u, tol=1e-5, max_iter=100):
    """
    Find a root of func in [x_l, x_u] using the Bisection Method.

    Returns:
        {
            'root': float,
            'iterations': int,
            'estimated_error': float
        }
    """
    f_l = func(x_l)
    f_u = func(x_u)

    if f_l * f_u >= 0:
        raise ValueError(
            "Sign condition is not met: f(x_l) and f(x_u) must have opposite "
            "signs, so a root is not guaranteed in the bracket."
        )

    x_m_old = None
    estimated_error = math.inf

    for iteration in range(1, max_iter + 1):
        x_m = (x_l + x_u) / 2.0
        f_m = func(x_m)

        if x_m_old is not None:
            if x_m != 0:
                estimated_error = abs((x_m - x_m_old) / x_m) * 100.0
            else:
                estimated_error = 0.0 if x_m == x_m_old else math.inf

        if f_m == 0 or estimated_error <= tol:
            return {
                "root": float(x_m),
                "iterations": iteration,
                "estimated_error": float(estimated_error),
            }

        if f_l * f_m < 0:
            x_u = x_m
            f_u = f_m
        else:
            x_l = x_m
            f_l = f_m

        x_m_old = x_m

    return {
        "root": float(x_m),
        "iterations": max_iter,
        "estimated_error": float(estimated_error),
    }


def false_position_method(func, x_l, x_u, tol=1e-5, max_iter=100):
    """
    Find a root of func in [x_l, x_u] using the False Position Method.

    Returns:
        {
            'root': float,
            'iterations': int,
            'estimated_error': float
        }
    """
    f_l = func(x_l)
    f_u = func(x_u)

    if f_l * f_u >= 0:
        raise ValueError(
            "Sign condition is not met: f(x_l) and f(x_u) must have opposite "
            "signs, so a root is not guaranteed in the bracket."
        )

    x_r_old = None
    estimated_error = math.inf

    for iteration in range(1, max_iter + 1):
        denominator = f_u - f_l
        if abs(denominator) < 1e-15:
            raise ZeroDivisionError("False position denominator is too close to zero.")

        x_r = x_u - f_u * (x_l - x_u) / (f_l - f_u)
        f_r = func(x_r)

        if x_r_old is not None:
            if x_r != 0:
                estimated_error = abs((x_r - x_r_old) / x_r) * 100.0
            else:
                estimated_error = 0.0 if x_r == x_r_old else math.inf

        if f_r == 0 or estimated_error <= tol:
            return {
                "root": float(x_r),
                "iterations": iteration,
                "estimated_error": float(estimated_error),
            }

        if f_l * f_r < 0:
            x_u = x_r
            f_u = f_r
        else:
            x_l = x_r
            f_l = f_r

        x_r_old = x_r

    return {
        "root": float(x_r),
        "iterations": max_iter,
        "estimated_error": float(estimated_error),
    }


def newton_raphson_method(func, dfunc, x_0, tol=1e-5, max_iter=100):
    """
    Find a root using the Newton-Raphson Method.

    Returns:
        {
            'root': float,
            'iterations': int,
            'estimated_error': float
        }
    """
    x_i = float(x_0)
    estimated_error = math.inf

    for iteration in range(1, max_iter + 1):
        slope = dfunc(x_i)

        if abs(slope) < 1e-12:
            raise ZeroDivisionError(
                "Newton-Raphson method diverges because the derivative is too "
                "close to zero."
            )

        x_next = x_i - func(x_i) / slope

        if x_next != 0:
            estimated_error = abs((x_next - x_i) / x_next) * 100.0
        else:
            estimated_error = 0.0 if x_next == x_i else math.inf

        if estimated_error <= tol:
            return {
                "root": float(x_next),
                "iterations": iteration,
                "estimated_error": float(estimated_error),
            }

        x_i = x_next

    return {
        "root": float(x_i),
        "iterations": max_iter,
        "estimated_error": float(estimated_error),
    }


def bairstows_method(coefficients, r_guess=0.1, s_guess=0.1, tol=1e-5, max_iter=100):
    """
    Extract all roots of a polynomial using Bairstow's Method.

    coefficients are in ascending power order:
        [a_0, a_1, ..., a_n]
    """
    if len(coefficients) < 2:
        return []

    polynomial = [float(value) for value in coefficients]
    roots = []

    while len(polynomial) > 1 and abs(polynomial[-1]) < 1e-15:
        polynomial.pop()

    while len(polynomial) >= 4:
        n = len(polynomial) - 1
        r = float(r_guess)
        s = float(s_guess)

        for _ in range(max_iter):
            b = _synthetic_division_b(polynomial, r, s)

            if max(abs(b[0]), abs(b[1])) <= tol:
                break

            c = _synthetic_division_c(b, r, s)

            determinant = c[1] * c[3] - c[2] * c[2]
            if abs(determinant) < 1e-15:
                raise ZeroDivisionError(
                    "Bairstow correction system is singular; try different "
                    "initial guesses for r and s."
                )

            delta_r = (-b[0] * c[3] + b[1] * c[2]) / determinant
            delta_s = (-c[1] * b[1] + c[2] * b[0]) / determinant

            r += delta_r
            s += delta_s
        else:
            raise RuntimeError("Bairstow's method did not converge within max_iter.")

        roots.extend(_quadratic_roots_from_bairstow_factor(r, s))

        b = _synthetic_division_b(polynomial, r, s)
        polynomial = b[2 : n + 1]

        while len(polynomial) > 1 and abs(polynomial[-1]) < 1e-15:
            polynomial.pop()

    if len(polynomial) == 3:
        roots.extend(_quadratic_roots_ascending(polynomial))
    elif len(polynomial) == 2:
        roots.append(_clean_complex(-polynomial[0] / polynomial[1]))

    return roots


def _synthetic_division_b(coefficients, r, s):
    """Synthetic division for b_k = a_k + r*b_{k+1} + s*b_{k+2}."""
    n = len(coefficients) - 1
    b = [0.0] * (n + 3)

    for k in range(n, -1, -1):
        b[k] = coefficients[k] + r * b[k + 1] + s * b[k + 2]

    return b


def _synthetic_division_c(b, r, s):
    """Second synthetic division for c_k = b_k + r*c_{k+1} + s*c_{k+2}."""
    n = len(b) - 3
    c = [0.0] * (n + 4)

    for k in range(n, -1, -1):
        c[k] = b[k] + r * c[k + 1] + s * c[k + 2]

    return c


def _quadratic_roots_from_bairstow_factor(r, s):
    """Roots of x^2 - r*x - s = 0."""
    discriminant = complex(r * r + 4.0 * s)
    sqrt_discriminant = cmath.sqrt(discriminant)

    return [
        _clean_complex((r + sqrt_discriminant) / 2.0),
        _clean_complex((r - sqrt_discriminant) / 2.0),
    ]


def _quadratic_roots_ascending(coefficients):
    """Roots of a_0 + a_1*x + a_2*x^2 = 0."""
    a_0, a_1, a_2 = coefficients
    discriminant = complex(a_1 * a_1 - 4.0 * a_2 * a_0)
    sqrt_discriminant = cmath.sqrt(discriminant)

    return [
        _clean_complex((-a_1 + sqrt_discriminant) / (2.0 * a_2)),
        _clean_complex((-a_1 - sqrt_discriminant) / (2.0 * a_2)),
    ]


def _clean_complex(value):
    """Return a float when the imaginary part is negligible."""
    if isinstance(value, complex) and abs(value.imag) < 1e-12:
        return float(value.real)
    return value

