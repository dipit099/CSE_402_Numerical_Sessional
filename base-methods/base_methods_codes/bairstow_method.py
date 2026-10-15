import math


def bairstow(coefficients, r, s, es=0.001, max_iter=100):
    """
    Finds roots of a polynomial using Bairstow's method.

    coefficients must be written from highest power to constant term.
    Example:
        x^4 - 5x^3 + 7x^2 - 5x + 6
        coefficients = [1, -5, 7, -5, 6]

    The quadratic factor form used here is:
        x^2 - r*x - s
    """
    a = coefficients[:]
    roots = []

    while len(a) > 3:
        n = len(a) - 1

        print(f"\nFinding quadratic factor for degree {n} polynomial")
        print("iter          r          s         dr         ds      ea_r(%)   ea_s(%)")

        for iteration in range(1, max_iter + 1):
            b = [0] * (n + 1)
            c = [0] * (n + 1)

            b[0] = a[0]
            b[1] = a[1] + r * b[0]

            for i in range(2, n + 1):
                b[i] = a[i] + r * b[i - 1] + s * b[i - 2]

            c[0] = b[0]
            c[1] = b[1] + r * c[0]

            for i in range(2, n + 1):
                c[i] = b[i] + r * c[i - 1] + s * c[i - 2]

            determinant = c[n - 2] * c[n - 2] - c[n - 3] * c[n - 1]

            if determinant == 0:
                print("Zero determinant. Try different initial r and s.")
                return roots

            dr = (-b[n - 1] * c[n - 2] + b[n] * c[n - 3]) / determinant
            ds = (-b[n] * c[n - 2] + b[n - 1] * c[n - 1]) / determinant

            r = r + dr
            s = s + ds

            ea_r = abs(dr / r) * 100 if r != 0 else abs(dr) * 100
            ea_s = abs(ds / s) * 100 if s != 0 else abs(ds) * 100

            print(
                f"{iteration:4d}  {r:9.5f}  {s:9.5f}  {dr:9.5f}  {ds:9.5f}"
                f"  {ea_r:9.5f}  {ea_s:9.5f}"
            )

            if ea_r <= es and ea_s <= es:
                break

        # Roots of x^2 - r*x - s = 0
        discriminant = r**2 + 4 * s

        if discriminant >= 0:
            root1 = (r + math.sqrt(discriminant)) / 2
            root2 = (r - math.sqrt(discriminant)) / 2
        else:
            real_part = r / 2
            imag_part = math.sqrt(abs(discriminant)) / 2
            root1 = complex(real_part, imag_part)
            root2 = complex(real_part, -imag_part)

        roots.append(root1)
        roots.append(root2)

        # Deflate polynomial using quotient b[0] to b[n-2]
        a = b[: n - 1]

    if len(a) == 3:
        aa, bb, cc = a
        discriminant = bb**2 - 4 * aa * cc

        if discriminant >= 0:
            roots.append((-bb + math.sqrt(discriminant)) / (2 * aa))
            roots.append((-bb - math.sqrt(discriminant)) / (2 * aa))
        else:
            real_part = -bb / (2 * aa)
            imag_part = math.sqrt(abs(discriminant)) / (2 * aa)
            roots.append(complex(real_part, imag_part))
            roots.append(complex(real_part, -imag_part))

    elif len(a) == 2:
        aa, bb = a
        roots.append(-bb / aa)

    return roots


# Example polynomial:
# f(x) = x^4 - 5x^3 + 7x^2 - 5x + 6
coefficients = [1, -5, 7, -5, 6]

# Initial guesses for quadratic factor x^2 - r*x - s
roots = bairstow(coefficients, r=1, s=1, es=0.001)

print("\nRoots:")
for root in roots:
    print(root)

