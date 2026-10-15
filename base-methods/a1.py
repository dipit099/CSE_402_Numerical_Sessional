import numpy as np

def significant_digits(relative_error):
    return np.floor(2 - np.log10(relative_error / 0.005))


def newton_raphson_method(f, f_prime, x, tol=1e-10, max_iter=100):
    root_estimates = [x]
    relative_errors = []

    fx = f(x)
    if fx == 0:
        return x, root_estimates, relative_errors

    print(f"{'Iter':<6}{'Vi':<15}{'f(Vi)':<15}{'f_prime(Vi)':<15}{'Vi+1':<15}{'ea':<15}")

    for i in range(max_iter):
        fpx = f_prime(x)
        if fpx == 0:
            return x, root_estimates, relative_errors

        x_new = x - fx / fpx
        root_estimates.append(x_new)

        error = abs((x_new - x) / x_new)
        relative_errors.append(error)

        print(f"{i+1:<6}{x:<15.6g}{fx:<15.6g}{fpx:<15.6g}{x_new:<15.6g}{error:<15.6g}")

        if error <= tol:
            return x_new, root_estimates, relative_errors
        
        x = x_new
        fx = f(x)

    return x, root_estimates, relative_errors


def main():
    n = 1.8
    vO = 0.65
    vT = 0.02585
    R = 0.5 * (10 ** 3)
    I = 0.2 * (10 ** (-3))
    f = lambda v : (10 ** (-12)) * (np.exp(v / (n * vT)) - 1) + (v / R) - I
    f_p = lambda v : (10 ** (-12)) * (1 / (n * vT)) * np.exp(v / (n * vT)) + (1 / R)

    v, _, relative_errors = newton_raphson_method(f, f_p, vO, 1e-6)
    last_error = relative_errors[-1]
    sf = int(significant_digits(last_error))
    print(f"Root: {v:.{sf}g}")
    print(f"S.f: {sf}")


if __name__ == "__main__":
    main()
