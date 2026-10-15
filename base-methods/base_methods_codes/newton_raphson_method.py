import math


# Example: f(x) = x^3 - x - 2
# Change this function in the exam.
def f(x):
    return x**3 - x - 2


# Derivative of f(x)
# Change this derivative in the exam.
def df(x):
    return 3 * x**2 - 1


def newton_raphson(x0, es=0.001, max_iter=100):
    old_x = x0

    print("iter       old_x       new_x      f(new_x)      ea(%)")

    for i in range(1, max_iter + 1):
        if df(old_x) == 0:
            print("Derivative is zero. Method fails.")
            return None

        new_x = old_x - f(old_x) / df(old_x)
        ea = abs((new_x - old_x) / new_x) * 100

        print(f"{i:4d}  {old_x:10.5f}  {new_x:10.5f}  {f(new_x):12.5f}  {ea:8.5f}")

        if ea <= es:
            return new_x

        old_x = new_x

    return new_x


root = newton_raphson(x0=1.5, es=0.001)

if root is not None:
    print(f"\nApproximate root = {root:.6f}")

