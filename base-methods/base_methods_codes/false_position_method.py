import math


# Example: f(x) = x^3 - x - 2
# Change this function in the exam.
def f(x):
    return x**3 - x - 2


def false_position(a, b, es=0.001, max_iter=100):
    if f(a) * f(b) > 0:
        print("Invalid interval: f(a) and f(b) must have opposite signs.")
        return None

    old_c = None

    print("iter         a         b         c        f(c)      ea(%)")

    for i in range(1, max_iter + 1):
        fa = f(a)
        fb = f(b)

        c = (a * fb - b * fa) / (fb - fa)
        fc = f(c)

        if old_c is None:
            ea = None
        else:
            ea = abs((c - old_c) / c) * 100

        if ea is None:
            print(f"{i:4d}  {a:8.5f}  {b:8.5f}  {c:8.5f}  {fc:10.5f}       ---")
        else:
            print(f"{i:4d}  {a:8.5f}  {b:8.5f}  {c:8.5f}  {fc:10.5f}  {ea:8.5f}")

        if fc == 0 or (ea is not None and ea <= es):
            return c

        if fa * fc < 0:
            b = c
        else:
            a = c

        old_c = c

    return c


root = false_position(a=1, b=2, es=0.001)

if root is not None:
    print(f"\nApproximate root = {root:.6f}")


print(f"\nAppro")