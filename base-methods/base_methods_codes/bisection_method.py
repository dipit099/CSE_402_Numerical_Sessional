import math


# Example: f(x) = x^3 - x - 2
# Change this function in the exam.
def f(x):
    return x**3 - x - 2


def bisection(a, b, es=0.001, max_iter=100):
    if f(a) * f(b) > 0:
        print("Invalid interval: f(a) and f(b) must have opposite signs.")
        return None


    old_mid = None

    print("iter         a         b       mid      f(mid)      ea(%)")

    for i in range(1, max_iter + 1):
        mid = (a + b) / 2
        fmid = f(mid)

        if old_mid is None:
            ea = None
        else:
            ea = abs((mid - old_mid) / mid) * 100

        if ea is None:
            print(f"{i:4d}  {a:8.5f}  {b:8.5f}  {mid:8.5f}  {fmid:10.5f}       ---")
        else:
            print(f"{i:4d}  {a:8.5f}  {b:8.5f}  {mid:8.5f}  {fmid:10.5f}  {ea:8.5f}")

        if fmid == 0 or (ea is not None and ea <= es):
            return mid #ZERO holei stop bcz root peye gesi / error rate komleo stop jdi es deya thake.

        if f(a) * fmid < 0:
            b = mid
        else:
            a = mid

        old_mid = mid

    return mid


root = bisection(a=1, b=2, es=0.001)

if root is not None:
    print(f"\nApproximate root = {root:.6f}")

