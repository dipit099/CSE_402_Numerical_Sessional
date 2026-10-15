"""
Root-Finding Methods -- Exam Template
--------------------------------------------------------------
Covers, based on previous-year exam patterns (A1/B1/B2/C1/C2):
 1. Approximate relative error (shared stopping criterion, all methods)
 2. Sign-change interval scanning (step size h) + direct-root detection
 3. Bisection method
 4. False Position (Regula Falsi) method
 5. Newton-Raphson method
 6. Plotting (linear and log-x scale)
 7. Bisection vs False Position comparison (endpoint-update counters)

NOT covered: Bairstow's method (polynomial/complex roots) -- none of the
five previous exam questions asked for it; see base_methods_codes/bairstow_method.py
if a future question needs it.

HOW TO USE IN THE EXAM: copy the relevant function(s), swap in the exam's
f(x)/df(x), interval or initial guess, and es (tolerance), then call the
matching print_*_table() to get the exact table format expected.
"""

import math

import matplotlib.pyplot as plt
import numpy as np


def approx_relative_error(new_value, old_value):
    if new_value == 0:
        if old_value == 0:
            return 0.0
        else:
            return math.inf

    return abs((new_value - old_value) / new_value) * 100.0


def significant_digits(relative_error_pct):
    """
    How many significant figures are trustworthy given the final ea(%).
    relative_error_pct is in PERCENT (matches approx_relative_error's
    convention), so it's divided by 100 to get the raw fraction before
    applying the standard sig-fig criterion (Scarborough):
    sf = floor(2 - log10(ea_fraction / 0.005))
    """
    ea_fraction = relative_error_pct / 100.0
    if ea_fraction <= 0:
        return math.inf
    return int(math.floor(2 - math.log10(ea_fraction / 0.005)))


def is_undefined_error(exc):
    # Any of these means f(x) (or f'(x)) couldn't be evaluated at this
    # point -- e.g. log/sqrt of a negative number, division by zero, or
    # exp() overflowing to infinity. Catching them lets a run report
    # status="undefined" instead of crashing the whole script.
    return isinstance(exc, (ValueError, ZeroDivisionError, OverflowError))


def find_sign_change_intervals(f, a, b, h, direct_root_tol=None):
    """
    Scan [a, b] in steps of h, collecting:
      - "direct" candidates: a single x where f(x) is (near) zero.
      - "interval" candidates: [x1, x2] where f changes sign.
    Pass direct_root_tol to treat |f(x)| <= tol as a direct hit (B2-style);
    leave it None to only treat an EXACT f(x) == 0 as direct.
    """

    def is_direct(x, fx):
        # If a tolerance was given, treat f(x) as "basically zero" when
        # its magnitude is within that tolerance -- this is the B2-style
        # direct-root check, since real f(x) almost never hits EXACTLY 0
        # at a grid point.
        if direct_root_tol is not None:
            if abs(fx) <= direct_root_tol:
                return True
            else:
                return False
        else:
            # No tolerance given -- only count an EXACT zero as a direct hit
            if fx == 0:
                return True
            else:
                return False

    candidates = []

    x1 = a
    f1 = f(x1)

    # check the very first point (a) too, in case the root sits right at
    # the left edge of the scan
    if is_direct(x1, f1):
        candidates.append({"type": "direct", "x": x1, "f": f1})

    while x1 < b:
        x2 = x1 + h
        if x2 > b:
            x2 = b  # don't overshoot past b on the last partial step

        f2 = f(x2)

        if is_direct(x2, f2):
            candidates.append({"type": "direct", "x": x2, "f": f2})
        elif f1 * f2 < 0:
            # sign flipped between x1 and x2 -> a root is bracketed somewhere in between
            candidates.append({"type": "interval", "a": x1, "b": x2})

        x1 = x2
        f1 = f2

    return candidates


def print_candidates(f, candidates, residual_tol=None):
    print(f"{'type':<10}{'x / interval':<22}{'f(value)':>15}{'verdict':>12}")
    for c in candidates:
        if c["type"] == "direct":
            verdict = "accept"
            if residual_tol is not None:
                verdict = "accept" if abs(c["f"]) <= residual_tol else "reject"
            print(f"{'direct':<10}{c['x']:<22.6f}{c['f']:>15.6e}{verdict:>12}")
        else:
            label = f"[{c['a']:.2f}, {c['b']:.2f}]"
            print(f"{'interval':<10}{label:<22}{'-':>15}{'pending':>12}")


def bisection_method(f, a, b, es=1e-4, max_iter=200):
    """
    Returns dict: root, table, iterations, updates_a, updates_b, status
    status in {'converged', 'max_iter', 'undefined', 'invalid_interval'}
    """
    table = []
    updates_a = 0
    updates_b = 0

    try:
        fa = f(a)
        fb = f(b)
    except Exception as exc:
        if is_undefined_error(exc):
            return {"root": None, "table": table, "iterations": 0,
                    "updates_a": 0, "updates_b": 0, "status": "undefined"}
        raise

    if fa * fb > 0:
        return {"root": None, "table": table, "iterations": 0,
                "updates_a": 0, "updates_b": 0, "status": "invalid_interval"}

    old_mid = None
    status = "max_iter"

    for i in range(1, max_iter + 1):
        mid = (a + b) / 2.0
        try:
            fmid = f(mid)
        except Exception as exc:
            if is_undefined_error(exc):
                status = "undefined"
                break
            raise

        if old_mid is None:
            ea = None
        else:
            ea = approx_relative_error(mid, old_mid)

        table.append((i, a, b, mid, fmid, ea))

        if fmid == 0 or (ea is not None and ea <= es):
            status = "converged"
            old_mid = mid
            break

        # keep the half of [a,b] that still brackets the root -- compare
        # against fa (not the OLD fmid) so this stays correct every iteration
        if fa * fmid < 0:
            b = mid
            updates_b += 1
        else:
            a = mid
            fa = fmid
            updates_a += 1

        old_mid = mid

    return {"root": old_mid, "table": table, "iterations": len(table),
            "updates_a": updates_a, "updates_b": updates_b, "status": status}


def print_bisection_table(table):
    print("iter         a         b       mid      f(mid)      ea(%)")
    for i, a, b, mid, fmid, ea in table:
        ea_str = "     ---" if ea is None else f"{ea:8.5f}"
        print(f"{i:4d}  {a:8.5f}  {b:8.5f}  {mid:8.5f}  {fmid:10.5f}  {ea_str}")


def false_position_method(f, a, b, es=1e-4, max_iter=200):
    """
    Same contract as bisection_method(), but the update rule is the
    false-position formula: x_r = (a*f(b) - b*f(a)) / (f(b) - f(a)).
    """
    table = []
    updates_a = 0
    updates_b = 0

    try:
        fa = f(a)
        fb = f(b)
    except Exception as exc:
        if is_undefined_error(exc):
            return {"root": None, "table": table, "iterations": 0,
                    "updates_a": 0, "updates_b": 0, "status": "undefined"}
        raise

    if fa * fb > 0:
        return {"root": None, "table": table, "iterations": 0,
                "updates_a": 0, "updates_b": 0, "status": "invalid_interval"}

    old_c = None
    status = "max_iter"

    for i in range(1, max_iter + 1):
        denom = fb - fa
        if abs(denom) < 1e-15:  # WHYY???? -- guards against dividing by ~0 when f(a) ≈ f(b)
            status = "undefined"
            break

        c = (a * fb - b * fa) / denom
        try:
            fc = f(c)
        except Exception as exc:
            if is_undefined_error(exc):
                status = "undefined"
                break
            raise

        if old_c is None:
            ea = None
        else:
            ea = approx_relative_error(c, old_c)

        table.append((i, a, b, c, fc, ea))

        if fc == 0 or (ea is not None and ea <= es):
            status = "converged"
            old_c = c
            break

        if fa * fc < 0:
            b = c
            fb = fc
            updates_b += 1
        else:
            a = c
            fa = fc
            updates_a += 1

        old_c = c

    return {"root": old_c, "table": table, "iterations": len(table),
            "updates_a": updates_a, "updates_b": updates_b, "status": status}


def print_false_position_table(table):
    print("iter         a         b         c        f(c)      ea(%)")
    for i, a, b, c, fc, ea in table:
        ea_str = "     ---" if ea is None else f"{ea:8.5f}"
        print(f"{i:4d}  {a:8.5f}  {b:8.5f}  {c:8.5f}  {fc:10.5f}  {ea_str}")


def newton_raphson_method(f, df, x0, es=1e-4, max_iter=200):
    """
    Table columns match A1.jpg exactly: iter, x_i, f(x_i), f'(x_i),
    x_{i+1}, ea.

    Returns dict: root, table, iterations, status
    status in {'converged', 'max_iter', 'undefined', 'zero_derivative'}
    """
    table = []
    old_x = x0
    status = "max_iter"

    for i in range(1, max_iter + 1):
        try:
            fx = f(old_x)
            dfx = df(old_x)
        except Exception as exc:
            if is_undefined_error(exc):
                status = "undefined"
                break
            raise

        # zero_derivative is its own status (not "undefined") because
        # it's a well-known specific failure mode -- a flat tangent line,
        # not a domain/overflow error.
        if dfx == 0:
            status = "zero_derivative"
            break

        new_x = old_x - fx / dfx
        ea = approx_relative_error(new_x, old_x)
        table.append((i, old_x, fx, dfx, new_x, ea))

        if ea <= es:
            status = "converged"
            old_x = new_x
            break

        old_x = new_x

    return {"root": old_x, "table": table, "iterations": len(table), "status": status}


def print_newton_raphson_table(table):
    print("iter          x_i        f(x_i)       f'(x_i)      x_(i+1)      ea(%)")
    for i, xi, fxi, dfxi, xi1, ea in table:
        print(f"{i:4d}  {xi:12.6f}  {fxi:12.6f}  {dfxi:12.6f}  {xi1:12.6f}  {ea:8.5f}")


def plot_function(f, a, b, n=1000, title="Graph of f(x)", mark_points=None):
    x = np.linspace(a, b, n)
    y = [f(float(v)) for v in x]

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label="f(x)")
    plt.axhline(0, color="black", linewidth=1)
    if mark_points:
        xs = list(mark_points)
        ys = [f(float(v)) for v in xs]
        plt.scatter(xs, ys, color="red", zorder=3, label="marked points")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.savefig("graph.png", dpi=200, bbox_inches="tight")
    plt.show()


def plot_function_logscale(f, a, b, n=1000, title="Graph of f(x) (log-x scale)"):
    """For domains like [1e-4, 1e4] (C2) where a linear x-axis is useless."""
    x = np.logspace(math.log10(a), math.log10(b), n)
    y = [f(float(v)) for v in x]

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label="f(x)")
    plt.axhline(0, color="black", linewidth=1)
    plt.xscale("log")
    plt.xlabel("x (log scale)")
    plt.ylabel("f(x)")
    plt.title(title)
    plt.grid(True, which="both")
    plt.legend()
    plt.savefig("graph.png", dpi=200, bbox_inches="tight")
    plt.show()


def compare_bisection_vs_false_position(f, a, b, es=1e-4):
    """
    Runs both methods on the SAME interval and prints a side-by-side
    comparison table: root, iteration count, f(xr), and how many times
    each endpoint (a / b) actually got updated.
    """
    bis = bisection_method(f, a, b, es=es)
    fp = false_position_method(f, a, b, es=es)

    print(f"{'Method':<16}{'Root':>14}{'Iterations':>12}{'f(xr)':>15}"
          f"{'a-updates':>11}{'b-updates':>11}{'S.f':>6}")

    for name, r in (("Bisection", bis), ("False Position", fp)):
        if r["root"] is None:
            print(f"{name:<16}{'N/A':>14}{r['iterations']:>12d}"
                  f"{'-':>15}{r['updates_a']:>11d}{r['updates_b']:>11d}  "
                  f"status={r['status']}")
            continue
        fxr = f(r["root"])
        sf = significant_digits(r["table"][-1][-1])
        print(f"{name:<16}{r['root']:>14.6f}{r['iterations']:>12d}"
              f"{fxr:>15.6e}{r['updates_a']:>11d}{r['updates_b']:>11d}{sf:>6d}")

    return bis, fp


def main():

    print("=" * 70)
    print("A1: NEWTON-RAPHSON -- diode equation")
    print("=" * 70)

    n_diode = 1.8
    VT = 0.02585
    R = 500.0
    IL = 0.2e-3  # 0.2 mA, converted to amps (SI base unit) -- R is already in ohms
    V0 = 0.65

    def f_diode(V):
        return 1e-12 * (math.exp(V / (n_diode * VT)) - 1) + V / R - IL

    def df_diode(V):
        return 1e-12 * math.exp(V / (n_diode * VT)) / (n_diode * VT) + 1 / R

    result_a1 = newton_raphson_method(f_diode, df_diode, V0, es=0.0001)
    print_newton_raphson_table(result_a1["table"])
    sf_a1 = significant_digits(result_a1["table"][-1][-1])
    print(f"\nStatus = {result_a1['status']}")
    print(f"V = {result_a1['root']:.{sf_a1}g} V  (S.f: {sf_a1})\n")

    print("=" * 70)
    print("B1: BISECTION -- multi-root sign-change scan")
    print("=" * 70)

    sth = 1.0

    def f_b1(x):
        return 0.6 * math.log(x + 1) - sth * math.sin(1.7 * x) - 0.08 * x**2 - 0.08

    plot_function(f_b1, 0, 10, title="B1: f(x) on [0, 10]")

    intervals_b1 = find_sign_change_intervals(f_b1, 0, 10, h=0.1)
    print("Sign-change intervals (h = 0.1):")
    for c in intervals_b1:
        print(f"  [{c['a']:.1f}, {c['b']:.1f}]")

    roots_b1 = []
    for k, c in enumerate(intervals_b1, start=1):
        res = bisection_method(f_b1, c["a"], c["b"], es=0.001)
        roots_b1.append(res["root"])
        print(f"\nRoot {k}: interval [{c['a']:.1f}, {c['b']:.1f}]")
        print_bisection_table(res["table"])
        sf_b1 = significant_digits(res["table"][-1][-1])
        print(f"Approximate root = {res['root']:.{sf_b1}g}  (S.f: {sf_b1})")


    print("=" * 70)
    print("B2: BISECTION -- direct-root vs interval candidates")
    print("=" * 70)

    def f_b2(x):
        return ((x - 2.5) ** 2 * (x + 1.052)) / (x - 3.551)

    plot_function(f_b2, -2, 5, title="B2: f(x) on [-2, 5]")

    candidates_b2 = find_sign_change_intervals(f_b2, -2, 5, h=0.1, direct_root_tol=1e-8)
    print("Root candidates (direct_root_tol = 1e-8):")
    print_candidates(f_b2, candidates_b2)

    print("\nResolving interval candidates with bisection "
          "(ea <= 0.0001%, accept only if |f(x_r)| <= 1e-6):")
    residual_tol = 1e-6
    for c in candidates_b2:
        if c["type"] == "direct":
            # already (approximately) the root itself -- no bracket to
            # bisect, just verify the residual directly
             
            if abs(c["f"]) <= residual_tol :
                verdict = "accept"

            else:
                verdict = "reject"
            print(f"x={c['x']:.6f} (direct) -> f(x)={c['f']:.3e}, verdict={verdict}")
            continue
        # ELSE if interval
        res = bisection_method(f_b2, c["a"], c["b"], es=0.0001)

        if res["status"] == "undefined":
            print(f"[{c['a']:.2f}, {c['b']:.2f}] -> undefined value encountered, run terminated")
            continue

        fxr = f_b2(res["root"])
        if abs(fxr) <= residual_tol:  # func r moddhe diye FINAL ta pass korar por
            verdict = "accept"  
        else:
            verdict = "reject"
        sf_b2 = significant_digits(res["table"][-1][-1])
        print(f"[{c['a']:.2f}, {c['b']:.2f}] -> root={res['root']:.{sf_b2}g} (S.f: {sf_b2}), "
              f"f(root)={fxr:.3e}, verdict={verdict}")

    print("\nWhy classify candidates this way:")
    print("- No sign change AND |f(x)| not tiny -> no root in that sub-interval,")
    print("  skip it entirely.")
    print("- Sign change without an actual root would only happen at a pole")
    print("  (function jumps from -inf to +inf without crossing 0) -- not the")
    print("  case for this polynomial-like f(x), but always worth checking.")
    print("- The residual check |f(x_r)| <= 1e-6 after bisection converges is")
    print("  a SEPARATE acceptance test from the ea<=es stopping criterion --")
    print("  ea small just means x_r stopped CHANGING, not that f(x_r) is")
    print("  actually near zero (rare, but possible near a flat region).\n")

    print("=" * 70)
    print("C1: FALSE POSITION -- first sign-change interval only")
    print("=" * 70)

    def f_c1(x):
        return x**3 - x - 1

    plot_function(f_c1, -2, 2.5, title="C1: f(x) = x^3 - x - 1")

    intervals_c1 = find_sign_change_intervals(f_c1, -2, 2.5, h=0.1)
    print("Sign-change intervals (h = 0.1):")
    for c in intervals_c1:
        print(f"  [{c['a']:.1f}, {c['b']:.1f}]")
    print("All real roots of x^3 - x - 1 lie in one of these intervals; a")
    print("cubic has at most 3 real roots, so the scan above already found")
    print("all of them (this cubic has exactly 1 real root).")

    first = intervals_c1[0]
    res_c1 = false_position_method(f_c1, first["a"], first["b"], es=0.0001)
    print(f"\nRunning false position on the FIRST interval "
          f"[{first['a']:.1f}, {first['b']:.1f}]:")
    print_false_position_table(res_c1["table"])
    sf_c1 = significant_digits(res_c1["table"][-1][-1])
    print(f"Approximate root = {res_c1['root']:.{sf_c1}g}  (S.f: {sf_c1})\n")

    print("=" * 70)
    print("C2: BISECTION vs FALSE POSITION -- ln(x) on log-scale domain")
    print("=" * 70)

    def f_c2(x):
        return math.log(x)

    a_c2, b_c2 = 1e-4, 1e4
    plot_function_logscale(f_c2, a_c2, b_c2, title="C2: f(x) = ln(x)")

    print(f"f({a_c2}) = {f_c2(a_c2):.6f},  f({b_c2}) = {f_c2(b_c2):.6f}"
          f"  -> opposite signs, interval valid.\n")

    print("Bisection iteration table:")
    bis_c2 = bisection_method(f_c2, a_c2, b_c2, es=0.0001)
    print_bisection_table(bis_c2["table"])

    print("\nFalse position iteration table:")
    fp_c2 = false_position_method(f_c2, a_c2, b_c2, es=0.0001)
    print_false_position_table(fp_c2["table"])

    print("\nComparison table:")
    compare_bisection_vs_false_position(f_c2, a_c2, b_c2, es=0.0001)

    print("\n(a) Faster method: whichever has fewer 'Iterations' above -- for")
    print("    ln(x) on this wildly asymmetric domain [1e-4, 1e4], BISECTION")
    print("    converges faster (its interval always halves), while false")
    print("    position needs far more iterations here.")
    print("(b) False position isn't guaranteed faster in general: its speed")
    print("    depends on curvature -- ln(x) is very flat/curved near the")
    print("    huge upper bound, so the chord from x_u to x_l keeps landing")
    print("    close to x_l instead of the true root, barely shrinking the")
    print("    bracket each step -- slower than bisection's steady halving.")
    print("(c) In the slower method, the a-updates count stays at 0:")
    print("    x_l = 1e-4 is NEVER replaced by false position -- the secant")
    print("    line from the huge x_u to the tiny x_l always lands on the")
    print("    same (right) side of the root, so only x_u ever updates.\n")


if __name__ == "__main__":
    main()
