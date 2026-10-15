"""
CSE 401: Random Variate Generation & Reliability Simulation (Simple Implementation)

This file covers the one pipeline topic that had no home in the other files:
  1. Inverse-Transform Method  -> turn Uniform(0,1) into other distributions.
  2. Series / Parallel rules    -> combine component lifetimes (series=min, parallel=max).
  3. Monte Carlo reliability    -> the B2 stochastic-network survival task.

Do not use Python's random module; a uniform stream is supplied explicitly.
(You may feed in msws_generator(...) from middle_square_simple.py instead of the
Lehmer stream below -- both return lists of uniforms in [0, 1).)
"""

import math


# ==============================================================================
# 0. A uniform source (Lehmer LCG). Swap in any stream you like (e.g. MSWS).
# ==============================================================================
def get_uniforms(n, seed=123457):
    """Generates n uniforms in [0, 1) via the Lehmer LCG (a=16807, m=2^31-1)."""
    x = seed
    a, m = 16807, 2**31 - 1
    out = []
    for _ in range(n):
        x = (a * x) % m
        out.append(x / m)
    return out


# ==============================================================================
# 1. Inverse-Transform Method:  X = F^{-1}(U)
#    (Draw U ~ Uniform(0,1), then map it through the inverse CDF.)
# ==============================================================================
def inverse_uniform(u, a, b):
    # F(x) = (x - a)/(b - a)   ->   x = a + (b - a)*U
    return a + (b - a) * u

def inverse_exponential(u, mean):
    # F(x) = 1 - e^{-x/mean}   ->   x = -mean * ln(1 - U)
    # IMPORTANT: use (1 - U), never ln(U). U can be 0 (ln(0) = -inf), but 1 - U
    # stays in (0, 1], so the log is always safe.
    return -mean * math.log(1.0 - u)

def inverse_weibull(u, shape_k, scale_c):
    # F(x) = 1 - e^{-(x/c)^k}  ->   x = c * (-ln(1 - U))^{1/k}
    return scale_c * (-math.log(1.0 - u)) ** (1.0 / shape_k)

def inverse_triangular(u, a, b, mode_c):
    # Triangular(min=a, max=b, mode=c). Split point is where the CDF hits (c-a)/(b-a).
    split = (mode_c - a) / (b - a)
    if u <= split:
        return a + math.sqrt(u * (b - a) * (mode_c - a))
    return b - math.sqrt((1.0 - u) * (b - a) * (b - mode_c))

def inverse_discrete(u, values, probs):
    # Staircase CDF: return the first value whose cumulative probability >= U.
    cumulative = 0.0
    for v, p in zip(values, probs):
        cumulative += p
        if u <= cumulative:
            return v
    return values[-1]


# ==============================================================================
# 2. Series / Parallel combination rules
# ==============================================================================
def series(*lifetimes):
    """All components needed -> system dies at the FIRST failure -> min()."""
    return min(lifetimes)

def parallel(*lifetimes):
    """Any one component suffices -> system lives until the LAST failure -> max()."""
    return max(lifetimes)


# ==============================================================================
# 3. Reliability simulation of the B2 stochastic network
#
#    Topology A -> B:
#        top path   : component 2 alone
#        bottom path: component 1 in series with (component 3 parallel component 4)
#        overall    : top path  parallel  bottom path
#
#    =>  Y = max( T2 , min( T1 , max(T3, T4) ) )
# ==============================================================================
def b2_system_survival(t1, t2, t3, t4):
    bottom = series(t1, parallel(t3, t4))   # 1  -in series with-  (3 || 4)
    return parallel(t2, bottom)             # top (2)  ||  bottom

def reliability_simulation(N, betas, theta, uniforms):
    """
    Monte Carlo estimate of P(system survives >= theta).
      betas    = (b1, b2, b3, b4): exponential means for the 4 components.
      uniforms = flat list of >= 4N uniforms. Replication k uses the B2
                 stream-assignment scheme:  U(4k), U(4k+1), U(4k+2), U(4k+3).
    """
    survivors = 0
    for k in range(N):
        t1 = inverse_exponential(uniforms[4 * k + 0], betas[0])
        t2 = inverse_exponential(uniforms[4 * k + 1], betas[1])
        t3 = inverse_exponential(uniforms[4 * k + 2], betas[2])
        t4 = inverse_exponential(uniforms[4 * k + 3], betas[3])
        if b2_system_survival(t1, t2, t3, t4) >= theta:
            survivors += 1
    p_hat = survivors / N
    se = math.sqrt(p_hat * (1.0 - p_hat) / N)
    return {
        "p_hat": p_hat,
        "std_error": se,
        "confidence_interval": (p_hat - 1.96 * se, p_hat + 1.96 * se),
    }

def reliability_with_spare(N, betas, theta, spare_component):
    """
    Same network, but a redundant (identical) unit is added in PARALLEL with one
    chosen component -> that component's lifetime becomes max(two draws).
      spare_component: 0-indexed component to protect (0->comp1 ... 3->comp4),
                       or None for the baseline network.
    Used to answer B2 Task 3 (which single spare helps p the most).
    """
    base = get_uniforms(4 * N)
    extra = get_uniforms(N, seed=987654321)   # independent stream for the spare
    survivors = 0
    for k in range(N):
        t = [inverse_exponential(base[4 * k + i], betas[i]) for i in range(4)]
        if spare_component is not None:
            i = spare_component
            t[i] = max(t[i], inverse_exponential(extra[k], betas[i]))
        if b2_system_survival(*t) >= theta:
            survivors += 1
    return survivors / N


# ==============================================================================
# Main: verification & B2 demo
# ==============================================================================
if __name__ == "__main__":
    # ----- 1. Inverse-transform verified by moments -----
    N_mom = 10000
    u = get_uniforms(N_mom)
    print("=" * 70)
    print(f"1. Inverse-Transform Verification by Moments (N = {N_mom})")
    print("=" * 70)

    exp_data = [inverse_exponential(r, 2.0) for r in u]
    m = sum(exp_data) / N_mom
    v = sum((x - m) ** 2 for x in exp_data) / (N_mom - 1)
    print(f"Exponential(mean=2): mean={m:.3f} (true 2.000), var={v:.3f} (true 4.000)")

    uni_data = [inverse_uniform(r, 5, 15) for r in u]
    m = sum(uni_data) / N_mom
    v = sum((x - m) ** 2 for x in uni_data) / (N_mom - 1)
    print(f"Uniform(5,15):       mean={m:.3f} (true 10.000), var={v:.3f} (true {100/12:.3f})")

    # ----- 2. B2 reliability network -----
    print("\n" + "=" * 70)
    print("2. B2 Reliability Network:  Y = max( T2 , min( T1 , max(T3, T4) ) )")
    print("=" * 70)
    N = 10000
    betas = (10, 8, 7, 5)   # b1, b2, b3, b4
    theta = 7
    res = reliability_simulation(N, betas, theta, get_uniforms(4 * N))
    lo, hi = res["confidence_interval"]
    print(f"P(survive >= {theta} days) ~= {res['p_hat']:.4f}   95% CI [{lo:.4f}, {hi:.4f}]")

    # ----- 3. Task 3: which single spare helps most? (let the sim decide) -----
    print("\n" + "=" * 70)
    print("3. Task 3 -- add ONE spare in parallel with a single component:")
    print("=" * 70)
    baseline = reliability_with_spare(N, betas, theta, None)
    print(f"Baseline (no spare):        p = {baseline:.4f}")
    best_i, best_p = None, baseline
    for i in range(4):
        p = reliability_with_spare(N, betas, theta, i)
        print(f"Spare on component {i + 1} (beta={betas[i]:2d}): p = {p:.4f}")
        if p > best_p:
            best_p, best_i = p, i
    print(f"\nBest choice: component {best_i + 1} (p = {best_p:.4f}).")
    print("Structural reason: component 2 IS the entire top A->B path by itself, so")
    print("doubling it strengthens a whole independent route; components 3 & 4 are")
    print("already redundant (parallel), giving a spare there the least benefit.")
