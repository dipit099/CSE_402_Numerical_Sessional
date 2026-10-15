"""
CSE 401: Random Number Generation (Simple Implementation)
Linear Congruential Generator (LCG) & Maximum Period Rules
"""

from math import gcd

# ==============================================================================
# 0. Number-theory helpers (needed for the prime-modulus / primitive-root case)
# ==============================================================================
def _is_prime(n):
    """Simple trial-division primality test (fine for m up to 2^31)."""
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def _prime_factors(n):
    """Returns the set of distinct prime factors of n."""
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1 if d == 2 else 2
    if n > 1:
        factors.add(n)
    return factors


def is_primitive_root(a, m):
    """
    True if a is a primitive root modulo a prime m, i.e. the multiplicative
    order of a equals m - 1. Checked via: for every prime factor q of (m-1),
    a^((m-1)/q) mod m != 1.
    """
    if a % m == 0:
        return False
    order = m - 1
    for q in _prime_factors(order):
        if pow(a, order // q, m) == 1:
            return False
    return True


# ==============================================================================
# 1. Linear Congruential Generator (LCG)
# ==============================================================================
def lcg_step(x, a, c, m):
    """
    Computes one step of LCG:
        X_{i+1} = (a * X_i + c) mod m
    """
    return (a * x + c) % m


def generate_lcg_integers(seed, a, c, m, n):
    """
    Generates n random integers X_1, ..., X_n.
    """
    values = []
    x = seed
    for _ in range(n):
        x = lcg_step(x, a, c, m)
        values.append(x)
    return values


def generate_lcg_uniforms(seed, a, c, m, n):
    """
    Generates n uniform random numbers R_i = X_i / m in [0, 1).
    """
    integers = generate_lcg_integers(seed, a, c, m, n)
    return [x / m for x in integers]


# ==============================================================================
# 2. Period Detection
# ==============================================================================
def find_lcg_period(seed, a, c, m):
    """
    Finds the period (cycle length) of an LCG starting from a given seed.
    """
    seen = {}
    x = seed
    step = 0
    while x not in seen:
        seen[x] = step
        x = lcg_step(x, a, c, m)
        step += 1
    
    first_seen_step = seen[x]
    cycle_length = step - first_seen_step
    return cycle_length


# ==============================================================================
# 3. Checking Maximum Period Conditions (From Slides)
# ==============================================================================
def check_max_period_conditions(a, c, m, seed):
    """
    Checks the 3 theoretical cases for maximum period:
      Case 1: m = 2^b and c != 0 (Mixed LCG) -> Max Period = m
              Conditions: gcd(c, m) == 1 and a = 1 + 4k (i.e. (a-1)%4 == 0)
      Case 2: m = 2^b and c == 0 (Multiplicative LCG) -> Max Period = m / 4
              Conditions: seed is odd and a = 3 + 8k or 5 + 8k
      Case 3: m is prime and c == 0 -> Max Period = m - 1
              Conditions: a is a primitive root modulo m
    """
    is_power_of_two = (m > 0) and (m & (m - 1)) == 0

    if is_power_of_two and c != 0:
        cond1 = (gcd(c, m) == 1)
        cond2 = ((a - 1) % 4 == 0)
        satisfied = cond1 and cond2
        return "Case 1: m=2^b, c!=0", m, satisfied, f"gcd(c,m)=1: {cond1}, a=1+4k: {cond2}"

    elif is_power_of_two and c == 0:
        cond1 = (seed % 2 == 1)
        cond2 = (a % 8 == 3 or a % 8 == 5)
        satisfied = cond1 and cond2
        return "Case 2: m=2^b, c=0", m // 4, satisfied, f"X0 is odd: {cond1}, a=3+8k or 5+8k: {cond2}"

    elif _is_prime(m) and c == 0:
        cond1 = (seed % m != 0)
        cond2 = is_primitive_root(a, m)
        satisfied = cond1 and cond2
        return "Case 3: m prime, c=0", m - 1, satisfied, f"seed != 0 (mod m): {cond1}, a is primitive root mod m: {cond2}"

    else:
        return "General Case", None, None, "No standard maximum-period rule matches these parameters"


# ==============================================================================
# Main: Slide Examples Demonstration
# ==============================================================================
if __name__ == "__main__":
    # ----- Example 1: Mixed LCG -----
    seed1 = 27
    a1 = 17
    c1 = 43
    m1 = 100
    n1 = 4
    print("=" * 70)
    print(f"Example 1: Mixed LCG (X0 = {seed1}, a = {a1}, c = {c1}, m = {m1})")
    print("=" * 70)
    x = seed1
    random_num_list = []
    for i in range(1, n1 + 1):
        next_x = lcg_step(x, a1, c1, m1)
        print(f"X{i} = ({a1} * {x} + {c1}) mod {m1} = {next_x:2d}  ->  R{i} = {next_x / m1:.6f}")
        random_num_list = next_x/m1
        x = next_x

    print(f"Measured Period = {find_lcg_period(seed1, a1, c1, m1)}")

    # ----- Example 2: Multiplicative LCG -----
    a2 = 13
    c2 = 0
    m2 = 64
    n2 = 8
    seeds2 = (1, 2, 3, 4)
    check_seed2 = 1
    print("\n" + "=" * 70)
    print(f"Example 2: Multiplicative LCG (X_{{i+1}} = {a2} * X_i mod {m2})")
    print("=" * 70)
    for s in seeds2:
        seq = generate_lcg_integers(s, a=a2, c=c2, m=m2, n=n2)
        p = find_lcg_period(s, a=a2, c=c2, m=m2)
        print(f"Seed {s:2d} -> Period: {p:2d} | Sequence: {seq}")

    case, max_p, ok, details = check_max_period_conditions(a2, c2, m2, seed=check_seed2)
    print(f"\nCondition Check for Seed {check_seed2}: {case} -> Max Period = {max_p} (Satisfied: {ok})")
    print(f"Details: {details}")

    # ----- Example 4: Standard Lehmer Generator -----
    lehmer_seed = 123457
    lehmer_a = 16807
    lehmer_c = 0
    lehmer_m = 2**31 - 1
    lehmer_n = 3
    print("\n" + "=" * 70)
    print(f"Example 4: Standard Lehmer Generator (a = {lehmer_a}, c = {lehmer_c}, m = 2^31 - 1)")
    print("=" * 70)
    lehmer_sample = generate_lcg_integers(seed=lehmer_seed, a=lehmer_a, c=lehmer_c, m=lehmer_m, n=lehmer_n)
    for i, val in enumerate(lehmer_sample, 1):
        print(f"X{i} = {val:10d}  ->  R{i} = {val / lehmer_m:.6f}")
    print(f"Theoretical Maximum Period = m - 1 = {lehmer_m - 1}")

    case, max_p, ok, details = check_max_period_conditions(lehmer_a, lehmer_c, lehmer_m, seed=lehmer_seed)
    print(f"\nCondition Check: {case} -> Max Period = {max_p} (Satisfied: {ok})")
    print(f"Details: {details}")

   