"""
CSE 401: Middle-Square Random Number Generator (Simple Implementation)
Tasks 1 to 4 from Section C1 Online-3.
Also includes the Middle-Square Weyl Sequence (MSWS) repair (Section B2).
"""

from scipy.stats import chisquare

def middle_square(seed, n, d=4):
    """
    d-digit Middle-Square generator.
      1. Square the current value.
      2. Pad the square to 2*d digits with leading zeros.
      3. Extract the middle d digits -> next value.
      4. Scale to [0, 1) by dividing by 10^d.
    d=4 reproduces the classic 4-digit method (8-digit square, middle [2:6]).
    """
    results = []
    x = seed
    start = d // 2
    modulus = 10 ** d
    for _ in range(n):
        sq_str = str(x * x).zfill(2 * d)
        x = int(sq_str[start:start + d])
        results.append(x / modulus)
    return results

def find_cycle(seed, d=4):
    """Returns (first_repeated_value, iteration_at_detection, cycle_length) for d digits."""
    seen = {}
    x = seed
    iteration = 0
    start = d // 2
    while x not in seen:
        seen[x] = iteration
        sq_str = str(x * x).zfill(2 * d)
        x = int(sq_str[start:start + d])
        iteration += 1
    return x, iteration, iteration - seen[x]

def msws_generator(n, x0=0, w0=0, c=0xb5ad4ec5, m=2**32):
    """
    Middle-Square Weyl Sequence (MSWS), 32-bit (Section B2).
    Keeps TWO running states: x (middle-square) and w (Weyl). Each step:
        w = (w + c) mod m           # advance the Weyl state (c must be odd)
        x = (x*x + w) mod m         # square-and-add
        U = (x // 2^16) / 2^16      # keep the high 16 bits, drop the low 16
    Returns a list of U in [0, 1). The odd Weyl increment gives a long period
    and prevents the zero-collapse that plain middle-square suffers from,
    because the state is now the PAIR (x, w) -- w can never get stuck.
    (Integer '// 2**16' is used instead of the bit-shift '>> 16'; same result.)
    """
    results = []
    x, w = x0, w0
    for _ in range(n):
        w = (w + c) % m
        x = (x * x + w) % m
        u = (x // 2**16) / 2**16
        results.append(u)
    return results

def perform_chi_square_test(numbers, k=10, alpha=0.05):
    observed = [0] * k
    for r in numbers:
        bin_idx = min(int(r * k), k - 1)
        observed[bin_idx] += 1
    chi2_stat, p_value = chisquare(observed)
    decision = "Reject H0" if p_value < alpha else "Do not reject H0"
    return chi2_stat, p_value, decision, observed

if __name__ == "__main__":
    # ----- TASK 1: Generate values from a seed -----
    task1_seed = 5731
    task1_n = 100
    task1_preview = 10
    print(f"TASK 1: Generate {task1_n} values from seed {task1_seed}:")
    sample = middle_square(task1_seed, task1_n)
    print(f"First {task1_preview} values:", [round(v, 4) for v in sample[:task1_preview]])

    # ----- TASK 2: Edge case / cycle detection -----
    task2_seed = 3792
    print(f"\nTASK 2: Edge Case with Seed {task2_seed}:")
    val, it, cycle = find_cycle(task2_seed)
    print(f"Seed: {task2_seed} | First Repeated: {val} | Iteration: {it} | Cycle: {cycle}")

    # ----- TASK 3: Chi-Square test over several seeds -----
    task3_n = 1000
    task3_seeds = (5731, 6239, 3792)
    print(f"\nTASK 3: Chi-Square Test (N = {task3_n}):")
    for s in task3_seeds:
        chi2, p_val, dec, _ = perform_chi_square_test(middle_square(s, task3_n))
        print(f"Seed {s:4d} -> Chi^2: {chi2:8.2f} | p-value: {p_val:10.4e} | Decision: {dec}")

    # ----- MSWS (Section B2): the Weyl-repaired middle square -----
    msws_n = 1000
    print(f"\nMSWS (Middle-Square Weyl, N = {msws_n}): passes the same Chi-Square test")
    msws_sample = msws_generator(msws_n, x0=0, w0=0)
    print(f"First 5 values: {[round(v, 4) for v in msws_sample[:5]]}")
    chi2, p_val, dec, _ = perform_chi_square_test(msws_sample)
    print(f"Chi^2: {chi2:8.2f} | p-value: {p_val:10.4e} | Decision: {dec}")
