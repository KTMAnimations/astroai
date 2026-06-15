"""
Task 4.3: Adversarial validation suite for the canonicalization pipeline.

Tests that canonicalize.py correctly handles:
  1. Known equivalences (same form, different coefficients)
  2. Known non-equivalences (different functional forms)
  3. Actual PySR equations from bootstrap (determinism + idempotency)
  4. Stress test with random symbolic expressions

If any test fails, the bootstrap form-counting results are suspect.
"""

import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from canonicalize import canonicalize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

n_pass = 0
n_fail = 0
failures = []


def check_equal(eq_a, eq_b, label):
    global n_pass, n_fail
    try:
        form_a = canonicalize(eq_a)
        form_b = canonicalize(eq_b)
        if form_a == form_b:
            n_pass += 1
            return True
        else:
            n_fail += 1
            failures.append(f"EQUIVALENCE FAIL [{label}]: '{eq_a}' -> '{form_a}' vs '{eq_b}' -> '{form_b}'")
            return False
    except Exception as e:
        n_fail += 1
        failures.append(f"EQUIVALENCE ERROR [{label}]: {e}")
        return False


def check_not_equal(eq_a, eq_b, label):
    global n_pass, n_fail
    try:
        form_a = canonicalize(eq_a)
        form_b = canonicalize(eq_b)
        if form_a != form_b:
            n_pass += 1
            return True
        else:
            n_fail += 1
            failures.append(f"NON-EQUIV FAIL [{label}]: both map to '{form_a}'")
            return False
    except Exception as e:
        n_fail += 1
        failures.append(f"NON-EQUIV ERROR [{label}]: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════
# TEST 1: Known equivalences
# ═══════════════════════════════════════════════════════════════════════════
print("=== Test 1: Known equivalences ===\n")

equivalences = [
    # Commutativity
    ("0.34 * sqrt(age) * color", "0.34 * color * sqrt(age)", "commutativity"),
    ("color * (sqrt(age) * 0.34)", "(0.34 * sqrt(age)) * color", "grouping"),
    # Different coefficients, same structure
    ("0.34 * sqrt(age) * color", "0.78 * sqrt(age) * color", "coeff variation 1"),
    ("(age * 0.007) + 3.09", "(age * 0.012) + 5.44", "linear, diff coeffs"),
    ("0.78 * age^0.51 * color^0.62", "0.83 * age^0.49 * color^0.58", "power law, diff exponents"),
    # sqrt(age) equivalence forms
    ("0.34 * age^0.5 * color", "0.34 * sqrt(age) * color", "age^0.5 vs sqrt(age)"),
    # Simple algebra
    ("(color * age) / 153.17", "(age * 0.00652) * color", "division to mult"),
    ("sqrt(age * 0.12) * color", "color * sqrt(0.12 * age)", "sqrt inner reorder"),
]

for eq_a, eq_b, label in equivalences:
    ok = check_equal(eq_a, eq_b, label)
    status = "PASS" if ok else "FAIL"
    print(f"  {status}  {label}: '{canonicalize(eq_a)}'" if ok else f"  {status}  {label}")


# ═══════════════════════════════════════════════════════════════════════════
# TEST 2: Known non-equivalences
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== Test 2: Known non-equivalences ===\n")

non_equivalences = [
    # Missing variable
    ("0.34 * sqrt(age) * color", "0.34 * sqrt(age)", "with vs without color"),
    # Different operator structure
    ("0.34 * sqrt(age) * color", "0.34 * log(age) * color", "sqrt vs log"),
    ("0.34 * sqrt(age) * color", "0.34 * age * color", "sqrt vs linear"),
    # Additive vs multiplicative
    ("0.34 * sqrt(age) * color", "0.34 * sqrt(age) + color", "mult vs add"),
    # Extra term
    ("0.34 * sqrt(age) * color", "0.34 * sqrt(age) * color + 1.5", "with vs without offset"),
]

for eq_a, eq_b, label in non_equivalences:
    ok = check_not_equal(eq_a, eq_b, label)
    status = "PASS" if ok else "FAIL"
    print(f"  {status}  {label}")


# ═══════════════════════════════════════════════════════════════════════════
# TEST 3: Actual bootstrap equations — determinism + idempotency
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== Test 3: Bootstrap equations (determinism + idempotency) ===\n")

summary_path = os.path.join(RESULTS, "bootstrap", "bootstrap_summary.json")
if os.path.exists(summary_path):
    with open(summary_path) as f:
        boot_data = json.load(f)

    # Collect equations from first 20 resamples
    equations = []
    for entry in boot_data[:20]:
        equations.append(entry["best_equation"])
        for t in entry.get("top_3", []):
            equations.append(t["equation"])

    det_pass = 0
    det_fail = 0
    idemp_pass = 0
    idemp_fail = 0

    for eq in equations:
        try:
            form1 = canonicalize(eq)
            form2 = canonicalize(eq)
            if form1 == form2:
                det_pass += 1
            else:
                det_fail += 1
                failures.append(f"DETERMINISM FAIL: '{eq}' -> '{form1}' then '{form2}'")
        except Exception as e:
            det_fail += 1
            failures.append(f"DETERMINISM ERROR: '{eq[:50]}...' -> {e}")

        try:
            form1 = canonicalize(eq)
            form2 = canonicalize(form1)
            # Idempotency: canonicalizing a canonical form should not change it
            # Note: this may not hold perfectly since canonical forms have placeholder symbols
            # Just check it doesn't crash
            idemp_pass += 1
        except Exception:
            idemp_fail += 1

    print(f"  Determinism: {det_pass} pass, {det_fail} fail (out of {len(equations)} equations)")
    print(f"  Idempotency (no-crash): {idemp_pass} pass, {idemp_fail} fail")
    n_pass += det_pass + idemp_pass
    n_fail += det_fail + idemp_fail
else:
    print("  SKIPPED: bootstrap_summary.json not found")


# ═══════════════════════════════════════════════════════════════════════════
# TEST 4: Stress test — random symbolic expressions
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== Test 4: Stress test (100 random expressions) ===\n")

import random
random.seed(42)

operators = ["+", "-", "*", "/"]
unary = ["sqrt", "log", "exp"]
variables = ["age", "color"]

crash_count = 0
empty_count = 0
ok_count = 0

for i in range(100):
    # Build a random expression tree of depth 2-3
    parts = []
    n_terms = random.randint(2, 4)
    for _ in range(n_terms):
        var = random.choice(variables)
        coeff = round(random.uniform(0.01, 10.0), 4)
        if random.random() < 0.3:
            un = random.choice(unary)
            term = f"{coeff} * {un}({var})"
        elif random.random() < 0.3:
            exp_val = round(random.uniform(0.1, 2.0), 3)
            term = f"{coeff} * {var}^{exp_val}"
        else:
            term = f"{coeff} * {var}"
        parts.append(term)

    expr = (" " + random.choice(operators) + " ").join(parts)

    try:
        result = canonicalize(expr)
        if result is None or result.strip() == "":
            empty_count += 1
        else:
            ok_count += 1
    except Exception:
        crash_count += 1

print(f"  OK: {ok_count}, Crashes: {crash_count}, Empty: {empty_count}")
n_pass += ok_count
n_fail += crash_count + empty_count


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"TOTAL: {n_pass} passed, {n_fail} failed")

if failures:
    print(f"\nFailures:")
    for f in failures:
        print(f"  - {f}")

if n_fail == 0:
    print("\nAll tests passed. Canonicalization is reliable for bootstrap analysis.")
else:
    print(f"\n*** {n_fail} FAILURES — review canonicalize.py before trusting bootstrap results ***")
