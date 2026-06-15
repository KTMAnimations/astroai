"""
Canonicalization pipeline for PySR equations.

Converts equation strings into canonical functional forms where all numerical
constants are replaced with ordered placeholders (C0, C1, ...). Two equations
with the same operator structure but different coefficients produce the same
canonical form.

Usage:
    from canonicalize import canonicalize
    form = canonicalize("(color * age) / 153.17")
    # Returns: "C0*age*color"
"""

import sympy
from sympy import Symbol, preorder_traversal, simplify, sympify

# Fixed variable symbols
_age = Symbol("age", positive=True)
_color = Symbol("color", positive=True)
_VARS = {"age": _age, "color": _color}


def canonicalize(equation_str: str) -> str:
    """Convert a PySR equation string into a canonical functional form.

    Steps:
    1. Parse with sympy.sympify (handle PySRFunction wrapper if present)
    2. Replace all numerical constants with ordered placeholders C0, C1, C2, ...
       ordered by first appearance in a preorder (depth-first) traversal
    3. Simplify with sympy.simplify()
    4. Return the string representation

    Two equations with the same operator structure but different coefficients
    should return the same canonical form.
    """
    # Clean input
    expr_str = equation_str.strip()
    if "PySRFunction" in expr_str:
        expr_str = expr_str.split("=>", 1)[1].rstrip(")")
    expr_str = expr_str.replace("^", "**")

    # Parse to sympy expression
    expr = sympify(expr_str, locals=_VARS)

    # First simplify to normalize the expression tree
    expr = simplify(expr)

    # Convert Float exponents to Rational so that age**0.5 and sqrt(age)
    # produce the same canonical form. Without this, Float(0.5) gets replaced
    # by a placeholder while S.Half (in sqrt) does not, causing inconsistency.
    from sympy import nsimplify as _nsimplify
    expr = _nsimplify(expr, rational=True, tolerance=1e-6)

    # Collect all numerical atoms (not 0, 1, -1 which are structural)
    # We need to handle numbers that appear as exponents specially
    counter = [0]
    replacement_map = {}

    def _get_placeholder(num):
        """Get or create a placeholder symbol for a number."""
        # Check if we already have a placeholder for this exact number
        for k, v in replacement_map.items():
            if k == num:
                return v
        sym = Symbol(f"C{counter[0]}")
        counter[0] += 1
        replacement_map[num] = sym
        return sym

    # Walk the expression tree in preorder, replacing numbers
    # We need to be careful: sympy's preorder_traversal visits nodes in
    # depth-first order, and we use xreplace for the actual substitution.
    numbers_in_order = []
    for node in preorder_traversal(expr):
        if node.is_Number and node not in (sympy.S.Zero, sympy.S.One, sympy.S.NegativeOne,
                                            sympy.S.Half):
            if node not in [n for n in numbers_in_order]:
                numbers_in_order.append(node)

    # Build substitution dict: each unique number gets a placeholder
    subs = {}
    for num in numbers_in_order:
        subs[num] = _get_placeholder(num)

    # Apply substitutions
    canonical = expr.xreplace(subs)

    # Simplify again after substitution
    canonical = simplify(canonical)

    return str(canonical)


def canonical_hash(equation_str: str) -> str:
    """Return a hashable canonical form string."""
    return canonicalize(equation_str)


if __name__ == "__main__":
    # ── Test on known pairs ──────────────────────────────────────────────
    import os
    import pandas as pd

    print("=== Test: Known pairs that should canonicalize identically ===\n")

    test_pairs = [
        (
            "(color * age) / 153.17",
            "(age * 0.00652) * color",
            "C0 * age * color",
        ),
        (
            "0.78 * age**0.51 * color**0.62",
            "0.83 * age**0.49 * color**0.58",
            "C0 * age**C1 * color**C2",
        ),
        (
            "(age * 0.007) + 3.09",
            "(age * 0.012) + 5.44",
            "C0 * age + C1",
        ),
    ]

    n_pass = 0
    n_fail = 0

    for eq_a, eq_b, expected_desc in test_pairs:
        form_a = canonicalize(eq_a)
        form_b = canonicalize(eq_b)
        match = form_a == form_b

        status = "PASS" if match else "FAIL"
        if match:
            n_pass += 1
        else:
            n_fail += 1

        print(f"  {status}  A: {eq_a}")
        print(f"         B: {eq_b}")
        print(f"         Expected form: {expected_desc}")
        print(f"         canon(A) = {form_a}")
        print(f"         canon(B) = {form_b}")
        print(f"         Match: {match}")
        print()

    print(f"Known pairs: {n_pass} passed, {n_fail} failed\n")

    # ── Apply to all Pareto equations ────────────────────────────────────
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RESULTS = os.path.join(ROOT, "results")

    print("=== Canonicalization of all Pareto equations ===\n")

    all_forms = {}  # canonical_form -> list of (run, cx, equation)

    for label, csv_name in [
        ("C", "pysr_constrained/pareto_constrained.csv"),
        ("U", "pysr_unconstrained/pareto_unconstrained.csv"),
    ]:
        csv_path = os.path.join(RESULTS, csv_name)
        pareto = pd.read_csv(csv_path)

        for _, row in pareto.iterrows():
            cx = int(row["complexity"])
            eq = row["sympy_format"]
            try:
                form = canonicalize(eq)
            except Exception as e:
                form = f"ERROR: {e}"

            key = form
            if key not in all_forms:
                all_forms[key] = []
            all_forms[key].append((label, cx, str(row["equation"])[:60]))

            print(f"  {label} cx={cx:2d}: {form}")

    # ── Group by canonical form ──────────────────────────────────────────
    print(f"\n=== Equations grouped by canonical form ===\n")
    for form, members in sorted(all_forms.items(), key=lambda x: -len(x[1])):
        if len(members) > 1:
            print(f"  Form: {form}")
            for run, cx, eq in members:
                print(f"    {run} cx={cx:2d}: {eq}")
            print()

    print(f"\n=== Summary ===")
    print(f"  Total equations: {sum(len(m) for m in all_forms.values())}")
    print(f"  Unique canonical forms: {len(all_forms)}")
    n_shared = sum(1 for m in all_forms.values() if len(m) > 1)
    print(f"  Forms shared by 2+ equations: {n_shared}")
