import random

# Simple valence table
VALENCE_TABLE = {
    "Li": 1,
    "Na": 1,
    "Mg": 2,
    "Ca": 2,
    "Al": 3,
    "Fe": 2,
    "C": -4,
    "O": -2,
    "N": -3,
    "Si": 4
}

# Split into positive and negative valence groups
POSITIVE_ELEMENTS = [e for e, v in VALENCE_TABLE.items() if v > 0]
NEGATIVE_ELEMENTS = [e for e, v in VALENCE_TABLE.items() if v < 0]


def generate_valid_compound():
    """
    Generate a chemically valid binary compound
    using valence balance.
    """

    # Pick one positive and one negative element
    pos_elem = random.choice(POSITIVE_ELEMENTS)
    neg_elem = random.choice(NEGATIVE_ELEMENTS)

    pos_val = VALENCE_TABLE[pos_elem]
    neg_val = abs(VALENCE_TABLE[neg_elem])

    # Balance charges
    ratio_pos = neg_val
    ratio_neg = pos_val

    # Simplify ratio
    from math import gcd
    g = gcd(ratio_pos, ratio_neg)
    ratio_pos //= g
    ratio_neg //= g

    # Build formula
    formula = f"{pos_elem}{ratio_pos if ratio_pos>1 else ''}" \
              f"{neg_elem}{ratio_neg if ratio_neg>1 else ''}"

    return formula


if __name__ == "__main__":
    print("Generated compounds:")
    for _ in range(5):
        print(generate_valid_compound())
