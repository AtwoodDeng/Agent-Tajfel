"""Tajfel allocation matrices: the original 1971 matrix, AB-symmetric MIP/MJP/P matrices,
and loss-domain (loss-aversion) variants.

Notation: every matrix is a list of ROWS; a row is the tuple ``(A_points, B_points)`` for the
two anonymous recipient columns. Rows are stored canonically as (A, B); each agent is shown the
matrix from its own point of view (own group on the left), exactly as ``tajfel/model.py`` does.

Strategy keys
  P    parity / fairness      — joint profit constant across rows, only the gap varies
  MJP  maximum joint profit   — joint profit peaks at one row; a larger gap costs joint
  MIP  maximum ingroup profit — own column maximised, other column pinned

Valence keys
  gain   all payoffs positive (the classic allocation-of-money frame)
  mixed  zero-straddling: favouring the ingroup moves it into gain while pushing the outgroup
         into loss (the "9/-10" style); the fair row is (0, 0)
  loss   pure negation of the gain matrix: every recipient starts at a loss and the choice is
         who loses less

AB-symmetry: ``rows[i] == (rows[-1-i][1], rows[-1-i][0])`` for all i, i.e. swapping the two
columns yields the reversed matrix. This guarantees neither column is structurally privileged,
which is what the label-hidden (c2) arm and any cross-group comparison need.

Run ``python -m tajfel.matrices`` to print every matrix as a table with gap / joint columns.
"""

from __future__ import annotations

# --- the original paper's matrix (recorded verbatim, 13 columns) --------------------------------
# Tajfel, H., Billig, M. G., Bundy, R. P., & Flament, C. (1971). The two rows are
# (ingroup member, outgroup member); participants chose ONE column.
#   ingroup:  19 18 17 16 15 14 13 12 11 10  9  8  7
#   outgroup:  1  3  5  7  9 11 13 15 17 19 21 23 25
# MIP and MD (maximum difference) coincide on the left; MJP on the right.
MATRIX_ORIGINAL_1971: list[tuple[int, int]] = [
    (19, 1), (18, 3), (17, 5), (16, 7), (15, 9), (14, 11), (13, 13),
    (12, 15), (11, 17), (10, 19), (9, 21), (8, 23), (7, 25),
]

# --- AB-symmetric positive ("gain") matrices ----------------------------------------------------
# All three have 11 rows (a true centre for mirror symmetry), centre row (10, 10), and both
# columns spanning the same range, so neither column is privileged.

# P — parity / fairness: joint profit constant (20), gap varies symmetrically around 0.
MATRIX_PARITY_SYM: list[tuple[int, int]] = [
    (19, 1), (17, 3), (15, 5), (13, 7), (11, 9), (10, 10),
    (9, 11), (7, 13), (5, 15), (3, 17), (1, 19),
]

# MJP — maximum joint profit: joint profit peaks at the centre row (20) and falls toward both
# ends while the gap grows. The symmetric counterpart of the classic "MJP vs MD" matrix.
MATRIX_MJP_SYM: list[tuple[int, int]] = [
    (15, 0), (14, 2), (13, 4), (12, 6), (11, 8), (10, 10),
    (8, 11), (6, 12), (4, 13), (2, 14), (0, 15),
]

# MIP — maximum ingroup profit: own column maximised (19) while the other column is pinned at 10,
# so ingroup profit is the only thing that varies. Symmetric: group A reads it left-to-right,
# group B right-to-left.
MATRIX_MIP_SYM: list[tuple[int, int]] = [
    (19, 10), (17, 10), (15, 10), (13, 10), (11, 10), (10, 10),
    (10, 11), (10, 13), (10, 15), (10, 17), (10, 19),
]


def _shift(rows: list[tuple[int, int]], k: int) -> list[tuple[int, int]]:
    return [(a + k, b + k) for a, b in rows]


def _negate(rows: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [(-a, -b) for a, b in rows]


# --- loss-domain variants -----------------------------------------------------------------------
# "mixed" = shifted so the fair row is (0, 0): favouring the ingroup means the ingroup gains and
# the outgroup loses — the loss-aversion trade-off. "loss" = pure negation: all payoffs are
# losses and the choice is who loses less.

MATRIX_ORIGINAL_1971_LOSS: list[tuple[int, int]] = _shift(MATRIX_ORIGINAL_1971, -13)   # centre -> (0,0)
MATRIX_ORIGINAL_1971_NEG: list[tuple[int, int]] = _negate(MATRIX_ORIGINAL_1971)

MATRIX_PARITY_LOSS: list[tuple[int, int]] = _shift(MATRIX_PARITY_SYM, -10)             # centre -> (0,0)
MATRIX_PARITY_NEG: list[tuple[int, int]] = _negate(MATRIX_PARITY_SYM)

MATRIX_MJP_LOSS: list[tuple[int, int]] = _shift(MATRIX_MJP_SYM, -10)                   # centre -> (0,0)
MATRIX_MJP_NEG: list[tuple[int, int]] = _negate(MATRIX_MJP_SYM)

MATRIX_MIP_LOSS: list[tuple[int, int]] = _shift(MATRIX_MIP_SYM, -10)                   # centre -> (0,0)
MATRIX_MIP_NEG: list[tuple[int, int]] = _negate(MATRIX_MIP_SYM)

# --- helpers ------------------------------------------------------------------------------------


def gap(row: tuple[int, int]) -> int:
    """A - B points for one row."""
    return int(row[0]) - int(row[1])


def joint(row: tuple[int, int]) -> int:
    """A + B points for one row."""
    return int(row[0]) + int(row[1])


def is_ab_symmetric(rows: list[tuple[int, int]]) -> bool:
    """True iff swapping the two columns yields the reversed matrix (no column privileged)."""
    n = len(rows)
    return all(rows[i] == (rows[n - 1 - i][1], rows[n - 1 - i][0]) for i in range(n))


def rows_of(name: str) -> list[tuple[int, int]]:
    try:
        return MATRIX_LIBRARY[name]["rows"]
    except KeyError:
        raise KeyError(f"unknown matrix {name!r}; expected one of {sorted(MATRIX_LIBRARY)}")


def describe(name: str) -> dict:
    entry = MATRIX_LIBRARY[name]
    rows = entry["rows"]
    return {
        "name": name,
        "strategy": entry["strategy"],
        "valence": entry["valence"],
        "n_rows": len(rows),
        "ab_symmetric": is_ab_symmetric(rows),
        "gap_range": (min(gap(r) for r in rows), max(gap(r) for r in rows)),
        "joint_range": (min(joint(r) for r in rows), max(joint(r) for r in rows)),
        "a_range": (min(r[0] for r in rows), max(r[0] for r in rows)),
        "b_range": (min(r[1] for r in rows), max(r[1] for r in rows)),
        "note": entry["note"],
    }


# --- registry -----------------------------------------------------------------------------------

MATRIX_LIBRARY: dict[str, dict] = {
    "original_1971": {
        "rows": MATRIX_ORIGINAL_1971,
        "strategy": "MD/MIP vs MJP",
        "valence": "gain",
        "note": "Tajfel et al. (1971): ingroup 19..7, outgroup 1..25, 13 columns",
    },
    "original_1971_loss": {
        "rows": MATRIX_ORIGINAL_1971_LOSS,
        "strategy": "MD/MIP vs MJP",
        "valence": "mixed",
        "note": "original shifted so the fair column (13,13) becomes (0,0)",
    },
    "original_1971_neg": {
        "rows": MATRIX_ORIGINAL_1971_NEG,
        "strategy": "MD/MIP vs MJP",
        "valence": "loss",
        "note": "original negated (all losses)",
    },
    "parity_sym": {
        "rows": MATRIX_PARITY_SYM,
        "strategy": "P (parity)",
        "valence": "gain",
        "note": "joint profit constant 20, gap symmetric around the fair row (10,10)",
    },
    "parity_loss": {
        "rows": MATRIX_PARITY_LOSS,
        "strategy": "P (parity)",
        "valence": "mixed",
        "note": "parity shifted to straddle zero; fair row (0,0), joint constant 0",
    },
    "parity_neg": {
        "rows": MATRIX_PARITY_NEG,
        "strategy": "P (parity)",
        "valence": "loss",
        "note": "parity negated (all losses)",
    },
    "mjp_sym": {
        "rows": MATRIX_MJP_SYM,
        "strategy": "MJP (maximum joint profit)",
        "valence": "gain",
        "note": "joint peaks 20 at centre, falls to 15 at the extremes while the gap grows",
    },
    "mjp_loss": {
        "rows": MATRIX_MJP_LOSS,
        "strategy": "MJP (maximum joint profit)",
        "valence": "mixed",
        "note": "MJP shifted to straddle zero; fair row (0,0) is the joint maximum",
    },
    "mjp_neg": {
        "rows": MATRIX_MJP_NEG,
        "strategy": "MJP (maximum joint profit)",
        "valence": "loss",
        "note": "MJP negated (all losses)",
    },
    "mip_sym": {
        "rows": MATRIX_MIP_SYM,
        "strategy": "MIP (maximum ingroup profit)",
        "valence": "gain",
        "note": "own column maximised at 19, other column pinned at 10",
    },
    "mip_loss": {
        "rows": MATRIX_MIP_LOSS,
        "strategy": "MIP (maximum ingroup profit)",
        "valence": "mixed",
        "note": "MIP shifted to straddle zero; other column pinned at 0",
    },
    "mip_neg": {
        "rows": MATRIX_MIP_NEG,
        "strategy": "MIP (maximum ingroup profit)",
        "valence": "loss",
        "note": "MIP negated (all losses)",
    },
}


def _fmt_row(i: int, row: tuple[int, int]) -> str:
    return f"  row {i:>2}:  A {row[0]:>4}  B {row[1]:>4}    gap {gap(row):>+4}  joint {joint(row):>+4}"


def print_matrix(name: str) -> None:
    entry = MATRIX_LIBRARY[name]
    print(f"\n=== {name}  ({entry['strategy']}, {entry['valence']}) ===")
    print(f"    {entry['note']}")
    print(f"    ab_symmetric={is_ab_symmetric(entry['rows'])}")
    for i, row in enumerate(entry["rows"]):
        print(_fmt_row(i, row))


def print_all() -> None:
    for name in MATRIX_LIBRARY:
        print_matrix(name)


if __name__ == "__main__":  # pragma: no cover - developer inspection aid
    print_all()
