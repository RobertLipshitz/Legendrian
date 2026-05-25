"""
legendrian.py

Python implementation of Legendrian knot and link invariants,
with an object-oriented interface.

Original algorithms from a Mathematica notebook by Paul Melvin (core Z/2),
extended by Kirk Mangels, Alden Walker, Lenny Ng, Josh Sabloff, and Sumana Shrestha.

Classes
-------
GroundRing    Coefficient ring: Z/2, Z[λ], or Z/p for prime p.
Leg           A Legendrian knot or link given as a plat-closure braid.
DGA           The contact-homology DGA of a Leg over a chosen GroundRing.
Augmentation  A single augmentation of a DGA.

Quick start
-----------
    from legendrian import Leg, GroundRing

    k = Leg([2, 2, 2])          # trefoil by braid word
    k = Leg('mK3_1')            # trefoil from atlas
    print(k.tb, k.rot)

    d = k.dga()                 # DGA over Z/2 (default)
    augs = d.augmentations()
    for a in augs:
        print(a.lin_hom)

    d2 = k.dga('Zlambda')       # DGA over Z[λ]
    d2.check_d_squared()

Atlas naming convention
-----------------------
    'K3_1'        unique Legendrian rep of knot 3_1 (Rolfsen table)
    'K3_1.0'      same, with explicit 0-based index
    'mK5_2'       unique rep of mirror of 5_2
    'mK5_2.0'     first of two reps of mirror of 5_2
    'mK5_2.1'     second rep
    'K11n38'      knot 11n38 (Hoste-Thistlethwaite table)
    Mirror prefix 'm', knot prefix 'K', link prefix 'L'.
    Legendrian index is 0-based; omit when unique.

Input format
------------
Leg accepts four input forms.

Braid word — a list of positive integers encoding the plat closure of a positive
braid.  Generator i represents a positive crossing between strands i and i+1
(counting from the top, 1-indexed).  All left cusps share one x-coordinate and
all right cusps share another.

    Leg([2, 2, 2])    standard Legendrian right-handed trefoil

Tangle decomposition — a list of tuples describing a general front diagram as a
left-to-right sequence of elementary moves:

    ('<', h)   left cusp at height h (0-indexed from the top, current strand count)
    ('>', h)   right cusp at height h
    ('X', h)   positive crossing between strands h and h+1

The code converts the tangle to plat form internally using Legendrian Reidemeister
II moves, and stores the original sequence as self.tangle.

    Leg([('<', 0), ('<', 0), ('X', 1), ('X', 0), ('X', 1), ('>', 0), ('>', 0)])

Grid diagram — a pair of permutations (X_perm, O_perm) passed as a 2-tuple of
lists.  X_perm[j] is the row of the X marker in column j; O_perm[j] is the row
of the O marker in column j.  Rows are 0-indexed from the bottom; columns are
0-indexed from the left.  Vertical strands pass over horizontal strands.  The
Legendrian front projection is obtained by rotating the grid 45° counter-
clockwise, which maps horizontal segments to slope +1 and vertical segments to
slope -1.  The algorithm detects left cusps, right cusps, crossings, and kinks
(transitions between horizontal and vertical strands), produces a tangle
sequence, and converts it to plat form via Legendrian Reidemeister moves.

    Leg(([1, 0], [0, 1]))   2×2 grid for the Legendrian unknot (tb = -1)

Atlas name — a string key into the built-in ATLAS dictionary.

    Leg('mK3_1')      mirror trefoil (unique representative)
    Leg('mK5_2.0')    first of two reps of mirror 5_2 (0-indexed)

Dependencies (install via pip):
    matplotlib    -- for Leg.draw()
    numpy         -- for Leg.draw()
    scipy         -- for smooth spline curves (optional; falls back to polyline)

License
-------
This file is part of the Legendrian knot invariants library.
Copyright (C) 2025  Robert Lipshitz and contributors.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 3.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from __future__ import annotations

__version__ = "0.2.1"

import re
from collections import Counter
from functools import cached_property
from itertools import combinations, product
from typing import Any, ClassVar, cast


# ============================================================
# Section 1: GroundRing
# ============================================================

class GroundRing:
    """
    Coefficient ring for a DGA computation.

    Pre-defined class attributes (singletons):
        GroundRing.Z2      -- the field F_2 = Z/2Z
        GroundRing.ZLAMBDA -- the polynomial ring Z[λ]

    Factory for other primes:
        GroundRing.Zn(p)   -- the field F_p = Z/pZ  (p prime)

    The constructor also accepts a string:
        GroundRing('Z2'), GroundRing('Zlambda'), GroundRing('Z3')
    """

    Z2: ClassVar[GroundRing]
    ZLAMBDA: ClassVar[GroundRing]

    def __init__(self, modulus_or_str: Union[int, str]) -> None:
        if isinstance(modulus_or_str, str):
            other = GroundRing.from_str(modulus_or_str)
            self.modulus: int = other.modulus
        else:
            self.modulus = int(modulus_or_str)

    def __repr__(self) -> str:
        if self.modulus == 0:
            return 'GroundRing.ZLAMBDA'
        if self.modulus == 2:
            return 'GroundRing.Z2'
        return f'GroundRing.Zn({self.modulus})'

    def __hash__(self) -> int:
        return hash(self.modulus)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, GroundRing) and self.modulus == other.modulus

    @classmethod
    def Zn(cls, p: int) -> 'GroundRing':
        """Return the prime field Z/p."""
        return cls(p)

    @classmethod
    def from_str(cls, s: str) -> 'GroundRing':
        """Parse 'Z2', 'Zlambda', 'Z3', etc. into a GroundRing."""
        s = s.strip()
        if s in ('Z2', 'z2', 'F2', 'f2'):
            return cls.Z2
        if s.lower() in ('zlambda', 'z[lambda]', 'zlam'):
            return cls.ZLAMBDA
        m = re.match(r'[Zz](\d+)$', s)
        if m:
            return cls(int(m.group(1)))
        raise ValueError(
            f'Unknown ground ring {s!r}. Use "Z2", "Zlambda", "Z3", etc.'
        )


GroundRing.Z2 = GroundRing(2)
GroundRing.ZLAMBDA = GroundRing(0)

DEFAULT_GROUND_RING: GroundRing = GroundRing.Z2


# ============================================================
# Section 2: Private algorithmic subroutines
# ============================================================
# These are low-level building blocks used by multiple class methods.
# They are NOT 1-to-1 duplicates of any class member.

# --- Linear algebra over F_2 (used by DGA.lin_hom and Augmentation) ---

def _rref_f2(mat):
    """Row-reduce mat over F_2. Returns (rref_matrix, pivot_column_indices)."""
    m = [row[:] for row in mat]
    rows = len(m)
    if not rows or not m[0]:
        return m, []
    cols = len(m[0])
    pivot_cols, pivot_row = [], 0
    for col in range(cols):
        found = next((r for r in range(pivot_row, rows) if m[r][col] % 2), -1)
        if found == -1:
            continue
        m[pivot_row], m[found] = m[found], m[pivot_row]
        pivot_cols.append(col)
        for r in range(rows):
            if r != pivot_row and m[r][col] % 2:
                m[r] = [(m[r][j] ^ m[pivot_row][j]) for j in range(cols)]
        pivot_row += 1
    return m, pivot_cols


def _rank_f2(mat) -> int:
    if not mat or not mat[0]:
        return 0
    _, pivots = _rref_f2(mat)
    return len(pivots)


def _rref_zn(mat, n):
    """Row-reduce mat over Z/n (n prime). Returns (rref_matrix, pivot_column_indices)."""
    m = [[x % n for x in row] for row in mat]
    rows = len(m)
    if not rows or not m[0]:
        return m, []
    cols = len(m[0])
    pivot_cols, pivot_row = [], 0
    for col in range(cols):
        found = next((r for r in range(pivot_row, rows) if m[r][col] % n), -1)
        if found == -1:
            continue
        m[pivot_row], m[found] = m[found], m[pivot_row]
        inv = pow(m[pivot_row][col], -1, n)
        m[pivot_row] = [(x * inv) % n for x in m[pivot_row]]
        for r in range(rows):
            if r != pivot_row and m[r][col] % n:
                factor = m[r][col]
                m[r] = [(m[r][j] - factor * m[pivot_row][j]) % n for j in range(cols)]
        pivot_cols.append(col)
        pivot_row += 1
    return m, pivot_cols


def _rank_zn(mat, n) -> int:
    if not mat or not mat[0]:
        return 0
    _, pivots = _rref_zn(mat, n)
    return len(pivots)


# ============================================================
# Section 3: Leg
# ============================================================

class Leg:
    """
    A Legendrian knot or link in the front-projection plat form.

    Construction
    ------------
    Leg([2, 2, 2])      braid word — list of positive integers (plat closure)
    Leg('mK3_1')        atlas name — unique Legendrian representative
    Leg('mK5_2.0')      first of two reps of mirror of 5_2 (0-indexed)
    Leg([('<', 0), ('X', 0), ('>', 0)])
                        tangle decomposition — list of (op, height) tuples where
                        op is '<' (left cusp), '>' (right cusp), or 'X' (crossing);
                        heights are 0-indexed from the top at the current strand count.
                        Converted to plat form internally; stored as self.tangle.
    Leg((X_perm, O_perm))
                        grid diagram — a pair of permutations (as lists of ints).
                        X_perm[j] is the row of the X marker in column j,
                        O_perm[j] is the row of the O marker in column j,
                        with rows 0-indexed from the bottom and columns
                        0-indexed from the left.  Vertical strands go over
                        horizontal strands.  The Legendrian front is obtained
                        by rotating the grid 45° counter-clockwise; the result
                        is converted to a tangle and then to plat form.
                        Stored as self.grid; the intermediate tangle is stored
                        as self.tangle.

    Classical invariants  (computed once, cached as properties)
    -----------------------------------------------------------
    num_components     int
    strand_potentials  List[int]   Maslov potential, one value per braid strand
    grading            List[int]
    tb                 int   (Thurston-Bennequin number)
    rot                int for knots; Tuple[int,...] for links (one per component)
    ruling_invariant(grading_mod=0)  Dict[int, int]  (ruling polynomial in z;
                       works for links when maslov is specified)

    DGA and augmentations
    ---------------------
    dga(ring)           DGA over ring (default: DEFAULT_GROUND_RING = Z/2);
                        cached per ring
    augmentations(...)  shorthand for self.dga(Z/modulus).augmentations(...)
    lin_hom(...)        shorthand for self.dga(Z/modulus).lin_hom(...)
    rulings(grading_mod) cached per grading_mod

    Link support: differential works for links over Z/2 and Z[λ₁,…,λₗ] (one
    variable per component, basepoint at the last right cusp of each component).
    Augmentations and lin_hom work for links over Z/2 when maslov is set and
    grading_mod | 2*rot_c for every component. Z/n links are deferred.

    Visualization
    -------------
    draw(label_generators, method, color)  returns matplotlib Figure
    export_svg(filename)                   writes SVG, returns filename
    """

    def __init__(
        self,
        input: Union[List[int], List[tuple], str, tuple],
        name: Optional[str] = None,
        maslov: Optional[List[int]] = None,
    ) -> None:
        _is_grid = (
            isinstance(input, tuple) and len(input) == 2
            and isinstance(input[0], (list, tuple))
            and isinstance(input[1], (list, tuple))
        )
        _is_tangle = (
            not _is_grid
            and isinstance(input, list) and bool(input) and isinstance(input[0], tuple)
        )
        if _is_grid:
            X_perm, O_perm = list(input[0]), list(input[1])
            Leg._validate_grid(X_perm, O_perm)
            _tangle = Leg._grid_to_tangle(X_perm, O_perm)
            Leg._validate_tangle(_tangle)
            _braid, _ncusps = Leg._tangle_to_braid(_tangle)
            self.grid: Tuple[List[int], List[int]] = (X_perm, O_perm)
            self.tangle: List[tuple] = _tangle
            self.braid: List[int] = _braid
            self.num_cusps: int = _ncusps
            self.name: str = name if name is not None else repr(input)
        elif _is_tangle:
            Leg._validate_tangle(input)
            _braid, _ncusps = Leg._tangle_to_braid(input)
            self.tangle: List[tuple] = list(input)
            self.braid: List[int] = _braid
            self.num_cusps: int = _ncusps
            self.name: str = name if name is not None else repr(self.tangle)
        elif isinstance(input, list):
            self.braid = list(input)
            self.name = name if name is not None else repr(self.braid)
        elif isinstance(input, str):
            m = re.match(r'^(.+)\.(\d+)$', input)
            canon, idx = (m.group(1), int(m.group(2))) if m else (input, None)
            if canon not in ATLAS:
                raise ValueError(
                    f'Unknown knot name {canon!r}. '
                    'Call list(ATLAS) to see available names.'
                )
            entries = ATLAS[canon]
            if idx is None:
                if len(entries) == 1:
                    idx = 0
                else:
                    raise ValueError(
                        f'{canon!r} has {len(entries)} Legendrian representatives; '
                        f'specify an index 0..{len(entries) - 1}, '
                        f'e.g. "{canon}.0" or "{canon}.{len(entries) - 1}"'
                    )
            elif not (0 <= idx < len(entries)):
                raise IndexError(
                    f'Legendrian index {idx} out of range for {canon!r} '
                    f'({len(entries)} representative(s), indexed 0..{len(entries) - 1})'
                )
            self.braid = list(entries[idx])
            self.name = name if name is not None else input
        else:
            raise TypeError(
                f'Expected a braid word (list of int), tangle sequence (list of tuples), '
                f'atlas name (str), or grid diagram (tuple of two lists), '
                f'got {type(input).__name__!r}'
            )
        if not _is_tangle and not _is_grid:
            self.num_cusps = max(self.braid) // 2 + 1 if self.braid else 1
        assert self.num_cusps >= 1, f"num_cusps={self.num_cusps} is invalid"
        self._dga_cache: Dict[GroundRing, DGA] = {}
        self._rulings_cache: Dict[int, List[List[int]]] = {}
        self._maslov_seeds: Optional[List[int]] = maslov

    def __repr__(self) -> str:
        return f'Leg({self.name!r})'

    @staticmethod
    def _validate_grid(X_perm, O_perm) -> None:
        """Raise ValueError if the grid diagram is invalid."""
        n = len(X_perm)
        if len(O_perm) != n:
            raise ValueError('X and O permutations must have the same length')
        if n < 2:
            raise ValueError(f'Grid size must be at least 2, got {n}')
        if sorted(X_perm) != list(range(n)) or sorted(O_perm) != list(range(n)):
            raise ValueError('X and O must each be a permutation of {0, …, n-1}')
        if any(X_perm[j] == O_perm[j] for j in range(n)):
            raise ValueError('X and O markers cannot occupy the same cell')

    @staticmethod
    def _grid_to_tangle(X_perm, O_perm) -> List[tuple]:
        """
        Convert a grid diagram (two permutations, row 0 at bottom) to a tangle sequence.

        X_perm[j] = row of the X marker in column j.
        O_perm[j] = row of the O marker in column j.

        After a 45° CCW rotation, the diagram becomes a Legendrian front whose events
        are read left-to-right by new_x = col − row.  Horizontal grid segments become
        slope +1 strands; vertical segments become slope −1 strands.  Each marker is
        either a left cusp, a right cusp, or a kink (slope change with no new_x extremum).
        Each interior cell whose row lies strictly between the two markers in its column
        and whose column lies within its row's horizontal span is a crossing.
        """
        n = len(X_perm)

        # Inverse permutations: x_col[r] / o_col[r] = column of X/O in row r
        x_col = [0] * n
        o_col = [0] * n
        for j in range(n):
            x_col[X_perm[j]] = j
            o_col[O_perm[j]] = j

        # Collect all events as (new_x, new_z, type, col j, row r)
        # new_x = j - r, new_z = j + r
        events: List[tuple] = []

        for j in range(n):
            for r, is_x in [(X_perm[j], True), (O_perm[j], False)]:
                other_col = o_col[r] if is_x else x_col[r]
                other_row = O_perm[j] if is_x else X_perm[j]
                nx, nz = j - r, j + r

                if other_col > j and other_row < r:
                    events.append((nx, nz, 'LC', j, r))
                elif other_col < j and other_row > r:
                    events.append((nx, nz, 'RC', j, r))
                elif other_col > j:  # other_row > r: V_j → H_r (V ends, H starts)
                    events.append((nx, nz, 'KVH', j, r))
                else:               # other_col < j, other_row < r: H_r → V_j
                    events.append((nx, nz, 'KHV', j, r))

        # Crossings: cell (j, r) where vertical in col j spans row r strictly,
        # horizontal in row r spans col j, and (j, r) is not a marker.
        for j in range(n):
            lo, hi = min(X_perm[j], O_perm[j]), max(X_perm[j], O_perm[j])
            for r in range(lo + 1, hi):
                L_r = min(x_col[r], o_col[r])
                R_r = max(x_col[r], o_col[r])
                if L_r <= j <= R_r:
                    events.append((j - r, j + r, 'X', j, r))

        # Process top-to-bottom within each new_x value (descending new_z)
        events.sort(key=lambda e: (e[0], -e[1]))

        # Active strands: mutable [intercept, slope, label]
        # Horizontal in row r : slope=+1, intercept=2r  → new_z = new_x + 2r
        # Vertical   in col j : slope=−1, intercept=2j  → new_z = −new_x + 2j
        active: List[List] = []

        def nz_of(strand, nx):
            return strand[1] * nx + strand[0]

        def height_above(nx, nz_ref):
            return sum(1 for s in active if nz_of(s, nx) > nz_ref)

        def find(label):
            for s in active:
                if s[2] == label:
                    return s
            raise AssertionError(f'strand {label} missing from active set')

        tangle: List[tuple] = []

        for nx, nz, etype, j, r in events:
            if etype == 'LC':
                h = height_above(nx, nz)
                tangle.append(('<', h))
                # horizontal (above, slope+1) then vertical (below, slope−1)
                active.append([2 * r, +1, ('H', r)])
                active.append([2 * j, -1, ('V', j)])

            elif etype == 'RC':
                h = height_above(nx, nz)
                tangle.append(('>', h))
                active.remove(find(('H', r)))
                active.remove(find(('V', j)))

            elif etype == 'KVH':   # vertical in col j → horizontal in row r
                s = find(('V', j))
                s[0], s[1], s[2] = 2 * r, +1, ('H', r)

            elif etype == 'KHV':   # horizontal in row r → vertical in col j
                s = find(('H', r))
                s[0], s[1], s[2] = 2 * j, -1, ('V', j)

            else:   # crossing
                h = height_above(nx, nz)
                tangle.append(('X', h))

        return tangle

    @staticmethod
    def _validate_tangle(tangle) -> None:
        """Raise ValueError if tangle is not a valid closed Legendrian tangle sequence."""
        count = 0
        for idx, item in enumerate(tangle):
            if not (isinstance(item, tuple) and len(item) == 2):
                raise ValueError(
                    f'Tangle element {idx} must be a 2-tuple (op, h), got {item!r}'
                )
            op, h = item
            if op not in ('<', '>', 'X'):
                raise ValueError(
                    f"Tangle element {idx}: op must be '<', '>', or 'X', got {op!r}"
                )
            if not isinstance(h, int) or h < 0:
                raise ValueError(
                    f'Tangle element {idx}: height must be a non-negative int, got {h!r}'
                )
            if op == '<':
                if h > count:
                    raise ValueError(
                        f'Tangle element {idx}: LC height {h} out of range '
                        f'({count} strands present, valid 0..{count})'
                    )
                count += 2
            elif op == '>':
                if count < 2 or h + 1 >= count:
                    raise ValueError(
                        f'Tangle element {idx}: RC height {h} out of range '
                        f'({count} strands present, need at least {h + 2})'
                    )
                count -= 2
            else:  # 'X'
                if count < 2 or h + 1 >= count:
                    raise ValueError(
                        f'Tangle element {idx}: crossing height {h} out of range '
                        f'({count} strands present, need at least {h + 2})'
                    )
        if count != 0:
            raise ValueError(
                f'Tangle is not closed: {count} strand(s) remain at the end'
            )

    @staticmethod
    def _tangle_to_braid(tangle, debug_dir=None) -> Tuple[List[int], int]:
        """
        Convert a validated tangle sequence to (braid_word_1indexed, num_cusps).
        Applies LR-II commutation rules until plat form, then extracts braid word.

        If debug_dir is a path, saves a tangle picture after each rule application.
        """
        import os
        seq = list(tangle)
        num_cusps = sum(1 for op, _ in seq if op == '<')
        max_iters = max(10000, len(seq) ** 2 * 50)
        step = 0

        def _save_debug(rule_name):
            nonlocal step
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            stub = object.__new__(Leg)
            stub.tangle = seq[:]
            fig = stub.draw(method='tangle', color=False)
            fig.axes[0].set_title(f'Step {step}: {rule_name}\n{seq}', fontsize=7)
            path = os.path.join(debug_dir, f'step_{step:04d}_{rule_name}.png')
            fig.savefig(path, dpi=80, bbox_inches='tight')
            plt.close(fig)
            step += 1

        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            _save_debug('initial')

        for _ in range(max_iters):
            for i in range(len(seq) - 1):
                op_a, h_a = seq[i]
                op_b, h_b = seq[i + 1]

                # Rule A: X(h) · LC(j) — move LC left past X
                if op_a == 'X' and op_b == '<':
                    h, j = h_a, h_b
                    if h <= j - 2:
                        seq[i], seq[i + 1] = ('<', j), ('X', h)
                        rule = 'A_commute'
                    elif h == j - 1:
                        # LR-II: cusp rises by 1, two extra crossings in post-LC config
                        seq[i:i + 2] = [('<', j - 1), ('X', j + 1), ('X', j), ('X', j - 1)]
                        rule = 'A_LR2'
                    else:  # h >= j
                        seq[i], seq[i + 1] = ('<', j), ('X', h + 2)
                        rule = 'A_commute'
                    if debug_dir: _save_debug(rule)
                    break

                # Rule B: RC(j) · X(h) — move RC right past X (h in post-RC config)
                elif op_a == '>' and op_b == 'X':
                    j, h = h_a, h_b
                    if h == j - 1:
                        # LR-II: h+1 == j is a removed position, so strands are
                        # non-adjacent in pre-RC config; RC falls by 1
                        seq[i:i + 2] = [('X', j + 1), ('X', j), ('X', j - 1), ('>', j + 1)]
                        rule = 'B_LR2'
                    elif h < j:
                        seq[i], seq[i + 1] = ('X', h), ('>', j)
                        rule = 'B_commute'
                    else:  # h >= j
                        seq[i], seq[i + 1] = ('X', h + 2), ('>', j)
                        rule = 'B_commute'
                    if debug_dir: _save_debug(rule)
                    break

                # Rule C: RC(k) · LC(j) — move LC left past RC
                elif op_a == '>' and op_b == '<':
                    k, j = h_a, h_b
                    # j < k means LC is above RC (j = k-1 is an edge case, treated as j < k)
                    if j < k:
                        seq[i], seq[i + 1] = ('<', j), ('>', k + 2)
                    else:
                        seq[i], seq[i + 1] = ('<', j + 2), ('>', k)
                    if debug_dir: _save_debug('C')
                    break

                # LC-LC far-commutation: LC(m)·LC(n) with n > m+1 → LC(n-2)·LC(m)
                elif op_a == '<' and op_b == '<' and h_b > h_a + 1:
                    seq[i], seq[i + 1] = ('<', h_b - 2), ('<', h_a)
                    if debug_dir: _save_debug('LC_LC_commute')
                    break

                # Rule D: LC(k) · LC(k+1) — fix nested left cusps (R2 move)
                elif op_a == '<' and op_b == '<' and h_b == h_a + 1:
                    k = h_a
                    seq[i:i + 2] = [('<', k), ('<', k + 2), ('X', k + 1), ('X', k)]
                    if debug_dir: _save_debug('D_R2')
                    break

                # RC-RC far-commutation: RC(n)·RC(m) with n > m+1 → RC(m)·RC(n-2)
                elif op_a == '>' and op_b == '>' and h_a > h_b + 1:
                    seq[i], seq[i + 1] = ('>', h_b), ('>', h_a - 2)
                    if debug_dir: _save_debug('RC_RC_commute')
                    break

                # Rule F: RC(k+1) · RC(k) — fix nested right cusps (R2 move)
                elif op_a == '>' and op_b == '>' and h_a == h_b + 1:
                    k = h_b
                    seq[i:i + 2] = [('X', k), ('X', k + 1), ('>', k), ('>', k)]
                    if debug_dir: _save_debug('F_R2')
                    break

            else:
                break  # no violation found: sequence is in plat form
        else:
            raise RuntimeError(
                f'Tangle-to-braid conversion did not terminate after {max_iters} '
                'iterations — the tangle may be very complex or there is a bug.'
            )

        # Verify the result is valid plat form
        seen_x = False
        seen_rc = False
        for op, h in seq:
            if op == '<':
                if seen_x or seen_rc:
                    raise AssertionError(f'LC after X or RC in plat-form output: {seq}')
                if h % 2 != 0:
                    raise AssertionError(f'LC at odd height {h} in plat-form output: {seq}')
            elif op == 'X':
                if seen_rc:
                    raise AssertionError(f'X after RC in plat-form output: {seq}')
                seen_x = True
            elif op == '>':
                if h % 2 != 0:
                    raise AssertionError(f'RC at odd height {h} in plat-form output: {seq}')
                seen_rc = True

        braid = [h + 1 for op, h in seq if op == 'X']
        return braid, num_cusps

    # --- Classical invariants ---

    @cached_property
    def num_components(self) -> int:
        b = self.braid
        strand_num = 2 * self.num_cusps
        parent = list(range(strand_num + 1))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i in range(1, strand_num, 2):
            union(i, i + 1)
        at_pos = list(range(1, strand_num + 1))
        for gen in b:
            at_pos[gen - 1], at_pos[gen] = at_pos[gen], at_pos[gen - 1]
        for j in range(0, strand_num, 2):
            union(at_pos[j], at_pos[j + 1])
        return len({find(i) for i in range(1, strand_num + 1)})

    @cached_property
    def _comp_and_perm(self) -> Tuple[List[int], List[int]]:
        """
        Returns (comp_of, final_perm) where:
          comp_of[s]    = component index of 0-indexed strand s (ordered by first
                          appearance scanning strands 0 … 2p-1)
          final_perm[i] = strand at 0-indexed braid position i after the full braid
        """
        p = self.num_cusps
        n = 2 * p

        final_perm = list(range(n))
        for gen in self.braid:
            final_perm[gen - 1], final_perm[gen] = final_perm[gen], final_perm[gen - 1]

        parent = list(range(n))

        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(x: int, y: int) -> None:
            px, py = _find(x), _find(y)
            if px != py:
                parent[px] = py

        for j in range(p):
            _union(2 * j, 2 * j + 1)
        for j in range(p):
            _union(final_perm[2 * j], final_perm[2 * j + 1])

        root_to_comp: Dict[int, int] = {}
        comp_of = [0] * n
        for s in range(n):
            r = _find(s)
            if r not in root_to_comp:
                root_to_comp[r] = len(root_to_comp)
            comp_of[s] = root_to_comp[r]

        return comp_of, final_perm

    @cached_property
    def strand_potentials(self) -> List[int]:
        """
        Maslov potential on braid strands (0-indexed, length 2*num_cusps).

        strand_potentials[s] is the Maslov potential of strand s.  Strands are
        numbered 0 … 2p-1.  Left cusp j (j=0…p-1) connects upper strand 2j and
        lower strand 2j+1; right cusp j connects final_perm[2j] (upper) and
        final_perm[2j+1] (lower).  At every cusp: μ[upper] = μ[lower] + 1.

        Seeds (one integer per component, ordered by first appearance from the
        top of the diagram) are taken from the maslov constructor argument; if
        omitted, all seeds default to 0.  The seed for component i is the
        Maslov potential of the lower strand of the topmost left cusp in that
        component.
        """
        p = self.num_cusps
        n = 2 * p
        seeds = self._maslov_seeds if self._maslov_seeds is not None else [0] * self.num_components
        if len(seeds) != self.num_components:
            raise ValueError(
                f'maslov must have length num_components={self.num_components}, '
                f'got {len(seeds)}'
            )

        comp_of, final_perm = self._comp_and_perm

        # Seed each component from the topmost left cusp in that component.
        mu: List[Optional[int]] = [None] * n
        seeded: set = set()
        for j in range(p):
            c = comp_of[2 * j]
            if c not in seeded:
                mu[2 * j + 1] = seeds[c]        # lower strand = seed
                mu[2 * j] = seeds[c] + 1        # upper strand = seed + 1
                seeded.add(c)

        # BFS: propagate through cusp constraints until all strands are assigned.
        changed = True
        while changed:
            changed = False
            for j in range(p):
                # Left cusp j: upper = 2j, lower = 2j+1
                up, lo = 2 * j, 2 * j + 1
                if mu[up] is None and mu[lo] is not None:
                    mu[up] = mu[lo] + 1
                    changed = True
                elif mu[lo] is None and mu[up] is not None:
                    mu[lo] = mu[up] - 1
                    changed = True
                # Right cusp j: upper = final_perm[2j], lower = final_perm[2j+1]
                up, lo = final_perm[2 * j], final_perm[2 * j + 1]
                if mu[up] is None and mu[lo] is not None:
                    mu[up] = mu[lo] + 1
                    changed = True
                elif mu[lo] is None and mu[up] is not None:
                    mu[lo] = mu[up] - 1
                    changed = True

        return mu  # type: ignore[return-value]  # all entries are filled

    @staticmethod
    def _trugrad(b: List[int], i: int) -> int:
        """Maslov grading of crossing i (1-indexed) in braid b."""
        def grading_height(lst, height):
            for index in lst:
                if height == index or height == index + 1:
                    height += 1 if (height + index) % 2 == 0 else -1
            return height

        def cusp(t):
            return [t[0] - 1, 1] if t[0] % 2 == 0 else [t[0] + 1, -1]

        beginlist = list(reversed(b[:i]))
        endlist = b[i:]
        startheight = beginlist[0]
        result = 0
        temp = [startheight, 0]
        while True:
            temp[0] = grading_height(endlist, temp[0])
            temp = cusp(temp)
            result += temp[1]
            temp[0] = grading_height(list(reversed(endlist)), temp[0])
            if temp[0] in (startheight, startheight + 1):
                break
            temp[0] = grading_height(beginlist, temp[0])
            temp = cusp(temp)
            result += temp[1]
            temp[0] = grading_height(list(reversed(beginlist)), temp[0])
            if temp[0] == startheight + 1:
                break
        return result

    @cached_property
    def grading(self) -> List[int]:
        mu = self.strand_potentials
        perm = list(range(2 * self.num_cusps))
        gr = []
        for gen in self.braid:
            s_upper = perm[gen - 1]
            s_lower = perm[gen]
            gr.append(mu[s_upper] - mu[s_lower])
            perm[gen - 1], perm[gen] = perm[gen], perm[gen - 1]
        gr.extend([1] * self.num_cusps)
        return gr

    @cached_property
    def tb(self) -> int:
        return sum(1 if x % 2 == 0 else -1 for x in self.grading)

    @cached_property
    def rot(self) -> Union[int, Tuple[int, ...]]:
        """
        Rotation number.

        For a knot (num_components == 1): returns an int.
        For a link: returns a tuple of ints, one per component, in the same
        order as the maslov seeds (components ordered by first appearance from
        the top of the front diagram).

        Each value is non-negative (the sign depends on an orientation choice
        that is not fixed by the plat-form data).
        """
        if self.num_components == 1:
            b = self.braid
            if not b:
                return 0
            m = max(b)
            h = m + 2 if m % 2 == 0 else m + 1
            backend = list(reversed([h - x for x in b]))
            return abs((self._trugrad(b, 1) + self._trugrad(backend, len(b))) // 2)

        # For each component, the rotation number is half the total "inconsistency"
        # of the Maslov potential at the right cusps.  At right cusp j, the expected
        # potential difference is +1 (μ[upper] = μ[lower] + 1); the actual difference
        # may differ by 2r_c (twice the component's rotation number).
        mu = self.strand_potentials
        comp_of, final_perm = self._comp_and_perm
        totals = [0] * self.num_components
        for j in range(self.num_cusps):
            up = final_perm[2 * j]
            lo = final_perm[2 * j + 1]
            totals[comp_of[up]] += mu[up] - mu[lo] - 1
        return tuple(abs(t // 2) for t in totals)

    def ruling_invariant(self, grading_mod: int = 0) -> Dict[int, int]:
        """Ruling polynomial as a dictionary mapping degree to coefficient. See [CP05], [Fu03]."""
        c = self.num_cusps
        return dict(Counter(len(r) - c + 1 for r in self.rulings(grading_mod=grading_mod)))

    # --- DGA access ---

    def dga(
        self,
        ground_ring: Union[GroundRing, str, None] = None,
    ) -> 'DGA':
        """
        Return the DGA of this Leg over ground_ring.
        Defaults to DEFAULT_GROUND_RING (currently Z/2) when ground_ring is None.
        Results are cached: repeated calls with the same ring return the same object.
        ground_ring may be a GroundRing instance or a string ('Z2', 'Zlambda', 'Z3').
        """
        if ground_ring is None:
            ground_ring = DEFAULT_GROUND_RING
        if isinstance(ground_ring, str):
            ground_ring = GroundRing.from_str(ground_ring)
        if ground_ring not in self._dga_cache:
            self._dga_cache[ground_ring] = DGA(self, ground_ring)
        return self._dga_cache[ground_ring]

    # --- Convenience delegators ---

    def augmentations(
        self,
        grading_mod: int = 0,
        modulus: int = 2,
    ) -> List['Augmentation']:
        """
        All augmentations over Z/modulus.
        Delegates to self.dga(Z/modulus).augmentations(grading_mod).
        grading_mod: 0 = Z-graded, 1 = ungraded, n >= 2 = Z/n-graded.
        """
        ring = GroundRing.Z2 if modulus == 2 else GroundRing.Zn(modulus)
        return self.dga(ring).augmentations(grading_mod=grading_mod)

    @staticmethod
    def _propagate_rulings(crossings_with_indices, sw: set, p: int,
                           verbose: bool = False, label: str = '') -> List:
        """Propagate partial rulings forward through a sequence of crossings."""
        def valid_ruling_q(r, xing, switch):
            if not switch:
                return all(pair[0] < pair[1] for pair in r)
            t = next((pair for pair in r if xing in pair), None)
            bp = next((pair for pair in r if xing + 1 in pair), None)
            if t is None or bp is None:
                return False
            if not all(pair[0] < pair[1] for pair in r):
                return False
            t = sorted(t); bp = sorted(bp)
            t0, t1, b0, b1 = t[0], t[1], bp[0], bp[1]
            return (t0 < t1 < b0 < b1) or (t0 < b0 < b1 < t1) or (b0 < t0 < t1 < b1)

        def update_ruling(ar, xing, switch, i):
            switches, r = ar[0], ar[1]
            if not switch:
                new_r = [[xing + 1 if x == xing else (xing if x == xing + 1 else x)
                          for x in pair] for pair in r]
                newr = [switches, new_r]
            else:
                newr = [switches + [i], r]
            return newr if valid_ruling_q(newr[1], xing, switch) else None

        initial_r = [[2*i+1, 2*i+2] for i in range(p)]
        current = [[[], initial_r]]
        prefix = f"  [{label}] " if label else "  "
        for global_i, xing in crossings_with_indices:
            if global_i in sw:
                candidates = []
                for ar in current:
                    for sw_bool in (False, True):
                        upd = update_ruling(ar, xing, sw_bool, global_i)
                        if upd is not None:
                            candidates.append(upd)
            else:
                candidates = [upd for ar in current
                              for upd in [update_ruling(ar, xing, False, global_i)]
                              if upd is not None]
            seen, deduped = set(), []
            for ar in candidates:
                key = (tuple(ar[0]), tuple(tuple(pair) for pair in ar[1]))
                if key not in seen:
                    seen.add(key)
                    deduped.append(ar)
            current = deduped
            if verbose:
                sw_marker = " [switchable]" if global_i in sw else ""
                print(f"{prefix}crossing {global_i} (σ_{xing}){sw_marker}: "
                      f"{len(current)} candidate(s)")
        return current

    def rulings(
        self,
        grading_mod: int = 0,
        verbose: bool = False,
    ) -> List[List[int]]:
        """
        All graded rulings, computed with meet-in-the-middle. See [CP05], [Fu03].
        Cached per grading_mod (verbose output is not cached).
        grading_mod: 0 = Z-graded, 1 = ungraded, n >= 2 = Z/n-graded.
        Requires grading_mod | 2*rot_c for every component c (grading_mod=0
        requires rot_c = 0); raises ValueError otherwise.
        """
        rots = self.rot if isinstance(self.rot, tuple) else (self.rot,)
        if grading_mod == 0:
            if any(r != 0 for r in rots):
                raise ValueError(
                    f"Z-graded rulings require rot = 0 for each component; got rot={self.rot}"
                )
        elif any(2 * r % grading_mod != 0 for r in rots):
            raise ValueError(
                f"Z/{grading_mod}-graded rulings require grading_mod | 2*rot_c "
                f"for each component; got rot={self.rot}"
            )
        if grading_mod not in self._rulings_cache:
            b = self.braid
            n = len(b)
            cut = n // 2
            gr = self.grading
            if grading_mod == 0:
                sw = set(i + 1 for i, x in enumerate(gr) if x == 0)
            else:
                sw = set(i + 1 for i, x in enumerate(gr) if x % grading_mod == 0)
            p = self.num_cusps

            if verbose:
                print(f"rulings: braid length {n}, cut at {cut} "
                      f"({cut} left / {n - cut} right), "
                      f"{len(sw)} switchable crossing(s)")

            left = self._propagate_rulings(
                [(i + 1, b[i]) for i in range(cut)], sw, p,
                verbose=verbose, label='left')
            right = self._propagate_rulings(
                [(n - i, b[n - 1 - i]) for i in range(n - cut)], sw, p,
                verbose=verbose, label='right')

            def _config_key(r):
                return tuple(tuple(sorted(pair)) for pair in sorted(r))

            left_by_config: Dict = {}
            for switches, r in left:
                left_by_config.setdefault(_config_key(r), []).append(switches)

            right_by_config: Dict = {}
            for switches, r in right:
                right_by_config.setdefault(_config_key(r), []).append(switches)

            if verbose:
                n_matched = sum(1 for k in left_by_config if k in right_by_config)
                print(f"  left:  {len(left)} candidate(s), "
                      f"{len(left_by_config)} distinct config(s)")
                print(f"  right: {len(right)} candidate(s), "
                      f"{len(right_by_config)} distinct config(s)")
                print(f"  matching configs: {n_matched}")

            result = []
            for config, left_sw_lists in left_by_config.items():
                if config in right_by_config:
                    for l_sw in left_sw_lists:
                        for r_sw in right_by_config[config]:
                            result.append(sorted(l_sw + r_sw))

            if verbose:
                print(f"  => {len(result)} ruling(s) total")

            self._rulings_cache[grading_mod] = result
        return self._rulings_cache[grading_mod]

    def format_ruling_invariant(self, grading_mod: int = 0) -> str:
        """Format self.ruling_invariant as a polynomial string in z."""
        d = self.ruling_invariant(grading_mod)
        if not d:
            return "0"
        terms = []
        for power in sorted(d, reverse=True):
            c = d[power]
            terms.append(f"z^{power}" if c == 1 else f"{c}*z^{power}")
        return " + ".join(terms)

    def all_lin_hom(
        self,
        grading_mod: int = 0,
        modulus: int = 2,
        as_str: bool = False,
    ):
        """
        Set of distinct Poincaré-Chekanov polynomials over all augmentations.
        Delegates to self.dga(Z/modulus).all_lin_hom(grading_mod).
        Returns List[Dict[int,int]], or List[str] if as_str=True.
        """
        ring = GroundRing.Z2 if modulus == 2 else GroundRing.Zn(modulus)
        return self.dga(ring).all_lin_hom(grading_mod=grading_mod, as_str=as_str)

    # --- Visualization ---

    def _trace_braid(self) -> List[List[float]]:
        """Strand paths for plotting: each strand is a list of y-values."""
        def next_level(i, bi):
            if i == bi:
                return i + 1
            elif i == bi + 1:
                return i - 1
            return i

        b = self.braid
        strand_num = 2 * (max(b) // 2 + 1) if b else 2 * self.num_cusps
        strands = []
        for i in range(1, strand_num + 1):
            start = -i - 0.5 if i % 2 == 1 else -i + 0.5
            strands.append([start, -i])
        for bi in b:
            strands = [s + [-next_level(-int(s[-1]), bi)] for s in strands]
        result = []
        for s in strands:
            last = int(s[-1])
            result.append(s + [last - 0.5 if abs(last) % 2 == 1 else last + 0.5])
        return result

    def _trace_tangle(self) -> List[List[Tuple[float, float, bool]]]:
        """
        (x, y, cusp_tip) waypoints for each strand arc.
        cusp_tip=True marks a left/right cusp extremum, where the tangent is
        vertical; cusp_tip=False marks a regular waypoint with a horizontal tangent.
        """
        segments: List[List[Tuple[float, float, bool]]] = []
        active: List[int] = []   # active[height] = segment index
        x = 0.0
        W = 2.0

        for op, h in self.tangle:
            x_L, x_R = x, x + W

            if op == '<':
                for i, seg_idx in enumerate(active):
                    if segments[seg_idx][-1][0] < x_L:
                        segments[seg_idx].append((x_L, -float(i), False))
                    y_new = float(i + 2) if i >= h else float(i)
                    segments[seg_idx].append((x_R, -y_new, False))
                cusp_x = x_L + W * 0.5
                seg_top = len(segments)
                segments.append([(cusp_x, -(h + 0.5), True), (x_R, -float(h), False)])
                seg_bot = len(segments)
                segments.append([(cusp_x, -(h + 0.5), True), (x_R, -float(h + 1), False)])
                active = active[:h] + [seg_top, seg_bot] + active[h:]

            elif op == '>':
                cusp_x = x_R - W * 0.5
                for i, seg_idx in enumerate(active):
                    if segments[seg_idx][-1][0] < x_L:
                        segments[seg_idx].append((x_L, -float(i), False))
                    if i in (h, h + 1):
                        segments[seg_idx].append((cusp_x, -(h + 0.5), True))
                    else:
                        y_new = float(i - 2) if i > h + 1 else float(i)
                        segments[seg_idx].append((x_R, -y_new, False))
                active = active[:h] + active[h + 2:]

            elif op == 'X':
                for i, seg_idx in enumerate(active):
                    if segments[seg_idx][-1][0] < x_L:
                        segments[seg_idx].append((x_L, -float(i), False))
                active[h], active[h + 1] = active[h + 1], active[h]
                for i, seg_idx in enumerate(active):
                    segments[seg_idx].append((x_R, -float(i), False))

            x = x_R

        return segments

    def debug_tangle_to_braid(self, debug_dir: str) -> None:
        """Save a picture after each step of the tangle→plat conversion to debug_dir."""
        if not hasattr(self, 'tangle'):
            raise AttributeError('No tangle stored on this Leg.')
        Leg._tangle_to_braid(self.tangle, debug_dir=debug_dir)

    @staticmethod
    def _trace_grid_components(X_perm, O_perm):
        """Return list of (segs_h, segs_v) per link component for a grid diagram."""
        n = len(X_perm)
        x_row_to_col = [None] * n
        for j in range(n):
            x_row_to_col[X_perm[j]] = j
        visited = [False] * n
        components = []
        for start in range(n):
            if visited[start]:
                continue
            segs_h, segs_v = [], []
            col = start
            while not visited[col]:
                visited[col] = True
                xr, or_ = X_perm[col], O_perm[col]
                segs_v.append((col, min(xr, or_), max(xr, or_)))
                next_col = x_row_to_col[or_]
                segs_h.append((or_, min(col, next_col), max(col, next_col)))
                col = next_col
            components.append((segs_h, segs_v))
        return components

    def draw(self, label_generators: bool = False,
             method: str = 'plat', color: bool = False,
             use_tangle: bool = False):
        """
        Plot the front projection of this Leg. Returns a matplotlib Figure.

        Parameters
        ----------
        label_generators : bool
            Label DGA generators (plat method only).
        method : {'plat', 'tangle', 'grid'}
            'plat'   — braid-plat diagram (default). Always available.
            'tangle' — tangle-sequence diagram. Requires self.tangle
                       (initialize from a tangle sequence or grid).
            'grid'   — XO grid diagram. Requires self.grid
                       (initialize from a grid diagram).
        color : bool
            If True, each strand/component is drawn in a distinct color.
            If False (default), everything is drawn in black.
        use_tangle : bool
            Deprecated. Use method='tangle' instead.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        # Backwards-compatibility shim
        if use_tangle:
            method = 'tangle'

        _cmap = plt.get_cmap('tab10')
        _tab_colors = [_cmap(i) for i in range(10)]

        def _strand_color(k):
            return _tab_colors[k % 10] if color else 'black'

        # ── Grid diagram ──────────────────────────────────────────────────────
        if method == 'grid':
            if not hasattr(self, 'grid'):
                raise AttributeError(
                    'This Leg has no stored grid diagram; '
                    "initialize from a grid diagram to use method='grid'"
                )
            X_perm, O_perm = self.grid
            n = len(X_perm)
            fig, ax = plt.subplots(figsize=(max(3, n * 0.55), max(3, n * 0.55)))
            for i in range(n + 1):
                ax.axhline(i, color='lightgray', lw=0.4, zorder=0)
                ax.axvline(i, color='lightgray', lw=0.4, zorder=0)
            for ci, (segs_h, segs_v) in enumerate(
                    Leg._trace_grid_components(X_perm, O_perm)):
                c = _strand_color(ci)
                for col, r0, r1 in segs_v:
                    ax.plot([col + .5] * 2, [r0 + .5, r1 + .5],
                            color=c, lw=2, zorder=1)
                for row, c0, c1 in segs_h:
                    ax.plot([c0 + .5, c1 + .5], [row + .5] * 2,
                            color=c, lw=2, zorder=2)
            mk = 'black'
            for j in range(n):
                ax.plot(j + .5, X_perm[j] + .5, 'x',
                        color=mk, ms=5, mew=1.5, zorder=3)
                ax.plot(j + .5, O_perm[j] + .5, 'o',
                        color=mk, ms=4, mew=1.2, fillstyle='none', zorder=3)
            ax.set_xlim(0, n)
            ax.set_ylim(0, n)
            ax.set_aspect('equal')
            name_str = getattr(self, 'name', '') or ''
            ax.set_title(f'Grid diagram  {name_str}'.strip(), fontsize=9)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            plt.tight_layout()
            return fig

        # ── Tangle diagram ────────────────────────────────────────────────────
        if method == 'tangle':
            if not hasattr(self, 'tangle'):
                raise AttributeError(
                    'This Leg has no stored tangle; '
                    "initialize from a tangle sequence or grid to use method='tangle'"
                )
            segments = self._trace_tangle()
            all_x = [pt[0] for seg in segments for pt in seg]
            all_y = [pt[1] for seg in segments for pt in seg]
            fig, ax = plt.subplots(figsize=(max(4, max(all_x) * 0.5), 3))
            t = np.linspace(0, 1, 50)
            for k, seg in enumerate(segments):
                px, py = [], []
                for j in range(len(seg) - 1):
                    x0, y0, tip0 = seg[j]
                    x3, y3, tip3 = seg[j + 1]
                    dx = x3 - x0
                    # horizontal tangent at regular waypoints; zero velocity at cusp tips
                    c1 = (x0, y0) if tip0 else (x0 + dx / 3, y0)
                    c2 = (x3, y3) if tip3 else (x3 - dx / 3, y3)
                    bx = ((1-t)**3 * x0 + 3*(1-t)**2*t * c1[0]
                          + 3*(1-t)*t**2 * c2[0] + t**3 * x3)
                    by = ((1-t)**3 * y0 + 3*(1-t)**2*t * c1[1]
                          + 3*(1-t)*t**2 * c2[1] + t**3 * y3)
                    if px:
                        px.extend(bx[1:].tolist())
                        py.extend(by[1:].tolist())
                    else:
                        px.extend(bx.tolist())
                        py.extend(by.tolist())
                ax.plot(px, py, color=_strand_color(k), linewidth=2,
                        solid_capstyle='round', solid_joinstyle='round')
            ax.yaxis.set_visible(False)
            ax.xaxis.set_visible(False)
            ax.set_title(f'Legendrian Knot  {self.name}')
            plt.tight_layout()
            return fig

        # ── Plat diagram (default) ────────────────────────────────────────────
        b = self.braid
        strands = self._trace_braid()
        extra_w = 0.8 if label_generators else 0.0
        fig, ax = plt.subplots(figsize=(max(4, len(b) * 0.8) + extra_w, 3))
        for k, strand in enumerate(strands):
            xs = np.array(range(1, len(strand) + 1), dtype=float)
            ys = np.array(strand, dtype=float)
            t_fine = np.linspace(xs[0], xs[-1], 300)
            try:
                from scipy.interpolate import make_interp_spline
                spl = make_interp_spline(xs, ys, k=min(2, len(xs) - 1))
                ax.plot(t_fine, spl(t_fine), color=_strand_color(k), linewidth=2)
            except Exception:
                ax.plot(xs, ys, color=_strand_color(k), linewidth=2)
        if label_generators:
            _bbox = dict(boxstyle='round,pad=0.15', facecolor='white',
                         edgecolor='gray', alpha=0.85)
            for i in range(1, len(b) + 1):
                ax.text(i + 1.5, -(b[i - 1] + 0.5), str(i),
                        ha='center', va='center', fontsize=7, zorder=5, bbox=_bbox)
            for k in range(1, self.num_cusps + 1):
                ax.text(len(b) + 2.8, -(2 * k - 0.5), str(len(b) + k),
                        ha='left', va='center', fontsize=7, zorder=5, bbox=_bbox)
        ax.set_xticks(range(1, len(b) + 3))
        ax.set_xticklabels([str(i) for i in range(len(b) + 2)])
        ax.yaxis.set_visible(False)
        ax.set_title(f"Legendrian Knot  braid = {b}")
        ax.grid(axis='x', linestyle='--', alpha=0.4)
        plt.tight_layout()
        return fig

    def export_svg(
        self,
        filename: str,
        xstep: float = 80.0,
        ystep: float = 40.0,
        margin: float = 40.0,
        stroke_width: float = 2.0,
    ) -> str:
        """
        Export the front projection as an SVG file. Returns the filename written.

        Each strand becomes a cubic Bezier spline with Catmull-Rom tangents for
        interior segments and a strictly horizontal tangent at each cusp tip,
        producing the classic C-shaped cusp geometry.
        """
        strands = self._trace_braid()
        all_y = [y for s in strands for y in s]
        y_top = max(all_y)
        y_bot = min(all_y)

        def px(x_idx: float) -> float:
            return margin + (x_idx - 1) * xstep

        def py(y_val: float) -> float:
            return margin + (y_top - y_val) * ystep

        n_pts = len(strands[0])
        svg_w = 2 * margin + (n_pts - 1) * xstep
        svg_h = 2 * margin + (y_top - y_bot) * ystep

        palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                   '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        def _path(pts: List[Tuple[float, float]]) -> str:
            N = len(pts)
            g0 = (pts[0][0] - xstep, pts[0][1])
            gn = (pts[-1][0] + xstep, pts[-1][1])
            apts = [g0] + pts + [gn]
            d = f'M {pts[0][0]:.2f},{pts[0][1]:.2f}'
            for j in range(N - 1):
                p0, p1, p2, p3 = apts[j], apts[j+1], apts[j+2], apts[j+3]
                cp1 = (p1[0] + (p2[0]-p0[0])/6, p1[1] + (p2[1]-p0[1])/6)
                cp2 = (p2[0] - (p3[0]-p1[0])/6, p2[1] - (p3[1]-p1[1])/6)
                if j == 0:
                    cp1 = pts[0]
                    cp2 = (cp2[0], pts[0][1])
                elif j == N - 2:
                    cp2 = pts[-1]
                    cp1 = (cp1[0], pts[-1][1])
                if j == 1 and N > 3:
                    cp1 = (cp1[0], 2*pts[1][1] - pts[0][1])
                if j == N - 3 and N > 3:
                    cp2 = (cp2[0], 2*pts[-2][1] - pts[-1][1])
                d += (f' C {cp1[0]:.2f},{cp1[1]:.2f}'
                      f' {cp2[0]:.2f},{cp2[1]:.2f}'
                      f' {p2[0]:.2f},{p2[1]:.2f}')
            return d

        lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg"'
            f' width="{svg_w:.0f}" height="{svg_h:.0f}"'
            f' viewBox="0 0 {svg_w:.0f} {svg_h:.0f}">',
            f'  <title>Legendrian knot braid={self.braid}</title>',
        ]
        for k, strand in enumerate(strands):
            pts = [(px(i + 1), py(y)) for i, y in enumerate(strand)]
            d = _path(pts)
            color = palette[k % len(palette)]
            lines.append(
                f'  <path d="{d}" stroke="{color}"'
                f' stroke-width="{stroke_width}" fill="none"'
                f' stroke-linecap="round"/>'
            )
        lines.append('</svg>')
        svg_text = '\n'.join(lines) + '\n'
        with open(filename, 'w') as fh:
            fh.write(svg_text)
        return filename


# ============================================================
# Section 4: DGA
# ============================================================

class DGA:
    """
    The contact-homology DGA of a Legendrian knot over a chosen GroundRing.

    Obtain via Leg.dga(ring) rather than constructing directly.

    Differential  (lazily computed, cached)
    ----------------------------------------
    dga.differential   native format for ring:
      Z2      list of list-of-monomials
      ZLAMBDA list of (word_tuple, kind) → coeff dicts  (kind = 'int' or ('lambda', c))
      Z/p     same as ZLAMBDA format, integer coefficients reduced mod p

    Methods
    -------
    augmentations(grading_mod, lambda_values)  List[Augmentation], cached per (grading_mod, lambda_values)
    lin_hom(grading_mod)        List[Dict[int,int]], cached per grading_mod
    check_d_squared()           bool  (Z/2 and Z[λ] only)
    aug_count(grading_mod)      float  normalized augmentation number (Z/2, Z/p)
    print_differential()        print d(a[i]) for each generator
    """

    def __init__(self, leg: Leg, ring: GroundRing) -> None:
        self.leg = leg
        self.ring = ring
        self._differential = None
        self._augmentations_cache: Dict[tuple, List['Augmentation']] = {}
        self._lin_hom_cache: Dict[int, List[Tuple[Dict[int, int], 'Augmentation']]] = {}

    def __repr__(self) -> str:
        return f'DGA({self.leg!r}, ring={self.ring!r})'

    @cached_property
    def _gens_spaces(self) -> List[List[int]]:
        """Generators grouped by grading, highest first. Used by homology computations."""
        gr = self.leg.grading
        max_g, min_g = max(gr), min(gr)
        return [[i + 1 for i, g in enumerate(gr) if g == k]
                for k in range(max_g, min_g - 1, -1)]

    def _z_diff(self) -> List[Dict]:
        """
        DGA differential over Z[λ^±1].
        Returns a list (one entry per generator) of dicts mapping
          (word_as_tuple, kind) -> integer_coefficient
        where kind is 'int' (plain integer), ('lambda', c) for λ_c, or
        ('lambda_inv', c) for λ_c^{-1}.
        Used by self.differential for both ZLAMBDA and Z/n rings.
        """
        b = self.leg.braid

        def pos(lst):
            return [x for x in lst if x > 0]

        def glue_enhanced(u_path, lb_term):
            l_path, coeff = lb_term
            u_abs = {abs(x) for x in u_path if abs(x) > 0}
            l_abs = {abs(x) for x in l_path if abs(x) > 0}
            if u_abs & l_abs:
                return None
            return (list(reversed(pos(u_path))) + pos(l_path), coeff)

        def glue_lists_enhanced(ub, lbenhanced):
            p = len(ub) - 1
            seen, result = set(), []
            for j in range(1, p + 1):
                for u_path in ub[j]:
                    for lb_term in lbenhanced[j]:
                        g = glue_enhanced(u_path, lb_term)
                        if g is not None:
                            key = (tuple(g[0]), g[1])
                            if key not in seen:
                                seen.add(key)
                                result.append(g)
            return result

        def diff_enhanced():
            p = self.leg.num_cusps
            gr_b = self.leg.grading  # len = len(b) + p; correct for multi-component links
            bext = b + list(range(1, 2 * p, 2))
            n = len(bext)

            u = [[[] for _ in range(2 * p + 1)] for _ in range(n + 1)]
            lo = [[[] for _ in range(2 * p + 1)] for _ in range(n + 1)]
            for j in range(1, p + 1):
                u[1][2 * j] = [[j]]
                lo[1][2 * j - 1] = [[j]]

            for i in range(1, n):
                jj = bext[i - 1]
                for s in range(2 * p + 1):
                    u[i + 1][s] = u[i][s][:]
                    lo[i + 1][s] = lo[i][s][:]
                u[i + 1][jj + 1] = [path + [-i] for path in u[i][jj]]
                u[i + 1][jj] = [list(t) for t in
                                 {tuple(path + [-i]) for path in u[i][jj + 1]} |
                                 {tuple(path + [i]) for path in u[i][jj]}]
                lo[i + 1][jj] = [path + [-i] for path in lo[i][jj + 1]]
                lo[i + 1][jj + 1] = [list(t) for t in
                                      {tuple(path + [-i]) for path in lo[i][jj]} |
                                      {tuple(path + [i]) for path in lo[i][jj + 1]}]

            ub = [[[] for _ in range(p + 1)] for _ in range(n + 1)]
            lb = [[[] for _ in range(p + 1)] for _ in range(n + 1)]
            for i in range(1, n + 1):
                c = bext[i - 1]
                for j in range(1, p + 1):
                    ub[i][j] = [path[1:] for path in u[i][c + 1] if path and path[0] == j]
                    lb[i][j] = [path[1:] for path in lo[i][c] if path and path[0] == j]

            lbe = [[[] for _ in range(p + 1)] for _ in range(n + 1)]
            for i in range(1, n + 1):
                for j in range(1, p + 1):
                    for path in lb[i][j]:
                        sign = 1
                        for x in path:
                            if 0 < x <= len(gr_b):
                                sign *= (-1) ** (gr_b[x - 1] % 2 + 1)
                        lbe[i][j].append((path, sign))

            bd = [None] + [glue_lists_enhanced(ub[i], lbe[i]) for i in range(1, n + 1)]
            comp_of, final_perm = self.leg._comp_and_perm
            mu = self.leg.strand_potentials
            basepoint: Dict[int, int] = {}  # comp_index -> 0-indexed right cusp j
            for j in range(p):
                basepoint[comp_of[final_perm[2 * j]]] = j  # last right cusp per component
            for i in range(n - p + 1, n + 1):
                j = i - (n - p + 1)  # 0-indexed right cusp (1-indexed k = j+1 in _z_diff terms)
                c = comp_of[final_perm[2 * j]]
                if j == basepoint[c]:
                    # Lower strand's Maslov potential determines cusp orientation:
                    # odd → downward cusp → λ_c; even → upward cusp → λ_c^{-1}
                    lower = final_perm[2 * j + 1]
                    coeff = ('lambda', c) if mu[lower] % 2 == 1 else ('lambda_inv', c)
                else:
                    coeff = 1
                entry = bd[i]
                assert entry is not None
                bd[i] = [([], coeff)] + entry
            return bd[1:]

        result = []
        for terms in diff_enhanced():
            poly: Dict = {}
            assert terms is not None
            for word, coeff in terms:
                key_word = tuple(word)
                if isinstance(coeff, tuple):  # ('lambda', c)
                    k = (key_word, coeff)
                    poly[k] = poly.get(k, 0) + 1
                else:
                    k = (key_word, 'int')
                    poly[k] = poly.get(k, 0) + coeff
            result.append({k: v for k, v in poly.items() if v != 0})
        return result

    @property
    def differential(self):
        """
        The DGA differential, computed once and cached. Format depends on self.ring.

        The DGA itself is defined over Z/2 in [Ch02] and extended to Z-coefficients
        with coherent orientations in [ENS02]; the plat-form algorithm used here
        follows [Ng03].

        Links are supported over all rings. For ZLAMBDA and Z/n the returned list of
        dicts uses kind ('lambda', c) or ('lambda_inv', c) (0-indexed component c)
        for the basepoint cusp of each component — the choice depends on the cusp's
        orientation (lower-strand Maslov potential odd → λ_c, even → λ_c^{-1}).
        'int' terms are unchanged. Z/n integer coefficients are reduced mod n;
        λ_c terms remain symbolic for evaluation at augmentation time.
        """
        if self._differential is not None:
            return self._differential

        b = self.leg.braid

        if self.ring == GroundRing.Z2:

            def acd2(last_idx, top, bottom, top_ch, bot_ch):
                if top > bottom:
                    return []
                if last_idx == 1:
                    if top % 2 == 1 and bottom == top + 1:
                        return [[]] if not top_ch and not bot_ch else [top_ch + list(reversed(bot_ch))]
                    return []
                ni = last_idx - 1
                c = b[ni - 1]
                top_dir = 'must' if top == c else 'either' if top == c + 1 else 'stay'
                bot_dir = 'must' if bottom == c + 1 else 'either' if bottom == c else 'stay'
                if top_dir != 'stay' and bot_dir != 'stay':
                    return []
                if top_dir == 'must':
                    return acd2(ni, top + 1, bottom, top_ch, bot_ch)
                if top_dir == 'either':
                    return (acd2(ni, top - 1, bottom, top_ch, bot_ch) +
                            acd2(ni, top, bottom, top_ch + [ni], bot_ch))
                if bot_dir == 'must':
                    return acd2(ni, top, bottom - 1, top_ch, bot_ch)
                if bot_dir == 'either':
                    return (acd2(ni, top, bottom, top_ch, bot_ch + [ni]) +
                            acd2(ni, top, bottom + 1, top_ch, bot_ch))
                return acd2(ni, top, bottom, top_ch, bot_ch)

            crossing_diffs = [acd2(k, b[k-1], b[k-1]+1, [], [])
                              for k in range(1, len(b) + 1)]
            cusp_diffs = []
            for k in range(1, self.leg.num_cusps + 1):
                ts = 2 * k - 1
                cd = acd2(len(b) + 1, ts, ts + 1, [], [])
                cd.append([])
                cusp_diffs.append(cd)
            self._differential = crossing_diffs + cusp_diffs

        elif self.ring == GroundRing.ZLAMBDA:
            self._differential = self._z_diff()

        else:
            # Z/n: lift through Z[λ], reduce integer coefficients mod n, keep λ_c symbolic
            n = self.ring.modulus
            zd = self._z_diff()
            result = []
            for poly_dict in zd:
                reduced: Dict[tuple, int] = {}
                for (word, kind), coeff in poly_dict.items():
                    key = (word, kind)
                    c = coeff % n
                    if c:
                        reduced[key] = (reduced.get(key, 0) + c) % n
                result.append({k: v for k, v in reduced.items() if v})
            self._differential = result

        return self._differential

    def check_d_squared(self) -> bool:
        """
        Verify d² = 0.
        Implemented for Z/2 and Z[λ]; raises NotImplementedError for Z/p.
        """
        if self.ring == GroundRing.Z2:
            d = self.differential
            for d_gen in d:
                count = Counter()
                for monomial in d_gen:
                    for pos, gen in enumerate(monomial):
                        prefix = monomial[:pos]
                        suffix = monomial[pos + 1:]
                        for d_term in d[gen - 1]:
                            count[tuple(prefix + d_term + suffix)] += 1
                if any(v % 2 != 0 for v in count.values()):
                    return False
            return True

        if self.ring == GroundRing.ZLAMBDA:
            d = self.differential
            gr = self.leg.grading
            num_comp = self.leg.num_components

            def to_triples(poly_dict):
                # Represent lambda monomial as a tuple of signed exponents, one per component.
                result = []
                for (word, kind), coeff in poly_dict.items():
                    lam = [0] * num_comp
                    if isinstance(kind, tuple):
                        c = kind[1]
                        lam[c] = 1 if kind[0] == 'lambda' else -1
                    result.append((word, tuple(lam), coeff))
                return result

            d_triples = [to_triples(poly) for poly in d]
            for triples in d_triples:
                count = {}
                for word, lam, coeff in triples:
                    for pos, gen in enumerate(word):
                        prefix = word[:pos]
                        suffix = word[pos + 1:]
                        right_sign = (-1) ** sum(
                            gr[word[j] - 1] for j in range(pos + 1, len(word))
                        )
                        for d_word, d_lam, d_coeff in d_triples[gen - 1]:
                            new_word = prefix + d_word + suffix
                            new_lam = tuple(a + b for a, b in zip(lam, d_lam))
                            key = (new_word, new_lam)
                            count[key] = count.get(key, 0) + coeff * right_sign * d_coeff
                if any(v != 0 for v in count.values()):
                    return False
            return True

        raise NotImplementedError(
            f'check_d_squared not yet implemented for {self.ring}'
        )

    def augmentations(
        self,
        grading_mod: int = 0,
        lambda_values: Optional[Dict[int, int]] = None,
    ) -> List['Augmentation']:
        """
        All augmentations of this DGA over self.ring.
        Cached per (grading_mod, lambda_values).

        grading_mod: 0 = Z-graded, 1 = ungraded, n >= 2 = Z/n-graded.
        Requires grading_mod | 2*rot_c for every component (same condition as rulings).
        Not supported for Z[λ] (use Z/2 or Z/p instead).

        lambda_values: optional dict {component_index: value} fixing the image of each
        lambda_c under the augmentation.  Only used for Z/n (n > 2).
        - Knot default (1 component): lambda_0 -> n-1 (= -1 mod n).
        - Link default: search over all units in (Z/n)^x for each component.
        Override by passing e.g. lambda_values={0: 1} to fix lambda_0 = 1.
        Z/2 ignores lambda_values (only one unit: 1 = -1 mod 2).
        """
        if self.ring == GroundRing.ZLAMBDA:
            raise NotImplementedError(
                'Augmentations are not computed directly over Z[λ]. '
                'Use leg.dga(GroundRing.Z2) or leg.dga(GroundRing.Zn(p)).'
            )
        _rots = self.leg.rot if isinstance(self.leg.rot, tuple) else (self.leg.rot,)
        if grading_mod == 0:
            if any(r != 0 for r in _rots):
                raise ValueError(
                    f"Z-graded augmentations require rot = 0 for each component; "
                    f"got rot={self.leg.rot}"
                )
        elif any(2 * r % grading_mod != 0 for r in _rots):
            raise ValueError(
                f"Z/{grading_mod}-graded augmentations require grading_mod | 2*rot_c "
                f"for each component; got rot={self.leg.rot}"
            )
        modulus = self.ring.modulus
        num_comp = self.leg.num_components
        if modulus > 2:
            if lambda_values is not None:
                lv: Optional[Dict[int, int]] = lambda_values
            elif num_comp == 1:
                lv = {0: modulus - 1}  # knot default: lambda -> -1 mod n
            else:
                lv = None  # links: search over (Z/n)^x for each component
        else:
            lv = None  # Z/2: no lambda tracking needed
        lv_key = frozenset(lv.items()) if lv is not None else None
        cache_key = (grading_mod, lv_key)
        if cache_key not in self._augmentations_cache:
            gr = self.leg.grading
            r = self.leg.rot
            d = self.differential
            modulus = self.ring.modulus

            def rmv(dd, z):
                z_set = set(z)
                return [[m for m in poly if not z_set.intersection(m)] for poly in dd]

            def sub(dd, a):
                a_set = set(a)
                return [[[x for x in m if x not in a_set] for m in poly] for poly in dd]

            def trim(dd):
                def mod2(poly):
                    cnt = Counter(tuple(sorted(m)) for m in poly)
                    return [list(k) for k, v in cnt.items() if v % 2 == 1]
                reduced = [mod2(poly) for poly in dd]
                seen, result = set(), []
                for poly in reduced:
                    if not poly:
                        continue
                    key = frozenset(tuple(m) for m in poly)
                    if key not in seen:
                        seen.add(key)
                        result.append(poly)
                return result

            def ac(dd, grading_mod):
                a1_local: List[int] = []
                g = gr[:]
                if grading_mod == 0 and isinstance(r, int) and r != 0:
                    g = [x % (2 * r) for x in g]

                def is_active(x):
                    return x == 0 if grading_mod == 0 else x % grading_mod == 0

                def is_grade_one(x):
                    if grading_mod == 0: return x == 1
                    if grading_mod == 1: return True
                    return x % grading_mod == 1

                aQ_l = [i + 1 for i, x in enumerate(g) if is_active(x)]
                a0_l = [i + 1 for i in range(len(g)) if i + 1 not in set(aQ_l)]
                grade1_idx = [i for i, x in enumerate(g) if is_grade_one(x)]
                work = trim(rmv([dd[i] for i in grade1_idx], a0_l))

                def is_impossible(w): return any(poly == [[]] for poly in w)
                def get_c0(w): return sorted({poly[0][0] for poly in w if len(poly) == 1 and len(poly[0]) == 1})
                def get_c1(w):
                    res = []
                    for poly in w:
                        if len(poly) == 2 and [] in poly:
                            others = [m for m in poly if m]
                            if len(others) == 1:
                                res.extend(others[0])
                    return sorted(set(res))

                if is_impossible(work):
                    return None, None, None, None
                c0, c1 = get_c0(work), get_c1(work)
                while c0 or c1:
                    if set(c0) & set(c1):
                        return None, None, None, None
                    work = trim(sub(rmv(work, c0), c1))
                    aQ_l = [x for x in aQ_l if x not in c0 and x not in c1]
                    a1_local = sorted(set(a1_local) | set(c1))
                    a0_l = sorted(set(a0_l) | set(c0))
                    if is_impossible(work):
                        return None, None, None, None
                    c0, c1 = get_c0(work), get_c1(work)
                return aQ_l, a1_local, a0_l, work

            def aug_q(dd, s):
                s_set = set(s)
                for poly in dd:
                    subbed = [[x for x in m if x not in s_set] for m in poly]
                    if sum(1 for m in subbed if not m) % 2 != 0:
                        return False
                return True

            def aug_zn(zn_d, grading_mod, n, num_comp, lv_inner):
                from math import gcd as _gcd
                g_eff = [x % (2 * r) for x in gr] if (grading_mod == 0 and isinstance(r, int) and r != 0) else gr[:]

                def is_grade0(x):
                    return x == 0 if grading_mod == 0 else x % grading_mod == 0

                def is_grade1(x):
                    if grading_mod == 0: return x == 1
                    if grading_mod == 1: return True
                    return x % grading_mod == 1

                grade0_gens = [i + 1 for i, x in enumerate(g_eff) if is_grade0(x)]
                grade1_gens = [i + 1 for i, x in enumerate(g_eff) if is_grade1(x)]
                grade0_set = set(grade0_gens)

                units = [v for v in range(1, n) if _gcd(v, n) == 1]
                lambda_ranges = [
                    [lv_inner[c]] if (lv_inner is not None and c in lv_inner) else units
                    for c in range(num_comp)
                ]

                # Extract conditions from grade-1 differentials.
                # Lambda_c terms have empty word (always pass the grade-0 filter).
                conditions = [
                    [(w, kind, coeff) for (w, kind), coeff in zn_d[g1 - 1].items()
                     if all(gi in grade0_set for gi in w)]
                    for g1 in grade1_gens
                ]

                def eval_cond(poly_terms, augm_dict):
                    total = 0
                    for word, kind, coeff in poly_terms:
                        if isinstance(kind, tuple):
                            lam_val = augm_dict.get(('lambda', kind[1]), 1)
                            lam_factor = lam_val if kind[0] == 'lambda' else pow(lam_val, -1, n)
                        else:
                            lam_factor = 1
                        term = (coeff * lam_factor) % n
                        for gen in word:
                            term = (term * augm_dict.get(gen, 0)) % n
                        total = (total + term) % n
                    return total

                result = []
                for gen_vals in product(range(n), repeat=len(grade0_gens)):
                    base = dict(zip(grade0_gens, gen_vals))
                    for lam_vals in product(*lambda_ranges):
                        augm_dict = dict(base)
                        for c, v in enumerate(lam_vals):
                            augm_dict[('lambda', c)] = v
                        if all(eval_cond(cond, augm_dict) == 0 for cond in conditions):
                            result.append(dict(augm_dict))
                return result

            if modulus == 2:
                aQ, a1, a0, d_red = ac(d, grading_mod)
                if aQ is None:
                    raw = []
                elif not aQ and not a1:
                    raw = [[]]
                else:
                    raw = []
                    for sz in range(len(aQ) + 1):
                        for subset in combinations(aQ, sz):
                            s = sorted(a1 + list(subset))
                            if aug_q(d_red, s):
                                raw.append(s)
            else:
                raw = aug_zn(d, grading_mod, modulus, num_comp, lv)
            self._augmentations_cache[cache_key] = [
                Augmentation(a, self, grading_mod) for a in raw
            ]
        return self._augmentations_cache[cache_key]

    def lin_diff(self, augmentation: 'Augmentation', n: int = 1) -> List[List[List[int]]]:
        """
        Degree-n part of the augmented differential with respect to augmentation.

        Applies augmentation to self.differential, expanding each generator that
        is sent to 1, then keeps only monomials of output word length n.

        n=1 (default) gives the linearized differential — the map one would
        normally call the linearized differential in contact homology.
        n=2 gives the quadratic part, used for cup products.

        Only implemented over Z/2. For Z/n linearized homology use
        ``_dim_homology_zn`` directly.
        """
        if self.ring != GroundRing.Z2:
            raise NotImplementedError(
                'lin_diff is only implemented over Z/2; '
                'use _dim_homology_zn for Z/n linearized homology.'
            )
        d = self.differential
        aug_set = set(augmentation.data)
        aug_d = []
        for poly in d:
            new_terms = []
            for mono in poly:
                expanded = [[]]
                for x in mono:
                    if x in aug_set:
                        expanded = [e + [x] for e in expanded] + [e for e in expanded]
                    else:
                        expanded = [e + [x] for e in expanded]
                new_terms.extend(expanded)
            cnt = Counter(tuple(e) for e in new_terms)
            aug_d.append([list(k) for k, v in cnt.items() if v % 2 == 1])
        return [[m for m in poly if len(m) == n] for poly in aug_d]

    def _diff_matrix_f2(self, which: int, ld: List) -> List[List[int]]:
        """F_2 matrix of the linearized differential ld mapping _gens_spaces[which-1] → _gens_spaces[which]."""
        domain = self._gens_spaces[which - 1]
        rng = self._gens_spaces[which]
        flat = {i + 1: [x for m in poly for x in m] for i, poly in enumerate(ld)}
        return [[1 if rg in flat.get(dg, []) else 0 for dg in domain] for rg in rng]

    def _dim_homology_z2(self, place: int, ld: List) -> int:
        """Dimension of Z/2 linearized homology at grading position place (1-indexed in _gens_spaces)."""
        def nullity_f2(mat, ncols):
            if not mat:
                return ncols  # zero map into empty space; full domain is the kernel
            if not mat[0]:
                return 0
            return ncols - _rank_f2(mat)

        gs = self._gens_spaces
        num = len(gs)
        if num == 1:
            return len(gs[0])
        if place == 1:
            return nullity_f2(self._diff_matrix_f2(1, ld), len(gs[0]))
        if place == num:
            return len(gs[num - 1]) - _rank_f2(self._diff_matrix_f2(num - 1, ld))
        return (nullity_f2(self._diff_matrix_f2(place, ld), len(gs[place - 1])) -
                _rank_f2(self._diff_matrix_f2(place - 1, ld)))

    def _dim_homology_zn(self, place: int, augmentation: 'Augmentation') -> int:
        """Dimension of Z/n linearized homology at grading position place (1-indexed in _gens_spaces)."""
        zn_d = self.differential
        augm_dict = augmentation.data
        n = self.ring.modulus
        if n < 2 or any(n % k == 0 for k in range(2, int(n**0.5) + 1)):
            raise NotImplementedError(
                f'Linearized homology dimensions require a field coefficient ring; '
                f'Z/{n} is not a field (n must be prime). '
                f'Augmentation enumeration over Z/{n} is still supported.'
            )
        gs = self._gens_spaces

        def rank_zn(mat):
            return _rank_zn(mat, n)

        def nullity_zn(mat, ncols):
            if not mat:
                return ncols  # zero map into empty space; full domain is the kernel
            if not mat[0]:
                return 0
            return ncols - rank_zn(mat)

        def diff_matrix_zn(which):
            domain = gs[which - 1]
            rng = gs[which]
            rng_idx = {g: i for i, g in enumerate(rng)}
            mat = [[0] * len(domain) for _ in range(len(rng))]
            for col, gen_d in enumerate(domain):
                for (word, kind), coeff in zn_d[gen_d - 1].items():
                    lam_factor = augm_dict.get(kind, 1) if isinstance(kind, tuple) else 1
                    for j, a in enumerate(word):
                        if a in rng_idx:
                            prod = (coeff * lam_factor) % n
                            for li, b_gen in enumerate(word):
                                if li != j:
                                    prod = (prod * augm_dict.get(b_gen, 0)) % n
                            mat[rng_idx[a]][col] = (mat[rng_idx[a]][col] + prod) % n
            return mat

        num = len(gs)
        if num == 1:
            return len(gs[0])
        if place == 1:
            return nullity_zn(diff_matrix_zn(1), len(gs[0]))
        if place == num:
            return len(gs[num - 1]) - rank_zn(diff_matrix_zn(num - 1))
        return nullity_zn(diff_matrix_zn(place), len(gs[place - 1])) - rank_zn(diff_matrix_zn(place - 1))

    def dim_homology(self, place: int, augmentation: 'Augmentation') -> int:
        """
        Dimension of linearized homology at grading position place, with respect
        to augmentation.  Dispatches to _dim_homology_z2 or _dim_homology_zn
        depending on self.ring.
        """
        if self.ring == GroundRing.Z2:
            return self._dim_homology_z2(place, self.lin_diff(augmentation))
        else:
            return self._dim_homology_zn(place, augmentation)

    def _lin_hom_as_dict(self, augm: 'Augmentation', grading_mod: int) -> Dict[int, int]:
        """
        Poincaré-Chekanov polynomial for a single augmentation at grading_mod.

        For grading_mod=0 uses the integer-graded chain complex (differential is
        strictly degree -1 in ℤ, so adjacent-grade matrices are exact).
        For grading_mod≥1 builds merged ℤ/n spaces and includes every ld_ε
        contribution between the merged classes, which is required when skip-in-ℤ
        terms still land in the correct ℤ/n grade.
        """
        gr = self.leg.grading

        if grading_mod == 0:
            gs = self._gens_spaces
            max_g = max(gr)
            result: Dict[int, int] = {}
            for k in range(1, len(gs) + 1):
                dim = self.dim_homology(k, augm)
                if dim:
                    result[max_g - k + 1] = dim
            return result

        n = grading_mod
        # Group 1-indexed generators by grade mod n.
        from collections import defaultdict
        spaces: Dict[int, List[int]] = defaultdict(list)
        for i, g in enumerate(gr):
            spaces[g % n].append(i + 1)

        if self.ring == GroundRing.Z2:
            ld = self.lin_diff(augm)
            ld_out = {i + 1: set(x for m in poly for x in m)
                      for i, poly in enumerate(ld)}

            def diff_mat_f2(gfrom: int, gto: int) -> List[List[int]]:
                domain = spaces[gfrom % n]
                rng = spaces[gto % n]
                return [[1 if rg in ld_out.get(dg, set()) else 0
                         for dg in domain] for rg in rng]

            result = {}
            for j in list(spaces):
                mat_out = diff_mat_f2(j, (j - 1) % n)
                mat_in  = diff_mat_f2((j + 1) % n, j)
                dim = (len(spaces[j]) - _rank_f2(mat_out) - _rank_f2(mat_in))
                if dim:
                    result[j] = dim
            return result

        else:
            zn_d = self.differential
            augm_dict = augm.data
            p = self.ring.modulus

            def diff_mat_zp(gfrom: int, gto: int) -> List[List[int]]:
                domain = spaces[gfrom % n]
                rng = spaces[gto % n]
                rng_idx = {g: i for i, g in enumerate(rng)}
                mat = [[0] * len(domain) for _ in range(len(rng))]
                for col, gen_d in enumerate(domain):
                    for (word, kind), coeff in zn_d[gen_d - 1].items():
                        lam_factor = augm_dict.get(kind, 1) if isinstance(kind, tuple) else 1
                        for pos, a in enumerate(word):
                            if a in rng_idx:
                                prod = (coeff * lam_factor) % p
                                for li, b_gen in enumerate(word):
                                    if li != pos:
                                        prod = (prod * augm_dict.get(b_gen, 0)) % p
                                mat[rng_idx[a]][col] = (mat[rng_idx[a]][col] + prod) % p
                return mat

            result = {}
            for j in list(spaces):
                mat_out = diff_mat_zp(j, (j - 1) % n)
                mat_in  = diff_mat_zp((j + 1) % n, j)
                dim = (len(spaces[j]) - _rank_zn(mat_out, p) - _rank_zn(mat_in, p))
                if dim:
                    result[j] = dim
            return result

    def all_lin_hom(self, grading_mod: int = 0, as_str: bool = False):
        """
        Set of distinct Poincaré-Chekanov polynomials over all augmentations.
        Cached per grading_mod.  Not supported for Z[λ].
        Returns List[Dict[int,int]], or List[str] if as_str=True.
        """
        if self.ring == GroundRing.ZLAMBDA:
            raise NotImplementedError('all_lin_hom not implemented over Z[λ]')
        if grading_mod not in self._lin_hom_cache:
            augms = self.augmentations(grading_mod=grading_mod)
            seen, results = set(), []
            for augm_obj in augms:
                poly = self._lin_hom_as_dict(augm_obj, grading_mod)
                key = frozenset(poly.items())
                if key not in seen:
                    seen.add(key)
                    results.append((poly, augm_obj))
            results.sort(key=lambda pair: sorted(pair[0].items()))
            self._lin_hom_cache[grading_mod] = results
        pairs = self._lin_hom_cache[grading_mod]
        if as_str:
            return [augm.format_poincare() for _, augm in pairs]
        return [poly for poly, _ in pairs]

    def lin_hom_reps(self, grading_mod: int = 0) -> List[Tuple[Dict[int, int], 'Augmentation']]:
        """
        Returns (poly, representative_augmentation) pairs — one per distinct
        Poincaré-Chekanov polynomial.  Not supported for Z[λ].
        """
        if self.ring == GroundRing.ZLAMBDA:
            raise NotImplementedError('lin_hom_reps not implemented over Z[λ]')
        self.all_lin_hom(grading_mod=grading_mod)
        return list(self._lin_hom_cache[grading_mod])

    def aug_count(self, grading_mod: int = 0) -> float:
        """
        Normalized graded augmentation number (Ng-Sabloff normalization).
        |ring|^(−χ*_ρ/2) × |Aug_ρ|, where ρ = grading_mod.

        grading_mod=0 (Z-graded): χ*_0 = Σ_{k≥0}(-1)^k a_k + Σ_{k<0}(-1)^{k+1} a_k,
            with an additional −1 shift so the exponent is (−1−χ*_0)/2.
        grading_mod=ρ odd: χ*_ρ = Σ_{k=0}^{ρ-1} (-1)^k a_k (a_k = #generators of degree k mod ρ).
        grading_mod=ρ even, ρ>0: raises NotImplementedError (no well-defined invariant).
        Not supported for Z[λ].
        """
        if self.ring == GroundRing.ZLAMBDA:
            raise NotImplementedError('aug_count not defined over Z[λ]')
        if grading_mod > 0 and grading_mod % 2 == 0:
            raise NotImplementedError(
                f'aug_count not defined for even grading_mod={grading_mod}'
            )
        n = self.ring.modulus
        n_aug = len(self.augmentations(grading_mod=grading_mod))
        gr = self.leg.grading
        if grading_mod == 0:
            min_g, max_g = min(gr), max(gr)
            chi = sum(((-1) ** k) * gr.count(k) for k in range(0, max_g + 1))
            chi += sum(((-1) ** (k + 1)) * gr.count(k) for k in range(min_g, 0))
            exp = (-1 - chi) / 2
        else:
            rho = grading_mod
            chi = sum(((-1) ** k) * sum(1 for g in gr if g % rho == k) for k in range(rho))
            exp = -chi / 2
        return (n ** int(exp)) * n_aug if exp == int(exp) else (n ** exp) * n_aug

    def print_differential(self) -> None:
        """Print d(a[i]) for each generator in human-readable form."""
        if self.ring == GroundRing.Z2:
            for i, poly in enumerate(self.differential):
                terms = ' + '.join(
                    '1' if not m else ' * '.join(f'a[{g}]' for g in m)
                    for m in poly
                ) or '0'
                print(f'  d(a[{i + 1}]) = {terms}')
        elif self.ring == GroundRing.ZLAMBDA:
            # cast: differential is List[Dict] here but typed as List[List] for Z/2 compatibility
            for i, poly_dict in enumerate(cast(list[dict[Any, Any]], self.differential)):
                if not poly_dict:
                    entry = "0"
                else:
                    terms = []
                    for (word, kind), coeff in poly_dict.items():
                        if coeff == 0:
                            continue
                        mon = " * ".join(f"a[{k}]" for k in word) if word else "1"
                        if isinstance(kind, tuple):  # ('lambda'/'lambda_inv', c), 1-indexed
                            lsym = f"λ_{kind[1] + 1}" if kind[0] == 'lambda' else f"λ_{kind[1] + 1}⁻¹"
                            s = f"{lsym} * {mon}" if mon != "1" else lsym
                        else:
                            s = (mon if coeff == 1 else f"-{mon}" if coeff == -1
                                 else f"{coeff}*{mon}")
                        terms.append(s)
                    entry = " + ".join(terms) if terms else "0"
                print(f'  d(a[{i + 1}]) = {entry}')
        else:
            # Z/n: same dict format as ZLAMBDA; coefficients are non-negative mod n
            for i, poly_dict in enumerate(self.differential):
                if not poly_dict:
                    entry = "0"
                else:
                    terms = []
                    for (word, kind), coeff in poly_dict.items():
                        if coeff == 0:
                            continue
                        mon = " * ".join(f"a[{k}]" for k in word) if word else "1"
                        if isinstance(kind, tuple):  # ('lambda'/'lambda_inv', c), 1-indexed
                            lsym = f"λ_{kind[1] + 1}" if kind[0] == 'lambda' else f"λ_{kind[1] + 1}⁻¹"
                            s = f"{lsym} * {mon}" if mon != "1" else lsym
                        else:
                            s = mon if coeff == 1 else f"{coeff}*{mon}"
                        terms.append(s)
                    entry = " + ".join(terms) if terms else "0"
                print(f'  d(a[{i + 1}]) = {entry}')


# ============================================================
# Section 5: Augmentation
# ============================================================

class Augmentation:
    """
    A single augmentation ε: (DGA generators) → (ground ring).

    Obtained via DGA.augmentations() or Leg.augmentations().

    Attributes
    ----------
    data    augmentation data:
              Z/2 : List[int]           generators sent to 1 (others → 0)
              Z/p : Dict[int|tuple,int] generator → value; also ('lambda', c) → unit value
    dga     the parent DGA

    Properties  (computed once, cached)
    ------------------------------------
    lin_hom          Dict[int, int]  Poincaré-Chekanov polynomial
    cohomology_basis                 (Z/2 only) basis for linearized cohomology
    double_products                  (Z/2 only) cup-product table
    """

    def __init__(self, data, dga: DGA, grading_mod: int = 0) -> None:
        self.data = data
        self.dga = dga
        self.grading_mod = grading_mod

    def __repr__(self) -> str:
        if isinstance(self.data, dict):
            gen_data = {k: v for k, v in self.data.items() if isinstance(k, int)}
            lam_data = {k: v for k, v in self.data.items() if isinstance(k, tuple)}
            parts = [f'ring={self.dga.ring!r}', f'grading_mod={self.grading_mod!r}',
                     f'gens={gen_data!r}']
            if lam_data:
                parts.append(f'lambdas={lam_data!r}')
            return f'Augmentation({", ".join(parts)})'
        return f'Augmentation({self.data!r}, ring={self.dga.ring!r}, grading_mod={self.grading_mod!r})'

    @cached_property
    def lin_hom(self) -> Dict[int, int]:
        """
        Poincaré-Chekanov polynomial of linearized contact homology w.r.t. this ε.
        Returns {grading: dimension} (zero dimensions omitted).
        """
        ring = self.dga.ring
        if ring == GroundRing.ZLAMBDA:
            raise NotImplementedError('lin_hom not implemented over Z[λ]')
        return self.dga._lin_hom_as_dict(self, self.grading_mod)

    def format_poincare(self) -> str:
        """Format self.lin_hom as a Poincaré polynomial string in t."""
        poly = self.lin_hom
        if not poly:
            return "0"
        terms = []
        for g in sorted(poly, reverse=True):
            d = poly[g]
            terms.append(f"t^{g}" if d == 1 else f"{d}*t^{g}")
        return " + ".join(terms)

    @cached_property
    def cohomology_basis(self):
        """
        Basis for linearized cohomology (Z/2 only).

        Returns ``List[List[List[int]]]``:
          - Outer list: one entry per grading space, ordered highest grade first
            (same order as ``DGA._gens_spaces``).
          - Middle list: basis vectors for the cohomology at that grade.
          - Inner list: 1-indexed generator numbers where the F_2 basis vector
            has a 1 (i.e., the support of the basis vector as a subset of
            generators).

        Example: ``[[4]], [[1, 3], [2]]]`` means the highest-grade cohomology
        is 1-dimensional spanned by ``a_4``, and the next grade is 2-dimensional
        spanned by ``a_1 + a_3`` and ``a_2``.
        """
        if self.dga.ring != GroundRing.Z2:
            raise NotImplementedError(
                'cohomology_basis is only implemented over Z/2'
            )
        def null_space_f2(m):
            if not m or not m[0]:
                return []
            c = len(m[0])
            rref, pivot_cols = _rref_f2(m)
            pivot_set = set(pivot_cols)
            free_cols = [j for j in range(c) if j not in pivot_set]
            basis = []
            for fc in free_cols:
                vec = [0] * c
                vec[fc] = 1
                for pi, pc in enumerate(pivot_cols):
                    if pi < len(rref):
                        vec[pc] = rref[pi][fc] % 2
                basis.append(vec)
            return basis

        ld = self.dga.lin_diff(self)
        gs = self.dga._gens_spaces
        n_spaces = len(gs)
        result = []
        for k in range(1, n_spaces + 1):
            space = gs[k - 1]
            if not space:
                result.append([])
                continue
            if k == n_spaces or not gs[k]:
                ker_basis = [[1 if i == j else 0 for j in range(len(space))]
                             for i in range(len(space))]
            else:
                mat_k = self.dga._diff_matrix_f2(k, ld)
                if mat_k and mat_k[0]:
                    ker_basis = null_space_f2(mat_k)  # may legitimately be []
                else:
                    ker_basis = [[1 if i == j else 0 for j in range(len(space))]
                                 for i in range(len(space))]
            cohom = [[space[i] for i, v in enumerate(vec) if v] for vec in ker_basis]
            result.append([c for c in cohom if c])
        return result

    @cached_property
    def double_products(self):
        """Cup-product multiplication table on linearized cohomology (Z/2 only). See [CEKSW11]."""
        if self.dga.ring != GroundRing.Z2:
            raise NotImplementedError(
                'double_products is only implemented over Z/2'
            )
        double_diff = self.dga.lin_diff(self, 2)
        co_basis = [gen for space in self.cohomology_basis for gen in space]

        def multiply_once(first, second):
            count = Counter()
            for a_gen in first:
                for b_gen in second:
                    pair = tuple(sorted([a_gen, b_gen]))
                    for i, poly in enumerate(double_diff):
                        if pair in [tuple(sorted(m)) for m in poly if len(m) == 2]:
                            count[i + 1] += 1
            result_gens = [g for g, c in count.items() if c % 2 == 1]
            return result_gens if result_gens else '.'

        table = [[multiply_once(co_basis[j], co_basis[k])
                  for k in range(len(co_basis))]
                 for j in range(len(co_basis))]
        return table, co_basis


# ============================================================
# Section 6: Atlas
# ============================================================
# Keys follow the naming convention described in the module docstring.
# Values are lists of braid words, one per Legendrian isotopy class (0-indexed).
# Data from the Legendrian knot atlas (https://sites.math.duke.edu/~ng/atlas/).

ATLAS: Dict[str, List[List[int]]] = {
    'mK3_1': [[2, 3, 1, 3, 1, 3, 2]],
    'K4_1': [[2, 3, 1, 3, 1, 2, 2]],
    'mK5_1': [[2, 3, 1, 1, 3, 1, 1, 3, 2]],
    'mK5_2': [
        [2, 3, 1, 3, 4, 5, 1, 2, 3, 4, 2, 4],
        [2, 3, 1, 3, 1, 2, 2, 3, 1, 3, 2],
    ],
    'K6_1': [[2, 3, 1, 1, 3, 1, 1, 2, 2]],
    'mK6_1': [
        [4, 5, 3, 5, 6, 7, 3, 4, 2, 1, 3, 2, 5, 6, 2, 4, 6],
        [2, 3, 1, 3, 1, 2, 2, 1, 3, 2, 2],
    ],
    'K6_2': [[4, 5, 5, 3, 4, 4, 3, 2, 1, 5, 3, 1, 1, 2, 5, 4]],
    'mK6_2': [[4, 5, 3, 3, 5, 3, 2, 1, 4, 1, 3, 4, 3, 2]],
    'mK7_1': [[2, 3, 1, 1, 3, 1, 1, 1, 1, 3, 2]],
    'mK7_2': [
        [6, 7, 7, 5, 8, 9, 5, 4, 6, 3, 5, 2, 4, 7, 8, 3, 1, 2, 2, 4, 6, 8],
        [2, 3, 3, 1, 1, 2, 2, 3, 1, 2, 2, 1, 3, 3, 2],
        [4, 5, 5, 3, 4, 6, 7, 2, 3, 1, 5, 6, 2, 4, 5, 3, 5, 6, 3, 4, 2],
        [4, 5, 3, 5, 3, 2, 1, 4, 4, 5, 3, 2, 1, 4, 2, 4, 3, 5, 4, 2],
    ],
    'K7_3': [
        [2, 3, 1, 1, 3, 4, 5, 1, 1, 2, 3, 4, 2, 4],
        [2, 3, 1, 3, 1, 1, 1, 2, 2, 1, 3, 3, 2],
    ],
    'K7_4': [
        [4, 5, 5, 3, 6, 7, 2, 1, 4, 5, 6, 8, 9, 7, 8, 1, 3, 2, 4, 5, 3, 4, 6, 8, 4, 2],
        [6, 7, 7, 5, 6, 6, 5, 4, 3, 2, 1, 7, 5, 3, 1, 7, 6, 5, 4, 1, 2],
        [6, 7, 7, 5, 4, 3, 6, 5, 3, 2, 4, 1, 3, 6, 2, 5, 7, 4, 6, 2, 4, 5, 6, 3, 4],
        [2, 3, 3, 1, 2, 2, 1, 3, 1, 2, 2, 1, 3, 3, 2],
    ],
    'mK7_5': [
        [4, 5, 3, 5, 3, 6, 3, 7, 2, 1, 4, 5, 6, 1, 3, 4, 6, 3, 2],
        [2, 3, 3, 1, 1, 1, 2, 2, 1, 3, 1, 3, 2],
    ],
    'mK7_6': [
        [4, 5, 3, 5, 3, 2, 1, 4, 1, 3, 4, 1, 2, 4, 3, 5, 4, 2],
        [6, 7, 5, 7, 5, 4, 3, 3, 6, 2, 1, 5, 6, 3, 1, 1, 2, 5, 4],
        [4, 5, 3, 3, 5, 2, 1, 4, 1, 3, 1, 2, 4, 2],
    ],
    'mK7_7': [
        [6, 7, 5, 5, 7, 4, 3, 6, 3, 5, 2, 1, 4, 1, 3, 6, 4, 3, 2],
        [4, 5, 3, 3, 5, 2, 1, 4, 1, 3, 4, 1, 5, 2, 3, 2, 5, 4],
    ],
    'K8_19': [[4, 5, 2, 1, 3, 2, 1, 5, 2, 4, 1, 5, 2, 1, 2, 3, 1, 2, 5, 4]],
    'K8_21': [
        [4, 5, 5, 3, 2, 1, 4, 3, 4, 1, 3, 2, 4, 2],
        [4, 5, 3, 5, 2, 1, 4, 3, 4, 1, 3, 4, 5, 1, 2, 3, 5, 4],
    ],
    'mK8_21': [[4, 5, 3, 2, 1, 5, 1, 3, 4, 1, 2, 4, 3, 5, 4, 2]],
    'K9_42': [[4, 5, 3, 2, 5, 1, 4, 3, 4, 3, 1, 4, 3, 4, 1, 2]],
    'mK9_42': [[4, 5, 3, 2, 1, 5, 1, 3, 4, 2, 3, 3, 4, 2]],
    'K9_43': [[4, 5, 3, 5, 3, 2, 1, 4, 1, 3, 4, 1, 3, 4, 3, 2]],
    'mK9_44': [[4, 5, 5, 3, 3, 2, 1, 4, 3, 4, 1, 2, 5, 2, 3, 1, 5, 4, 1, 2]],
    'mK9_45': [
        [6, 7, 7, 5, 6, 4, 3, 6, 2, 5, 4, 6, 4, 7, 5, 3, 2, 7, 6, 5, 4],
        [4, 5, 3, 2, 1, 5, 3, 1, 2, 4, 2, 3, 1, 2, 4, 2],
    ],
    'K9_46': [[4, 5, 2, 1, 3, 2, 1, 5, 2, 4, 1, 5, 2, 3, 1, 4, 3, 4, 1, 2]],
    'mK9_46': [[4, 5, 2, 1, 3, 1, 2, 5, 2, 3, 1, 4, 2, 4, 3, 5, 4, 2]],
    'mK9_47': [[6, 7, 4, 3, 5, 3, 2, 4, 3, 1, 7, 2, 2, 6, 4, 2, 1, 3, 6, 5, 1, 2, 4, 7, 6]],
    'K9_48': [
        [6, 7, 5, 7, 4, 3, 2, 1, 6, 5, 3, 6, 1, 3, 4, 2, 3, 5, 4, 1, 2, 6, 4],
        [6, 7, 8, 9, 5, 7, 8, 4, 3, 2, 1, 9, 3, 2, 8, 6, 9, 7, 5, 2, 1, 8, 7, 6, 4,
         8, 7, 3, 5, 6, 9, 8, 1, 2, 5, 4],
        [4, 3, 6, 7, 5, 6, 3, 7, 8, 9, 4, 6, 2, 1, 5, 7, 8, 6, 7, 9, 8, 5, 7, 1, 3,
         8, 1, 2, 7, 6, 5, 4],
        [6, 7, 5, 7, 4, 3, 6, 2, 1, 5, 6, 1, 3, 5, 6, 7, 1, 2, 3, 4, 5, 7, 6],
    ],
    'K9_49': [
        [4, 5, 5, 3, 6, 7, 4, 2, 1, 5, 6, 1, 3, 4, 6, 3, 4, 5, 6, 1, 2, 3, 4, 7, 5, 7, 6],
        [4, 5, 3, 5, 2, 1, 4, 1, 3, 2, 4, 3, 5, 2, 1, 3, 4, 2, 5, 1, 2, 3, 1, 2, 5, 4],
    ],
    'K10_124': [[4, 5, 2, 1, 3, 2, 1, 2, 5, 1, 2, 4, 1, 5, 2, 1, 2, 3, 1, 2, 5, 4]],
    'K10_128': [
        [4, 5, 3, 5, 6, 7, 4, 2, 1, 5, 6, 3, 4, 3, 1, 6, 4, 3, 4, 3, 2],
        [6, 7, 5, 8, 9, 4, 3, 2, 1, 7, 9, 6, 3, 2, 1, 5, 6, 2, 4, 5, 1, 2, 3, 6, 9,
         8, 1, 2, 5, 4],
    ],
    'K10_132': [
        [6, 7, 4, 3, 7, 5, 3, 6, 4, 2, 5, 1, 3, 2, 5, 2, 4, 6, 2],
        [4, 5, 3, 5, 3, 2, 4, 1, 3, 2, 4, 2, 5, 1, 3, 2, 4, 4, 3, 5, 4, 2],
    ],
    'K10_136': [
        [4, 5, 5, 6, 7, 3, 4, 2, 1, 5, 3, 2, 4, 7, 6, 3, 2, 3, 5, 2, 6, 3, 2, 5, 4],
        [8, 9, 7, 6, 5, 9, 4, 3, 5, 4, 3, 8, 2, 1, 7, 8, 6, 4, 7, 1, 3, 4, 5, 8, 1,
         2, 3, 4, 7, 6],
        [6, 7, 5, 4, 3, 7, 2, 1, 6, 3, 2, 5, 1, 2, 4, 6, 1, 2, 3, 5, 1, 2, 6, 3, 4,
         7, 5, 7, 6],
        [4, 5, 3, 2, 5, 1, 4, 3, 4, 3, 1, 4, 3, 2, 4, 2],
    ],
    'K10_139': [[2, 3, 4, 5, 1, 3, 4, 5, 1, 2, 4, 5, 1, 2, 4, 1, 5, 3, 1, 2, 5, 4]],
    'mK10_140': [
        [4, 5, 5, 3, 4, 6, 7, 2, 1, 5, 6, 3, 2, 2, 4, 6, 1, 3, 5, 7, 1, 2, 3, 7, 6,
         3, 4],
        [2, 3, 1, 3, 4, 5, 2, 3, 4, 2, 1, 4, 3, 5, 1, 4, 2, 4, 3, 5, 4, 2],
    ],
    'K10_142': [
        [6, 7, 5, 7, 4, 3, 6, 8, 9, 2, 1, 5, 7, 8, 4, 6, 7, 3, 4, 8, 1, 3, 5, 7, 4,
         8, 3, 2, 7, 6],
        [4, 5, 3, 5, 2, 1, 4, 3, 4, 1, 3, 2, 5, 1, 2, 4, 5, 1, 4, 3, 5, 4, 1, 2],
    ],
    'mK10_145': [[2, 3, 4, 5, 1, 3, 4, 5, 1, 4, 2, 3, 3, 4, 2, 5, 3, 1, 1, 2, 5, 4]],
    'K10_160': [
        [6, 7, 5, 7, 4, 6, 3, 5, 2, 1, 3, 2, 4, 3, 6, 1, 5, 3, 6, 5, 1, 2, 6, 5, 4],
        [6, 7, 7, 5, 8, 9, 4, 3, 2, 6, 7, 5, 3, 6, 2, 9, 1, 3, 4, 7, 2, 3, 5, 1, 2,
         6, 5, 4, 9, 8],
    ],
    'mK10_161': [[2, 3, 4, 5, 1, 3, 4, 5, 1, 4, 2, 3, 5, 1, 3, 2, 4, 5, 1, 3, 1, 2, 5, 4]],
    'mK11n19': [[4, 5, 2, 1, 3, 2, 1, 2, 5, 1, 4, 2, 3, 1, 2, 5, 3, 2, 3, 2, 5, 4]],
    'K11n38': [[4, 5, 3, 5, 2, 1, 4, 3, 4, 3, 1, 4, 3, 4, 3, 4, 1, 2]],
    'K11n95': [[6, 7, 5, 4, 3, 2, 1, 7, 3, 2, 5, 6, 1, 2, 4, 5, 6, 3, 5, 7, 4, 1, 2,
                6, 5, 3, 7, 6, 2, 5, 4]],
    'K11n118': [[2, 3, 4, 5, 1, 3, 4, 5, 1, 2, 4, 3, 5, 1, 3, 4, 2, 4, 3, 5, 4, 2]],
    'K12n242': [[4, 5, 2, 1, 3, 2, 1, 2, 5, 1, 4, 2, 5, 1, 2, 4, 5, 1, 2, 3, 1, 2, 5, 4]],
    'K12n591': [[4, 5, 2, 1, 3, 2, 1, 5, 2, 4, 1, 2, 5, 3, 1, 3, 4, 2, 5, 1, 2, 3, 1,
                 2, 5, 4]],
    'K15n41185': [[4, 5, 2, 1, 6, 7, 5, 6, 3, 2, 4, 5, 1, 7, 2, 6, 1, 3, 7, 5, 2, 1, 6,
                   7, 3, 2, 5, 4, 6, 1, 5, 2, 3, 7, 6, 1, 2, 5, 4]],
    # ---- entries derived from grid atlas (rot != 0 or previously missing) ----
    'K3_1': [[4, 3, 2, 1, 1, 3, 5, 3, 2, 1, 4, 3, 2, 3, 4]],
    'K5_1': [[2, 1, 1, 3, 2, 2, 1, 3, 2, 2]],
    'K5_2': [[4, 3, 3, 3, 2, 1, 5, 1, 3, 3, 2, 1, 4, 3, 2, 3, 4]],
    'K6_3': [[4, 3, 2, 1, 1, 3, 5, 1, 2, 2, 2, 3, 4]],
    'K7_1': [[6, 5, 4, 3, 3, 2, 1, 5, 4, 3, 2, 2, 4, 7, 1, 3, 2, 2, 5, 6]],
    'K7_2': [[4, 3, 3, 3, 3, 5, 3, 2, 1, 1, 3, 3, 2, 1, 4, 3, 2, 3, 4]],
    'K7_5': [[8, 7, 7, 6, 5, 5, 4, 3, 2, 1, 9, 8, 7, 1, 3, 5, 8, 5, 4, 3, 2, 1,
              6, 5, 4, 3, 2, 5, 4, 3, 6, 5, 4, 5, 6]],
    'K7_6': [[8, 7, 7, 6, 5, 4, 3, 3, 2, 1, 9, 5, 7, 1, 3, 7, 6, 5, 4, 3, 2, 1,
              8, 7, 6, 5, 4, 3, 2, 7, 6, 5, 4, 3, 8, 7, 6, 5, 4, 7, 6, 5, 8, 7, 6, 7, 8]],
    'K7_7': [[6, 5, 4, 3, 3, 2, 1, 5, 1, 3, 2, 4, 7, 3, 5, 3, 5, 4, 3, 2, 1,
              6, 5, 4, 3, 2, 5, 4, 3, 6, 5, 4, 5, 6]],
    'K8_20': [[4, 3, 2, 1, 6, 5, 4, 3, 2, 1, 1, 7, 2, 4, 6, 1, 7, 1, 4, 3, 2, 5, 4, 3,
               4, 5, 5, 4, 3, 2, 1, 6, 5, 4, 3, 2, 5, 4, 3, 6, 5, 4, 5, 6]],
    'K9_44': [[4, 3, 6, 5, 4, 3, 3, 2, 1, 4, 3, 2, 6, 1, 5, 2, 4, 7, 3, 6, 1, 5,
               3, 6, 3, 2, 1, 4, 3, 2, 3, 4]],
    'K9_45': [[4, 3, 8, 7, 7, 6, 5, 4, 3, 3, 2, 1, 7, 9, 4, 6, 1, 3, 7, 4, 5, 7, 6, 5,
               4, 3, 2, 1, 8, 7, 6, 5, 4, 3, 2, 7, 6, 5, 4, 3, 8, 7, 6, 5, 4, 7, 6, 5,
               8, 7, 6, 7, 8]],
    'K9_47': [[2, 1, 6, 5, 8, 7, 6, 5, 5, 4, 3, 2, 1, 6, 9, 5, 4, 3, 8, 7, 1, 6, 9,
               3, 5, 7, 6, 7, 8, 3, 2, 1, 4, 3, 2, 3, 4]],
    'K10_140': [[4, 3, 6, 5, 4, 3, 3, 4, 3, 2, 1, 7, 4, 3, 2, 6, 1, 5, 2, 4, 1, 5, 2,
                 3, 5, 4, 3, 2, 1, 6, 5, 4, 3, 2, 5, 4, 3, 6, 5, 4, 5, 6]],
    'K10_145': [[2, 1, 4, 3, 2, 1, 1, 5, 2, 4, 5, 2, 4, 1, 5, 4, 3, 2, 2, 3, 4]],
    'K10_161': [[6, 5, 4, 3, 8, 7, 6, 5, 4, 3, 3, 2, 1, 4, 3, 2, 9, 6, 8, 1, 5, 7, 9,
                 2, 4, 6, 8, 6, 5, 4, 3, 2, 7, 6, 5, 4, 3, 6, 5, 4, 7, 6, 5, 6, 7,
                 7, 6, 5, 4, 3, 2, 1, 8, 7, 6, 5, 4, 3, 2, 7, 6, 5, 4, 3, 8, 7, 6, 5,
                 4, 7, 6, 5, 8, 7, 6, 7, 8]],
    'K11n19': [[4, 3, 2, 1, 1, 3, 2, 5, 2, 1, 3, 2, 4, 3, 3, 2, 4]],
    'mK7_3': [[2, 1, 1, 1, 3, 1, 2, 2, 1, 3, 2, 2]],
    'mK7_4': [[4, 3, 3, 3, 2, 1, 1, 5, 1, 3, 1, 3, 2, 1, 4, 3, 2, 3, 4]],
    'mK8_19': [[6, 5, 4, 3, 2, 1, 1, 3, 5, 7, 2, 4, 6, 6, 5, 4, 3, 2,
                5, 4, 3, 6, 5, 4, 5, 6]],
    'mK8_20': [[4, 3, 2, 1, 1, 3, 2, 5, 2, 1, 3, 2, 2, 3, 4]],
    'mK9_43': [[2, 1, 4, 3, 2, 1, 5, 4, 3, 1, 2, 3, 3, 2, 4, 3, 3, 2, 4]],
    'mK9_48': [[6, 5, 4, 3, 3, 3, 2, 1, 5, 4, 7, 1, 3, 4, 5, 3, 2, 1, 2, 3, 5, 4, 3,
                2, 1, 6, 5, 4, 3, 2, 5, 4, 3, 6, 5, 4, 5, 6]],
    'mK9_49': [[6, 5, 4, 3, 3, 3, 2, 1, 5, 7, 1, 3, 2, 4, 5, 4, 3, 2,
                5, 4, 3, 6, 5, 4, 5, 6]],
    'mK10_124': [[8, 7, 6, 5, 4, 3, 2, 1, 1, 3, 5, 7, 9, 2, 4, 6, 8, 8, 7, 6, 5, 4, 3,
                  2, 7, 6, 5, 4, 3, 8, 7, 6, 5, 4, 7, 6, 5, 8, 7, 6, 7, 8]],
    'mK10_128': [[6, 5, 5, 4, 3, 3, 2, 1, 5, 4, 3, 2, 7, 2, 4, 1, 3, 5, 5, 4, 3, 2, 1,
                  6, 5, 4, 3, 2, 5, 4, 3, 6, 5, 4, 5, 6]],
    'mK10_132': [[6, 5, 5, 7, 5, 4, 3, 2, 1, 6, 5, 4, 3, 2, 2, 4, 6, 1, 3, 5, 7, 5, 4,
                  3, 2, 1, 6, 5, 4, 3, 2, 5, 4, 3, 6, 5, 4, 5, 6]],
    'mK10_136': [[6, 5, 4, 3, 3, 5, 3, 2, 1, 4, 3, 2, 7, 2, 4, 1, 3, 5, 5, 4, 3, 2, 1,
                  6, 5, 4, 3, 2, 5, 4, 3, 6, 5, 4, 5, 6]],
    'mK10_139': [[6, 5, 4, 3, 3, 2, 1, 5, 7, 4, 6, 8, 3, 5, 7, 7, 6, 5, 4, 3, 8, 7, 6,
                  5, 4, 7, 6, 5, 8, 7, 6, 7, 8, 1, 2]],
    'mK10_142': [[6, 5, 4, 3, 3, 3, 2, 1, 5, 7, 6, 1, 3, 2, 4, 6, 5, 4, 3, 2,
                  5, 4, 3, 6, 5, 4, 5, 6]],
    'mK10_160': [[4, 3, 2, 1, 1, 3, 2, 2, 5, 1, 3, 2, 4, 2, 4, 3, 2, 3, 4]],
    'mK11n38': [[8, 7, 6, 5, 5, 4, 3, 2, 1, 7, 6, 5, 4, 3, 2, 9, 2, 4, 6, 1, 3, 5, 7,
                 7, 6, 5, 4, 3, 2, 1, 8, 7, 6, 5, 4, 3, 2, 7, 6, 5, 4, 3, 8, 7, 6, 5,
                 4, 7, 6, 5, 8, 7, 6, 7, 8]],
    'mK11n95': [[6, 5, 4, 3, 3, 5, 4, 7, 2, 6, 4, 3, 5, 2, 6, 1, 5, 4, 3,
                 6, 5, 4, 5, 6, 1, 2]],
    'mK11n118': [[2, 1, 6, 5, 8, 7, 6, 5, 5, 4, 3, 2, 1, 9, 6, 8, 1, 5, 7, 2, 4, 6, 9,
                  7, 6, 5, 4, 3, 2, 7, 6, 5, 4, 3, 8, 7, 6, 5, 4, 7, 6, 5, 8, 7, 6, 7, 8]],
    'mK12n242': [[4, 3, 2, 1, 1, 3, 5, 2, 4, 3, 3, 2, 4, 1, 3, 5, 2, 4, 4, 3, 2, 3, 4]],
    'mK12n591': [[4, 3, 2, 1, 1, 3, 2, 5, 4, 2, 1, 4, 3, 5, 2, 4, 4, 3, 2, 3, 4]],
    'mK15n41185': [[8, 7, 6, 5, 4, 3, 2, 1, 1, 3, 5, 7, 9, 2, 4, 6, 8, 3, 5, 7, 7, 6, 5,
                    4, 3, 2, 8, 7, 6, 5, 4, 7, 6, 5, 8, 7, 6, 7, 8]],
}


# ============================================================
# Section 7: Constructions
# ============================================================

def _cable_parts(leg: Leg):
    """Shared braid ingredients for 2-cable constructions."""
    p = leg.num_cusps
    pp = [4 * i - 2 for i in range(1, p + 1)]
    dc = [x for gen in leg.braid for x in (2*gen, 2*gen + 1, 2*gen - 1, 2*gen)]
    return pp, dc


def whitehead_double(leg: Leg) -> Leg:
    """
    Legendrian Whitehead double of leg. See [Ng01].
    Returns a new Leg whose braid encodes the Whitehead double plat closure.
    """
    pp, dc = _cable_parts(leg)
    return Leg([2] + pp + dc + pp, name=f'WhiteheadDouble({leg.name})')


def twisted_2cable(leg: Leg) -> Leg:
    """
    Legendrian twisted 2-cable of leg. See [Ng01].
    Returns a new Leg whose braid encodes the twisted 2-cable plat closure.
    """
    pp, dc = _cable_parts(leg)
    return Leg(pp + [1] + dc + pp, name=f'Twisted2Cable({leg.name})')
