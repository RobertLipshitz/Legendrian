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
Leg accepts three input forms.

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

__version__ = "0.1.0"

import re
from collections import Counter
from functools import cached_property
from itertools import combinations, product
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union, cast


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

    Classical invariants  (computed once, cached as properties)
    -----------------------------------------------------------
    num_components     int
    grading            List[int]
    tb                 int   (Thurston-Bennequin number)
    rot                int   (rotation number)
    ruling_invariant   Dict[int, int]  (Z-graded ruling polynomial in z)

    DGA and augmentations
    ---------------------
    dga(ring)           DGA over ring (default: DEFAULT_GROUND_RING = Z/2);
                        cached per ring
    augmentations(...)  shorthand for self.dga(Z/modulus).augmentations(...)
    lin_hom(...)        shorthand for self.dga(Z/modulus).lin_hom(...)
    rulings(grading_mod) cached per grading_mod

    Visualization
    -------------
    draw(label_generators)  returns matplotlib Figure
    export_svg(filename)    writes SVG, returns filename
    """

    def __init__(
        self,
        input: Union[List[int], List[tuple], str],
        name: Optional[str] = None,
    ) -> None:
        _is_tangle = (
            isinstance(input, list) and bool(input) and isinstance(input[0], tuple)
        )
        if _is_tangle:
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
                f'or atlas name (str), got {type(input).__name__!r}'
            )
        if not _is_tangle:
            self.num_cusps = max(self.braid) // 2 + 1
        self._dga_cache: Dict[GroundRing, DGA] = {}
        self._rulings_cache: Dict[int, List[List[int]]] = {}

    def __repr__(self) -> str:
        return f'Leg({self.name!r})'

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
    def _tangle_to_braid(tangle) -> Tuple[List[int], int]:
        """
        Convert a validated tangle sequence to (braid_word_1indexed, num_cusps).
        Applies LR-II commutation rules until plat form, then extracts braid word.
        """
        seq = list(tangle)
        num_cusps = sum(1 for op, _ in seq if op == '<')
        max_iters = max(10000, len(seq) ** 2 * 50)

        for _ in range(max_iters):
            for i in range(len(seq) - 1):
                op_a, h_a = seq[i]
                op_b, h_b = seq[i + 1]

                # Rule A: X(h) · LC(j) — move LC left past X
                if op_a == 'X' and op_b == '<':
                    h, j = h_a, h_b
                    if h <= j - 2:
                        seq[i], seq[i + 1] = ('<', j), ('X', h)
                    elif h == j - 1:
                        # LR-II: cusp rises by 1, two extra crossings in post-LC config
                        seq[i:i + 2] = [('<', j - 1), ('X', j + 1), ('X', j), ('X', j - 1)]
                    else:  # h >= j
                        seq[i], seq[i + 1] = ('<', j), ('X', h + 2)
                    break

                # Rule B: RC(j) · X(h) — move RC right past X (h in post-RC config)
                elif op_a == '>' and op_b == 'X':
                    j, h = h_a, h_b
                    if h < j:
                        seq[i], seq[i + 1] = ('X', h), ('>', j)
                    elif h == j - 1:
                        # LR-II: RC falls by 1, two extra crossings in pre-RC config
                        seq[i:i + 2] = [('X', j + 1), ('X', j), ('X', j - 1), ('>', j + 1)]
                    else:  # h >= j
                        seq[i], seq[i + 1] = ('X', h + 2), ('>', j)
                    break

                # Rule C: RC(k) · LC(j) — move LC left past RC
                elif op_a == '>' and op_b == '<':
                    k, j = h_a, h_b
                    # j < k means LC is above RC (j = k-1 is an edge case, treated as j < k)
                    if j < k:
                        seq[i], seq[i + 1] = ('<', j), ('>', k + 2)
                    else:
                        seq[i], seq[i + 1] = ('<', j + 2), ('>', k)
                    break

                # LC-LC far-commutation: LC(m)·LC(n) with n > m+1 → LC(n-2)·LC(m)
                elif op_a == '<' and op_b == '<' and h_b > h_a + 1:
                    seq[i], seq[i + 1] = ('<', h_b - 2), ('<', h_a)
                    break

                # Rule D: LC(k) · LC(k+1) — fix nested left cusps (R2 move)
                elif op_a == '<' and op_b == '<' and h_b == h_a + 1:
                    k = h_a
                    seq[i:i + 2] = [('<', k), ('<', k + 2), ('X', k + 1), ('X', k)]
                    break

                # RC-RC far-commutation: RC(n)·RC(m) with n > m+1 → RC(m)·RC(n-2)
                elif op_a == '>' and op_b == '>' and h_a > h_b + 1:
                    seq[i], seq[i + 1] = ('>', h_b), ('>', h_a - 2)
                    break

                # Rule F: RC(k+1) · RC(k) — fix nested right cusps (R2 move)
                elif op_a == '>' and op_b == '>' and h_a == h_b + 1:
                    k = h_b
                    seq[i:i + 2] = [('X', k), ('X', k + 1), ('>', k), ('>', k)]
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
        b = self.braid
        gr = [self._trugrad(b, i) for i in range(1, len(b) + 1)]
        gr.extend([1] * self.num_cusps)
        return gr

    @cached_property
    def tb(self) -> int:
        return sum(1 if x % 2 == 0 else -1 for x in self.grading)

    @cached_property
    def rot(self) -> int:
        b = self.braid
        if not b:
            return 0
        m = max(b)
        h = m + 2 if m % 2 == 0 else m + 1
        backend = list(reversed([h - x for x in b]))
        return abs((self._trugrad(b, 1) + self._trugrad(backend, len(b))) // 2)

    @cached_property
    def ruling_invariant(self) -> Dict[int, int]:
        "Ruling polynomial as a dictionary with keys the degree and values the coefficient"
        c = self.num_cusps
        return dict(Counter(len(r) - c + 1 for r in self.rulings()))

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
        All graded rulings, computed with meet-in-the-middle.
        Cached per grading_mod (verbose output is not cached).
        grading_mod: 0 = Z-graded, 1 = ungraded, n >= 2 = Z/n-graded.
        """
        if grading_mod not in self._rulings_cache:
            if self.num_components != 1:
                raise ValueError("rulings are only defined for knots, not links")
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
                ((i + 1, b[i]) for i in range(cut)), sw, p,
                verbose=verbose, label='left')
            right = self._propagate_rulings(
                ((n - i, b[n - 1 - i]) for i in range(n - cut)), sw, p,
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

    def format_ruling_invariant(self) -> str:
        """Format self.ruling_invariant as a polynomial string in z."""
        d = self.ruling_invariant
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
        format: bool = False,
    ):
        """
        Set of distinct Poincaré-Chekanov polynomials over all augmentations.
        Delegates to self.dga(Z/modulus).all_lin_hom(grading_mod).
        Returns List[Dict[int,int]], or List[str] if format=True.
        """
        ring = GroundRing.Z2 if modulus == 2 else GroundRing.Zn(modulus)
        return self.dga(ring).all_lin_hom(grading_mod=grading_mod, format=format)

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
        strand_num = 2 * (max(b) // 2 + 1)
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

    def draw(self, label_generators: bool = False, use_tangle: bool = False):
        """Plot the front projection of this Leg. Returns a matplotlib Figure."""
        import matplotlib.pyplot as plt
        import numpy as np

        if use_tangle:
            if not hasattr(self, 'tangle'):
                raise AttributeError(
                    'This Leg has no stored tangle; '
                    'initialize with a tangle sequence to use use_tangle=True'
                )
            segments = self._trace_tangle()
            all_x = [pt[0] for seg in segments for pt in seg]
            all_y = [pt[1] for seg in segments for pt in seg]
            fig, ax = plt.subplots(figsize=(max(4, max(all_x) * 0.5), 3))
            _cmap = plt.get_cmap('tab10')
            colors = [_cmap(i) for i in range(10)]
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
                ax.plot(px, py, color=colors[k % 10], linewidth=2,
                        solid_capstyle='round', solid_joinstyle='round')
            ax.yaxis.set_visible(False)
            ax.xaxis.set_visible(False)
            ax.set_title(f'Legendrian Knot  tangle = {self.tangle}')
            plt.tight_layout()
            return fig

        b = self.braid
        strands = self._trace_braid()
        extra_w = 0.8 if label_generators else 0.0
        fig, ax = plt.subplots(figsize=(max(4, len(b) * 0.8) + extra_w, 3))
        _cmap = plt.get_cmap('tab10')
        colors = [_cmap(i) for i in range(10)]
        for k, strand in enumerate(strands):
            xs = np.array(range(1, len(strand) + 1), dtype=float)
            ys = np.array(strand, dtype=float)
            t_fine = np.linspace(xs[0], xs[-1], 300)
            try:
                from scipy.interpolate import make_interp_spline
                spl = make_interp_spline(xs, ys, k=min(2, len(xs) - 1))
                ax.plot(t_fine, spl(t_fine), color=colors[k % 10], linewidth=2)
            except Exception:
                ax.plot(xs, ys, color=colors[k % 10], linewidth=2)
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
      ZLAMBDA list of (word_tuple, kind) → coeff dicts
      Z/p     list of [(word_tuple, coeff), ...]

    Methods
    -------
    augmentations(grading_mod)  List[Augmentation], cached per grading_mod
    lin_hom(grading_mod)        List[Dict[int,int]], cached per grading_mod
    check_d_squared()           bool  (Z/2 and Z[λ] only)
    aug_count()                 float  normalized augmentation number (Z/2, Z/p)
    print_differential()        print d(a[i]) for each generator
    """

    def __init__(self, leg: Leg, ring: GroundRing) -> None:
        self.leg = leg
        self.ring = ring
        self._differential = None
        self._augmentations_cache: Dict[int, List[Augmentation]] = {}
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
        where kind is 'int' (plain integer) or 'lambda' (λ coefficient).
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
            if not b:
                return []
            p = self.leg.num_cusps
            gr_b = [Leg._trugrad(b, i) for i in range(1, len(b) + 1)]
            gr_b.extend([1] * p)
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
            for i in range(n - p + 1, n + 1):
                coeff = 'lambda' if i == n else 1
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
                if coeff == 'lambda':
                    k = (key_word, 'lambda')
                    poly[k] = poly.get(k, 0) + 1
                else:
                    k = (key_word, 'int')
                    poly[k] = poly.get(k, 0) + coeff
            result.append({k: v for k, v in poly.items() if v != 0})
        return result

    @property
    def differential(self):
        """The DGA differential, computed once and cached. Format depends on self.ring."""
        if self._differential is not None:
            return self._differential

        if self.leg.num_components != 1:
            raise ValueError("the DGA differential is only defined for knots, not links")

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
            # Z/n: lift through Z[λ] then reduce mod n with λ = -1
            n = self.ring.modulus
            zd = self._z_diff()
            result = []
            for poly_dict in zd:
                terms: Dict[tuple, int] = {}
                for (word, kind), coeff in poly_dict.items():
                    c = coeff if kind == 'int' else -coeff
                    c %= n
                    if c:
                        terms[word] = (terms.get(word, 0) + c) % n
                result.append([(w, c) for w, c in terms.items() if c])
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

            def to_triples(poly_dict):
                return [(word, 1 if kind == 'lambda' else 0, coeff)
                        for (word, kind), coeff in poly_dict.items()]

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
                            key = (new_word, lam + d_lam)
                            count[key] = count.get(key, 0) + coeff * right_sign * d_coeff
                if any(v != 0 for v in count.values()):
                    return False
            return True

        raise NotImplementedError(
            f'check_d_squared not yet implemented for {self.ring}'
        )

    def augmentations(self, grading_mod: int = 0) -> List['Augmentation']:
        """
        All augmentations of this DGA over self.ring.
        Cached per grading_mod.

        grading_mod: 0 = Z-graded, 1 = ungraded, n >= 2 = Z/n-graded.
        Not supported for Z[λ] (use Z/2 or Z/p instead).
        """
        if self.ring == GroundRing.ZLAMBDA:
            raise NotImplementedError(
                'Augmentations are not computed directly over Z[λ]. '
                'Use leg.dga(GroundRing.Z2) or leg.dga(GroundRing.Zn(p)).'
            )
        if grading_mod not in self._augmentations_cache:
            if self.leg.num_components != 1:
                raise ValueError("augmentations are only defined for knots, not links")
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
                if grading_mod == 0 and r != 0:
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
                    return [], [], [], []
                c0, c1 = get_c0(work), get_c1(work)
                while c0 or c1:
                    if set(c0) & set(c1):
                        return [], [], [], []
                    work = trim(sub(rmv(work, c0), c1))
                    aQ_l = [x for x in aQ_l if x not in c0 and x not in c1]
                    a1_local = sorted(set(a1_local) | set(c1))
                    a0_l = sorted(set(a0_l) | set(c0))
                    if is_impossible(work):
                        return [], [], [], []
                    c0, c1 = get_c0(work), get_c1(work)
                return aQ_l, a1_local, a0_l, work

            def aug_q(dd, s):
                s_set = set(s)
                for poly in dd:
                    subbed = [[x for x in m if x not in s_set] for m in poly]
                    if sum(1 for m in subbed if not m) % 2 != 0:
                        return False
                return True

            def aug_zn(zn_d, grading_mod, n):
                g_eff = [x % (2 * r) for x in gr] if (grading_mod == 0 and r != 0) else gr[:]

                def is_grade0(x):
                    return x == 0 if grading_mod == 0 else x % grading_mod == 0

                def is_grade1(x):
                    if grading_mod == 0: return x == 1
                    if grading_mod == 1: return True
                    return x % grading_mod == 1

                grade0_gens = [i + 1 for i, x in enumerate(g_eff) if is_grade0(x)]
                grade1_gens = [i + 1 for i, x in enumerate(g_eff) if is_grade1(x)]
                grade0_set = set(grade0_gens)
                conditions = [
                    [(w, c) for w, c in zn_d[g1 - 1] if all(gi in grade0_set for gi in w)]
                    for g1 in grade1_gens
                ]

                def eval_zn(poly, augm_dict):
                    total = 0
                    for word, coeff in poly:
                        term = coeff
                        for gen in word:
                            term = (term * augm_dict.get(gen, 0)) % n
                        total = (total + term) % n
                    return total

                result = []
                for values in product(range(n), repeat=len(grade0_gens)):
                    augm_dict = dict(zip(grade0_gens, values))
                    if all(eval_zn(cond, augm_dict) == 0 for cond in conditions):
                        result.append(augm_dict)
                return result

            if modulus == 2:
                aQ, a1, a0, d_red = ac(d, grading_mod)
                if not aQ and not a1:
                    raw = []
                else:
                    raw = []
                    for sz in range(len(aQ) + 1):
                        for subset in combinations(aQ, sz):
                            s = sorted(a1 + list(subset))
                            if aug_q(d_red, s):
                                raw.append(s)
            else:
                raw = aug_zn(d, grading_mod, modulus)
            self._augmentations_cache[grading_mod] = [
                Augmentation(a, self) for a in raw
            ]
        return self._augmentations_cache[grading_mod]

    def lin_diff(self, augmentation: 'Augmentation', n: int = 1) -> List[List[List[int]]]:
        """
        Degree-n part of the augmented differential with respect to augmentation.

        Applies augmentation to self.differential, expanding each generator that
        is sent to 1, then keeps only monomials of output word length n.

        n=1 (default) gives the linearized differential — the map one would
        normally call the linearized differential in contact homology.
        n=2 gives the quadratic part, used for cup products.
        """
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
        def nullity_f2(mat):
            if not mat or not mat[0]:
                return len(mat[0]) if mat else 0
            return len(mat[0]) - _rank_f2(mat)

        gs = self._gens_spaces
        num = len(gs)
        if num == 1:
            return len(gs[0])
        if place == 1:
            return nullity_f2(self._diff_matrix_f2(1, ld))
        if place == num:
            return len(gs[num - 1]) - _rank_f2(self._diff_matrix_f2(num - 1, ld))
        return (nullity_f2(self._diff_matrix_f2(place, ld)) -
                _rank_f2(self._diff_matrix_f2(place - 1, ld)))

    def _dim_homology_zn(self, place: int, augmentation: 'Augmentation') -> int:
        """Dimension of Z/n linearized homology at grading position place (1-indexed in _gens_spaces)."""
        zn_d = self.differential
        augm_dict = augmentation.data
        n = self.ring.modulus
        gs = self._gens_spaces

        def rref_zn(mat):
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

        def rank_zn(mat):
            if not mat or not mat[0]:
                return 0
            _, pivots = rref_zn(mat)
            return len(pivots)

        def nullity_zn(mat):
            if not mat or not mat[0]:
                return len(mat[0]) if mat else 0
            return len(mat[0]) - rank_zn(mat)

        def diff_matrix_zn(which):
            domain = gs[which - 1]
            rng = gs[which]
            rng_idx = {g: i for i, g in enumerate(rng)}
            mat = [[0] * len(domain) for _ in range(len(rng))]
            for col, gen_d in enumerate(domain):
                for word, coeff in zn_d[gen_d - 1]:
                    for j, a in enumerate(word):
                        if a in rng_idx:
                            prod = coeff
                            for li, b_gen in enumerate(word):
                                if li != j:
                                    prod = (prod * augm_dict.get(b_gen, 0)) % n
                            mat[rng_idx[a]][col] = (mat[rng_idx[a]][col] + prod) % n
            return mat

        num = len(gs)
        if num == 1:
            return len(gs[0])
        if place == 1:
            return nullity_zn(diff_matrix_zn(1))
        if place == num:
            return len(gs[num - 1]) - rank_zn(diff_matrix_zn(num - 1))
        return nullity_zn(diff_matrix_zn(place)) - rank_zn(diff_matrix_zn(place - 1))

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

    def all_lin_hom(self, grading_mod: int = 0, format: bool = False):
        """
        Set of distinct Poincaré-Chekanov polynomials over all augmentations.
        Cached per grading_mod.  Not supported for Z[λ].
        Returns List[Dict[int,int]], or List[str] if format=True.
        """
        if self.ring == GroundRing.ZLAMBDA:
            raise NotImplementedError('all_lin_hom not implemented over Z[λ]')
        if grading_mod not in self._lin_hom_cache:
            augms = self.augmentations(grading_mod=grading_mod)
            gs = self._gens_spaces
            max_g = max(self.leg.grading)
            seen, results = set(), []
            for augm_obj in augms:
                poly = {}
                for k in range(1, len(gs) + 1):
                    dim = self.dim_homology(k, augm_obj)
                    if dim:
                        poly[max_g - k + 1] = dim
                key = frozenset(poly.items())
                if key not in seen:
                    seen.add(key)
                    results.append((poly, augm_obj))
            results.sort(key=lambda pair: sorted(pair[0].items()))
            self._lin_hom_cache[grading_mod] = results
        pairs = self._lin_hom_cache[grading_mod]
        if format:
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

    def aug_count(self) -> float:
        """
        Normalized graded augmentation number (Z-graded, Ng's normalization):
        |ring|^((−1−χ)/2) × |Aug|.
        Not supported for Z[λ].
        """
        if self.ring == GroundRing.ZLAMBDA:
            raise NotImplementedError('aug_count not defined over Z[λ]')
        gr = self.leg.grading
        n_aug = len(self.augmentations())
        min_g, max_g = min(gr), max(gr)
        chi = sum(((-1) ** k) * gr.count(k) for k in range(0, max_g + 1))
        chi += sum(((-1) ** (k + 1)) * gr.count(k) for k in range(min_g, 0))
        exp = (-1 - chi) / 2
        n = self.ring.modulus
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
            for i, poly_dict in enumerate(cast(List[Dict[Any, Any]], self.differential)):
                if not poly_dict:
                    entry = "0"
                else:
                    terms = []
                    for (word, kind), coeff in poly_dict.items():
                        if coeff == 0:
                            continue
                        mon = " * ".join(f"a[{k}]" for k in word) if word else "1"
                        if kind == 'lambda':
                            s = f"λ * {mon}" if mon != "1" else "λ"
                        else:
                            s = (mon if coeff == 1 else f"-{mon}" if coeff == -1
                                 else f"{coeff}*{mon}")
                        terms.append(s)
                    entry = " + ".join(terms) if terms else "0"
                print(f'  d(a[{i + 1}]) = {entry}')
        else:
            for i, terms in enumerate(self.differential):
                s = ' + '.join(
                    ('' if c == 1 else f'{c}*') + (
                        ' * '.join(f'a[{g}]' for g in w) if w else '1'
                    )
                    for w, c in terms
                ) or '0'
                print(f'  d(a[{i + 1}]) = {s}')


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
              Z/2 : List[int]      generators sent to 1 (others → 0)
              Z/p : Dict[int,int]  generator → value in {0, ..., p−1}
    dga     the parent DGA

    Properties  (computed once, cached)
    ------------------------------------
    lin_hom          Dict[int, int]  Poincaré-Chekanov polynomial
    cohomology_basis                 (Z/2 only) basis for linearized cohomology
    double_products                  (Z/2 only) cup-product table
    """

    def __init__(self, data, dga: DGA) -> None:
        self.data = data
        self.dga = dga

    def __repr__(self) -> str:
        return f'Augmentation({self.data!r}, ring={self.dga.ring!r})'

    @cached_property
    def lin_hom(self) -> Dict[int, int]:
        """
        Poincaré-Chekanov polynomial of linearized contact homology w.r.t. this ε.
        Returns {grading: dimension} (zero dimensions omitted).
        """
        ring = self.dga.ring
        if ring == GroundRing.ZLAMBDA:
            raise NotImplementedError('lin_hom not implemented over Z[λ]')
        gs = self.dga._gens_spaces
        max_g = max(self.dga.leg.grading)
        result = {}
        for k in range(1, len(gs) + 1):
            dim = self.dga.dim_homology(k, self)
            if dim:
                result[max_g - k + 1] = dim
        return result

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
        """Basis for linearized cohomology (Z/2 only)."""
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
                ker_vecs = null_space_f2(mat_k) if (mat_k and mat_k[0]) else (
                    [[1 if i == j else 0 for j in range(len(space))]
                     for i in range(len(space))])
                ker_basis = ker_vecs if ker_vecs else (
                    [[1 if i == j else 0 for j in range(len(space))]
                     for i in range(len(space))])
            cohom = [[space[i] for i, v in enumerate(vec) if v] for vec in ker_basis]
            result.append([c for c in cohom if c])
        return result

    @cached_property
    def double_products(self):
        """Cup-product multiplication table on linearized cohomology (Z/2 only)."""
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
}


# ============================================================
# Section 7: Constructions
# ============================================================

def whitehead_double(leg: Leg) -> Leg:
    """
    Legendrian Whitehead double of leg.
    Returns a new Leg whose braid encodes the Whitehead double plat closure.
    """
    p = leg.num_cusps
    pp = [4 * i - 2 for i in range(1, p + 1)]
    dc = [x for gen in leg.braid for x in (2*gen, 2*gen + 1, 2*gen - 1, 2*gen)]
    return Leg([2] + pp + dc + pp, name=f'WhiteheadDouble({leg.name})')


def twisted_2cable(leg: Leg) -> Leg:
    """
    Legendrian twisted 2-cable of leg.
    Returns a new Leg whose braid encodes the twisted 2-cable plat closure.
    """
    p = leg.num_cusps
    pp = [4 * i - 2 for i in range(1, p + 1)]
    dc = [x for gen in leg.braid for x in (2*gen, 2*gen + 1, 2*gen - 1, 2*gen)]
    return Leg(pp + [1] + dc + pp, name=f'Twisted2Cable({leg.name})')
