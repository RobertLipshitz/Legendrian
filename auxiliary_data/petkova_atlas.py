"""
petkova_atlas.py

Loader for the Petkova et al. Legendrian knot atlas.

Source
------
https://github.com/ipetkova/LegendrianAtlas
(I. Petkova et al., "A Legendrian Knot Atlas", 2023)

License note: the upstream repository has no explicit license as of May 2026.
Use is with the permission of the author (I. Petkova, personal communication,
May 19, 2026).

The atlas is kept separate from grid_atlas.py / ATLAS in legendrian.py because
it uses different knot-naming conventions and is less established in the
mathematical literature.

Data files (in petkova_data/)
------------------------------
layer1.csv  -- 2696 maximal-tb Legendrian representatives with actual grid
               diagrams (X/O permutations) and classical invariants.
layer2.csv  -- 2219 first stabilisations: tb, r, parent G-IDs.  No grid data.
layer3.csv  -- 2888 second stabilisations: tb, r, parent S-IDs.  No grid data.

Knot naming
-----------
Uses Hoste-Thistlethwaite notation: '8a1', 'm8a2', '11n38', etc.
  <crossing_count> a|n <index>   (a = alternating, n = non-alternating)
  m prefix = mirror image
This differs from the Rolfsen-style 'K8_19' names used in legendrian.py.

Grid conventions (layer 1)
--------------------------
X_perm and O_perm are 0-indexed tuples (rows counted from the bottom).
Column j has the X marker at row X_perm[j] and the O marker at row O_perm[j].
Vertical strands pass over horizontal strands.  This matches the convention
expected by Leg((X_perm, O_perm)).

Relation fields ('Legendrian Mirror', 'Orientation Reversal', …)
-----------------------------------------------------------------
Values are Grid IDs of related entries, or None if the relation is absent
('--') or unknown ('?') in the source data.
"""

import ast
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

_DATA_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Layer1Entry:
    knot: str                              # e.g. '8a1', 'm10n3'
    grid_id: str                           # e.g. 'G1011'
    x_perm: Tuple[int, ...]               # 0-indexed
    o_perm: Tuple[int, ...]               # 0-indexed
    tb: int
    r: int
    leg_mirror: Optional[str]             # Grid ID of Legendrian mirror, or None
    orient_rev: Optional[str]             # Grid ID after orientation reversal, or None
    leg_mirror_orient_rev: Optional[str]  # Grid ID of mirror of orient-reversal, or None


@dataclass
class Layer2Entry:
    knot: str
    grid_id: str          # 'S####'
    tb: int
    r: int
    parents: List[str]    # Layer-1 Grid IDs that stabilise to this entry


@dataclass
class Layer3Entry:
    knot: str
    grid_id: str          # 'T####'
    tb: int
    r: int
    parents: List[str]    # Layer-2 Grid IDs that stabilise to this entry


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _relation(value: str) -> Optional[str]:
    """Convert a relation field value to a Grid ID or None."""
    v = value.strip()
    if v in ('--', '?', ''):
        return None
    return v


def _parse_perm(s: str) -> Tuple[int, ...]:
    """Parse '(0, 9, 2, 1, ...)' into a tuple of ints."""
    return tuple(ast.literal_eval(s))


def _parse_parents(s: str) -> List[str]:
    """Parse a possibly comma-separated list of Grid IDs."""
    return [p.strip() for p in s.split(',') if p.strip()]


# ---------------------------------------------------------------------------
# Iterators
# ---------------------------------------------------------------------------

def layer1() -> Iterator[Layer1Entry]:
    """Yield every entry from layer1.csv as a Layer1Entry."""
    with open(_DATA_DIR / 'layer1.csv', encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            yield Layer1Entry(
                knot=row['Knot Type'],
                grid_id=row['Grid ID'],
                x_perm=_parse_perm(row['X-Permutation']),
                o_perm=_parse_perm(row['O-Permutation']),
                tb=int(row['tb']),
                r=int(row['r']),
                leg_mirror=_relation(row['Legendrian Mirror']),
                orient_rev=_relation(row['Orientation Reversal']),
                leg_mirror_orient_rev=_relation(
                    row['Legendrian Mirror of Orientation Reversal']
                ),
            )


def layer2() -> Iterator[Layer2Entry]:
    """Yield every entry from layer2.csv as a Layer2Entry."""
    with open(_DATA_DIR / 'layer2.csv', encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            yield Layer2Entry(
                knot=row['Knot'],
                grid_id=row['Grid ID'],
                tb=int(row['tb']),
                r=int(row['r']),
                parents=_parse_parents(row['Parents']),
            )


def layer3() -> Iterator[Layer3Entry]:
    """Yield every entry from layer3.csv as a Layer3Entry."""
    with open(_DATA_DIR / 'layer3.csv', encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            yield Layer3Entry(
                knot=row['Knot'],
                grid_id=row['Grid ID'],
                tb=int(row['tb']),
                r=int(row['r']),
                parents=_parse_parents(row['Parents']),
            )


# ---------------------------------------------------------------------------
# Convention converter
# ---------------------------------------------------------------------------

def to_leg_grid(entry: Layer1Entry) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    Convert a Layer1Entry to the (X_perm, O_perm) convention used by Leg().

    The Petkova CSV files index rows from the **top** (row 0 = top), while
    Leg((X_perm, O_perm)) expects rows indexed from the **bottom** (row 0 =
    bottom).  This function applies the map  v  →  n − 1 − v  to every
    element of both permutations, where n is the grid size.

    The rotation number of the resulting Leg object matches abs(entry.r) but
    the sign may differ (orientation of the knot is not fixed by the grid).

    Usage::

        for e in layer1():
            x, o = to_leg_grid(e)
            leg = Leg((list(x), list(o)))
            assert leg.tb == e.tb
            assert abs(leg.rot) == abs(e.r)
    """
    n = len(entry.x_perm)
    x = tuple(n - 1 - v for v in entry.x_perm)
    o = tuple(n - 1 - v for v in entry.o_perm)
    return x, o


# ---------------------------------------------------------------------------
# Lookup tables (built lazily)
# ---------------------------------------------------------------------------

_layer1_by_id: Optional[Dict[str, Layer1Entry]] = None
_layer1_by_knot: Optional[Dict[str, List[Layer1Entry]]] = None


def layer1_by_id() -> Dict[str, Layer1Entry]:
    """Return a dict mapping Grid ID → Layer1Entry (built once, then cached)."""
    global _layer1_by_id
    if _layer1_by_id is None:
        _layer1_by_id = {e.grid_id: e for e in layer1()}
    return _layer1_by_id


def layer1_by_knot() -> Dict[str, List[Layer1Entry]]:
    """Return a dict mapping knot name → list of Layer1Entry (cached)."""
    global _layer1_by_knot
    if _layer1_by_knot is None:
        result: Dict[str, List[Layer1Entry]] = {}
        for e in layer1():
            result.setdefault(e.knot, []).append(e)
        _layer1_by_knot = result
    return _layer1_by_knot
