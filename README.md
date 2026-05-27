# Legendrian Knot Invariants

Python library for computing invariants of Legendrian knots and links in contact topology. Knots are encoded as plat-closure braid words and the library computes classical invariants (tb, rot), the contact-homology DGA (over Z/2, Z[λ], or Z/p), augmentations, linearized homology, and rulings.

The algorithms originate in a Mathematica notebook by Paul Melvin, extended by Kirk Mangels, Alden Walker, Lenny Ng, Josh Sabloff, and Sumana Shrestha.

## Files

| File | Description |
|------|-------------|
| `legendrian.py` | Primary API |
| `test_legendrian.py` | Test suite (654 tests) |
| `auxiliary_data/grid_atlas.py` | XO grid-diagram representatives from the [Ng Legendrian atlas](https://sites.math.duke.edu/~ng/atlas/) |
| `auxiliary_data/link_grid_atlas.py` | XO grid-diagram representatives for 2-component Legendrian links (Chongchitmate–Ng) |
| `auxiliary_data/petkova_atlas.py` | Loader for the [Petkova et al. atlas](https://github.com/ipetkova/LegendrianAtlas) CSV data |
| `auxiliary_data/layer{1,2,3}.csv` | Downloaded CSV files from the Petkova atlas |

## Quick Start

```python
from legendrian import Leg, GroundRing

# Construct from braid word or atlas name
k = Leg([2, 2, 2])       # Legendrian trefoil
k = Leg('mK3_1')         # same, from atlas

# Classical invariants
print(k.tb, k.rot)        # 1, 0

# DGA over Z/2 and augmentations
d = k.dga()
for aug in d.augmentations():
    print(aug.lin_hom)    # {grading: dimension}

# DGA over Z[λ]
d2 = k.dga('Zlambda')
d2.print_differential()
d2.check_d_squared()

# Rulings
print(k.ruling_invariant())             # Z-graded ruling polynomial as {degree: coeff}
print(k.ruling_invariant(grading_mod=2))  # Z/2-graded ruling polynomial
print(k.format_ruling_invariant())      # ruling polynomial as a string in z
```

## Input Format

`Leg` accepts four input forms.

**Braid word** — a list of positive integers encoding the plat closure of a positive braid. The integer `i` represents σ_i, a positive crossing between strands `i` and `i+1` (1-indexed from the top). All left cusps share one x-coordinate and all right cusps share another.

```python
Leg([2, 2, 2])      # standard Legendrian right-handed trefoil
```

**Atlas name** — a string key into the built-in `ATLAS` of pre-computed representatives.

```python
Leg('mK3_1')        # mirror trefoil (unique representative)
Leg('mK5_2.0')      # first of two reps of mirror 5_2 (0-indexed)
```

**Tangle decomposition** — a list of tuples describing a general front diagram as a left-to-right sequence of elementary moves:

| Tuple | Meaning |
|-------|---------|
| `('<', h)` | Left cusp at height `h` |
| `('>', h)` | Right cusp at height `h` |
| `('X', h)` | Positive crossing between strands `h` and `h+1` |

Heights are 0-indexed from the top and refer to the strand configuration *at that moment* (the strand count changes as cusps are added or removed). The code converts the tangle to plat form internally by applying Legendrian Reidemeister II moves, so any valid front-diagram decomposition is accepted, not just plat-ordered ones.

```python
# Trefoil described as a tangle (left cusps first, then crossings, then right cusps)
Leg([('<', 0), ('<', 0),
     ('X', 1), ('X', 2), ('X', 0), ('X', 2), ('X', 0), ('X', 2), ('X', 1),
     ('>', 0), ('>', 0)])

# Same knot with cusps interleaved among crossings — algorithm rearranges it
Leg([('<', 0), ('X', 0), ('X', 0), ('<', 0), ('>', 0), ('>', 0)])
```

The original tangle is stored as `self.tangle`. The `name` keyword argument works the same way as for the other input forms.

**Grid diagram** — a 2-tuple of permutations `(X_perm, O_perm)`. `X_perm[j]` is the row of the X marker in column `j`; `O_perm[j]` is the row of the O marker in column `j`. Rows are 0-indexed from the **bottom**; columns are 0-indexed from the left.

Conventions:
- Vertical strands (in each column) pass **over** horizontal strands (in each row) at each crossing.
- The Legendrian front projection is obtained by rotating the grid **45° counter-clockwise**. Under this rotation, horizontal segments become slope +1 arcs and vertical segments become slope −1 arcs. The markers become cusps or smooth kink points.
- Left cusps arise where both the horizontal and vertical strand leave the marker going rightward (in the rotated diagram). Right cusps arise where both arrive from the left. Points where the strand transitions from slope −1 to slope +1 (or vice versa) without a cusp are kinks, handled internally.

The code produces a tangle sequence from the grid and then converts it to plat form via Legendrian Reidemeister moves. Both `self.grid` and `self.tangle` are stored.

```python
Leg(([1, 0], [0, 1]))           # 2×2 grid: Legendrian unknot, tb = -1
Leg(([0, 1, 2], [1, 2, 0]))     # 3×3 grid: once-stabilised unknot, tb = -2
```

## Classes

### `Leg`

A Legendrian knot or link.

```python
Leg([2,2,2])                          # from braid word (list of positive ints)
Leg('mK3_1')                          # unique atlas entry
Leg('mK3_1.0')                        # also works
Leg('mK5_2.0')                        # first of two reps, 0-indexed
Leg([('<', 0), ('X', 0), ('>', 0)])   # from tangle decomposition
Leg(([1, 0], [0, 1]))                 # from grid diagram (2-tuple of permutations)
Leg([1, 3], maslov=[0, 0])            # 2-component link with explicit Maslov potential
```

The optional `maslov` parameter is a list of integers, one per link component (ordered by first appearance from the top of the front diagram). Each value sets the Maslov potential of the lower strand of the topmost left cusp on that component. For knots the default is `[0]`; for links it is `[0, 0, …]`. Grading, tb, and ruling invariants for links require a Maslov potential.

**Classical invariants** (computed once, cached):

| Property | Type | Description |
|----------|------|-------------|
| `num_components` | `int` | Number of link components |
| `strand_potentials` | `List[int]` | Maslov potential on braid strands (length `2 * num_cusps`) |
| `grading` | `List[int]` | Maslov grading of generators |
| `tb` | `int` | Thurston-Bennequin number |
| `rot` | `int` (knots) / `Tuple[int,...]` (links) | Rotation number; returns `int` for knots, a per-component tuple for links |
| `ruling_invariant(grading_mod=0)` | `Dict[int, int]` | Ruling polynomial; requires `grading_mod \| 2·rot_c` for each component (`grading_mod=0` needs `rot_c=0`; `grading_mod=1,2` always valid) |

**DGA and augmentations:**

| Method | Returns | Description |
|--------|---------|-------------|
| `dga(ring=None)` | `DGA` | DGA over `ring`; defaults to `DEFAULT_GROUND_RING` (Z/2). Cached per ring. Links supported over Z/2 only. |
| `augmentations(grading_mod, modulus)` | `List[Augmentation]` | Shorthand for `dga(Z/modulus).augmentations(grading_mod)`. All rings supported for links; Z/2 and Z/n require `grading_mod \| 2·rot_c`. |
| `all_lin_hom(grading_mod, modulus, format)` | `List[Dict[int,int]]` or `List[str]` | Distinct Poincaré-Chekanov polynomials; `format=True` returns strings. Links over Z/2 require same grading condition. |
| `rulings(grading_mod)` | `List[List[int]]` | All graded rulings; cached per `grading_mod` |
| `format_ruling_invariant(grading_mod=0)` | `str` | `ruling_invariant` as a polynomial string in z |

**Visualization:**

```python
k.draw()                            # front projection (plat form); returns matplotlib Figure
k.draw(label_generators=True)       # same, with generator labels
k.draw(method='tangle')             # tangle-sequence diagram (requires stored tangle)
k.draw(method='grid')               # XO grid diagram (requires stored grid)
k.draw(color=True)                  # color each strand/component (default color=False is black)
k.export_svg('knot.svg')            # writes SVG of the plat diagram, returns filename
```

### `DGA`

The contact-homology DGA of a `Leg` over a chosen `GroundRing`. Obtain via `Leg.dga(ring)`.

```python
d = k.dga()                    # Z/2
d = k.dga('Zlambda')           # Z[λ]
d = k.dga(GroundRing.Zn(3))    # Z/3
```

| Member | Description |
|--------|-------------|
| `differential` | Differential; format depends on ring. Lazy, cached. Supported for links over all rings. For Z[λ] and Z/n, kind `('lambda', c)` (0-indexed) marks the basepoint cusp of component c; Z/n coefficients are reduced mod n but λ_c terms remain symbolic. |
| `augmentations(grading_mod, lambda_values)` | All augmentations; cached by `(grading_mod, lambda_values)`. Links supported over all rings (Z/2 and Z/n require `grading_mod \| 2·rot_c`). For Z/n (n>2): default sends λ→−1 for knots; for links, searches over all units (Z/n)^× per component. Override via `lambda_values={comp: val, …}`. Each returned `Augmentation.data` includes `('lambda', c)` keys for Z/n links. |
| `all_lin_hom(grading_mod, format)` | Distinct Poincaré-Chekanov polynomials; `format=True` returns strings. Cached. |
| `lin_hom_reps(grading_mod)` | `(poly, representative_augmentation)` pairs, one per distinct polynomial. |
| `check_d_squared()` | Verify d²=0. Works for Z/2 and Z[λ₁,…,λₗ] including links. |
| `aug_count(grading_mod=0)` | Normalized augmentation number \|ring\|^(−χ\*\_ρ/2) × \|Aug\_ρ\| (Z/2 and Z/p only). `grading_mod=0`: Z-graded (χ\*\_0 uses the shifted Euler characteristic of [NS06]); `grading_mod=ρ` odd: χ\*\_ρ = Σ\_{k=0}^{ρ−1}(−1)^k a\_k; `grading_mod=ρ` even, ρ>0: raises `NotImplementedError`. |
| `print_differential()` | Print d(a[i]) for each generator. Displays λ₁, λ₂, … for multi-component links. |

The `grading_mod` parameter controls grading: `0` = Z-graded, `1` = ungraded, `n ≥ 2` = Z/n-graded.

### `Augmentation`

A single augmentation ε: (DGA generators) → (ground ring). Obtained via `DGA.augmentations()`.

| Member | Description |
|--------|-------------|
| `data` | Raw augmentation data (Z/2: list of generators sent to 1; Z/p: dict generator → value) |
| `dga` | The parent `DGA` |
| `lin_hom` | Poincaré-Chekanov polynomial `{grading: dim}`; cached |
| `format_poincare()` | `lin_hom` formatted as a polynomial string in t |
| `cohomology_basis` | Basis for linearized cohomology (Z/2 only); cached. Returns `List[List[List[int]]]`: outer list is one entry per grading space (highest grade first), middle list is the basis for that grade, inner list is the 1-indexed generator numbers whose F_2 coefficient is 1 in that basis vector. |
| `double_products` | Cup-product table `(table, basis)` (Z/2 only); cached |

### `GroundRing`

Coefficient ring for DGA computations.

```python
GroundRing.Z2           # Z/2 (default)
GroundRing.ZLAMBDA      # Z[λ]
GroundRing.Zn(5)        # Z/5 (p prime)
GroundRing('Z2')        # string constructor
GroundRing('Zlambda')
GroundRing('Z3')
```

The global default ring is `DEFAULT_GROUND_RING = GroundRing.Z2`. You can change it:
```python
import legendrian
legendrian.DEFAULT_GROUND_RING = GroundRing.Zn(5)
```

## Atlases

### Included knot atlas

`ATLAS` is a dict of pre-computed Legendrian representatives for knots through ~15 crossings, keyed by name:

```python
from legendrian import ATLAS

list(ATLAS)          # all available names
ATLAS['mK5_2']       # list of two braid words
```

**Naming convention:**

| Name | Meaning |
|------|---------|
| `K3_1` | Right-handed trefoil (Rolfsen table, unique rep) |
| `mK3_1` | Mirror of 3_1 (unique rep) |
| `mK5_2.0` | First rep of mirror of 5_2 (0-indexed) |
| `mK5_2.1` | Second rep |
| `K11n38` | Knot 11n38 (Hoste-Thistlethwaite table) |

Prefix `m` = mirror, `K` = knot, `L` = link. Legendrian index is 0-based; omit when there is only one representative.

Data sourced from the [Legendrian knot atlas](https://sites.math.duke.edu/~ng/atlas/).

### Link Grid Atlas

`auxiliary_data/link_grid_atlas.py` provides XO grid-diagram representatives for 2-component Legendrian links. It contains 108 link types (215 total representatives) covering links through 11 crossings, sourced from Cromwell diagrams in a link table compiled by Wutichai Chongchitmate (advised by Lenhard Ng), available from [Ng's webpage](https://sites.math.duke.edu/~ng/atlas/).

```python
from auxiliary_data.link_grid_atlas import LINK_GRID_ATLAS
from legendrian import Leg

list(LINK_GRID_ATLAS)              # all available names, e.g. 'L2a1', 'm(L4a1)', ...
LINK_GRID_ATLAS['L6a3']            # list of (X_perm, O_perm) pairs for L6a3

# Construct a Legendrian representative (first of four for L6a3)
grid = LINK_GRID_ATLAS['L6a3'][0]
lnk = Leg(grid, maslov=[0, 0])
print(lnk.num_components)          # 2
print(lnk.tb, lnk.rot)
```

Grid conventions are identical to those in `auxiliary_data/grid_atlas.py`: rows 0-indexed from the bottom, columns from the left, vertical strands over horizontal strands. The `maslov` parameter must be supplied explicitly when computing graded invariants.

**Naming convention:** `L2a1` = 2-component link with 2 crossings, alternating, first in table. Prefix `m(…)` = mirror. Names follow the Chongchitmate–Ng table.

### Petkova Atlas

`auxiliary_data/petkova_atlas.py` provides a read-only loader for the separate [Petkova et al. atlas](https://github.com/ipetkova/LegendrianAtlas), which covers 466 knot types with 2696 Legendrian representatives. It is kept separate because it uses different conventions and is less established in the literature.

```python
from auxiliary_data.petkova_atlas import layer1, layer1_by_knot, to_leg_grid
from legendrian import Leg

# Iterate all maximal-tb representatives
for entry in layer1():
    x, o = to_leg_grid(entry)          # converts row-indexing convention
    leg = Leg((list(x), list(o)))
    assert leg.tb == entry.tb          # always holds
    assert abs(leg.rot) == abs(entry.r)

# Look up by knot name (HT notation)
reps = layer1_by_knot()['8a1']        # list of Layer1Entry for 8a1
```

**Knot naming:** uses Hoste-Thistlethwaite notation (`8a1`, `m10n3`, `11n38`) rather than the Rolfsen-style `K8_19` names used in `ATLAS`.

**Convention note:** the CSV files index grid rows from the top (row 0 = top); `to_leg_grid` converts to the bottom-indexed convention expected by `Leg`. The sign of the rotation number may differ by a global orientation choice.

**Layers:** Layer 1 contains full grid data. Layers 2 and 3 contain stabilised representatives (only tb, r, and parent IDs — no grid diagrams).

## Constructions

Module-level functions that build a new `Leg` from an existing one:

| Function | Description |
|----------|-------------|
| `whitehead_double(leg)` | Legendrian Whitehead double of `leg` |
| `twisted_2cable(leg)` | Legendrian twisted 2-cable of `leg` |
| `two_copy(leg)` | Legendrian 2-copy (push-off): 2-component link of `leg` and its push-off; `maslov=[0,0]` |
| `two_copy_shifted(leg)` | Same as `two_copy` but with `maslov=[1,0]` |

```python
from legendrian import Leg, whitehead_double, twisted_2cable, two_copy, two_copy_shifted

k = Leg('mK3_1')
wd = whitehead_double(k)        # Leg('WhiteheadDouble(mK3_1)')
tc = twisted_2cable(k)          # Leg('Twisted2Cable(mK3_1)')
lk = two_copy(k)                # 2-component link; lk(K, K') = tb(k)
lk2 = two_copy_shifted(k)       # same link, maslov shifted by 1 on first component
print(wd.tb, wd.rot)
```

## Examples

### Trefoil

```python
from legendrian import Leg

k = Leg('mK3_1')
print(k.tb, k.rot)         # 1, 0
print(k.num_components)    # 1

d = k.dga()
print(d.check_d_squared()) # True

augs = d.augmentations()
print(len(augs))           # 10
for a in augs:
    print(a.lin_hom)       # each: {1: 1, 0: 2}
```

### Figure-eight knot

```python
k = Leg('K4_1')
print(k.tb, k.rot)         # -3, 0

d = k.dga()
print(d.aug_count())

print(k.rulings())
print(k.ruling_invariant())
```

### Z[λ] differential

```python
k = Leg([2, 2, 2])
d = k.dga('Zlambda')
d.print_differential()
print(d.check_d_squared())
```

### Legendrian link with Maslov potential

```python
from legendrian import Leg

# 2-component link from the link atlas; maslov sets one potential per component
lnk = Leg('L2a1.0', maslov=[0, 0])
print(lnk.num_components)          # 2
print(lnk.strand_potentials)       # Maslov potential on each braid strand
print(lnk.grading)                 # generator gradings (depends on maslov choice)
print(lnk.tb)                      # total writhe
print(lnk.rot)                     # tuple of per-component rotation numbers
print(lnk.ruling_invariant())      # graded ruling polynomial

# Z[λ₁,λ₂] differential — one λ per component
d = lnk.dga('Zlambda')
d.print_differential()             # displays λ₁, λ₂ for the two components
print(d.check_d_squared())         # True
```

Shifting a component's seed by `k` shifts that component's crossing gradings by `k` for crossings between different components:

```python
lnk0 = Leg('L2a1.0', maslov=[0, 0])
lnk1 = Leg('L2a1.0', maslov=[0, 2])   # component 1 shifted by 2
```

### Multiple Legendrian reps

```python
from legendrian import Leg, ATLAS

for i in range(len(ATLAS['mK5_2'])):
    k = Leg(f'mK5_2.{i}')
    print(k.name, k.tb, k.rot, k.all_lin_hom())
```

## Running the Tests

```bash
python3 -m pytest test_legendrian.py
```

## Dependencies

- Python 3.8+
- `sympy` (for Z[λ] polynomial arithmetic)
- `matplotlib` (for `draw` and `export_svg`)

## Related Projects

[legendrian_links](https://github.com/RAvdek/legendrian_links) (Avdek et al.) is a Python project with overlapping goals and a similar plat-diagram approach. It focuses on augmentation enumeration, bilinearized homology, and planar diagram algebras (RSFT), and includes a web application for interactive exploration. It also supports LCH differentials over the integers.

## Acknowledgments

This is a conversion and extension of a Mathematica notebook by Paul Melvin, Kirk Mangels, Alden Walker, Lenny Ng, Josh Sabloff, and Sumana Shrestha. The conversion was performed by Anthropic's Claude chatbot, with assistance from Robert Lipshitz, and was supported by U.S. National Science Foundation grant DMS-2505715. RL thanks Lenhard Ng for helpful discussions about this code and the underlying mathematics.

## References

- [CEKSW11] G. Civan, P. Koprowski, J. Etnyre, J. M. Sabloff, and A. Walker, "Product structures for Legendrian contact homology," *Math. Proc. Cambridge Philos. Soc.* **150** (2011), no. 2, 291–311. [DOI:10.1017/S0305004110000460](https://doi.org/10.1017/S0305004110000460)
- [Ch02] Yu. Chekanov, "Differential algebra of Legendrian links," *Invent. Math.* **150** (2002), no. 3, 441–483.
- [CP05] Yu. Chekanov and P. Pushkar, "Combinatorics of Legendrian links and the Arnol'd 4-conjectures," *Uspekhi Mat. Nauk* **60** (2005), no. 1, 99–154; translated in *Russian Math. Surveys* **60** (2005), no. 1, 95–149.
- [ENS02] J. B. Etnyre, L. L. Ng, and J. M. Sabloff, "Invariants of Legendrian knots and coherent orientations," *J. Symplectic Geom.* **1** (2002), no. 2, 321–367. [DOI:10.4310/jsg.2001.v1.n2.a5](https://doi.org/10.4310/jsg.2001.v1.n2.a5)
- [Fu03] D. Fuchs, "Chekanov–Eliashberg invariant of Legendrian knots: existence of augmentations," *J. Geom. Phys.* **47** (2003), no. 1, 43–65.
- [Ng03] L. Ng, "Computable Legendrian invariants," *Topology* **42** (2003), no. 1, 55–82.
- [NS06] L. L. Ng and J. M. Sabloff, "The correspondence between augmentations and rulings for Legendrian knots," *Pacific J. Math.* **224** (2006), no. 1, 141–150. [DOI:10.2140/pjm.2006.224.141](https://doi.org/10.2140/pjm.2006.224.141)

## License

This project is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.html) (GPL-3.0).

You are free to use, modify, and distribute this software under the terms of the GPL v3. Any derivative work or software that incorporates this code must also be released under the GPL v3.