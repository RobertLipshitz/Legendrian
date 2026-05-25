"""
test_legendrian.py

Tests for legendrian.py (the object-oriented interface).

Run with:  pytest test_legendrian.py
Single:    pytest test_legendrian.py::TestClassicalInvariants::test_trefoil_tb
"""

import copy
import pytest

from legendrian import (
    Leg, DGA, Augmentation, GroundRing, DEFAULT_GROUND_RING, ATLAS,
)


# ---------------------------------------------------------------------------
# Knots used across tests  (same braid words as test_legendrian_invariants.py)
# ---------------------------------------------------------------------------
TREFOIL   = [2, 2, 2]                               # m(3_1), tb=1, rot=0
FIG_EIGHT = [2, 1, 4, 5, 1, 1, 3, 5, 1, 2, 5, 4]  # 4_1,   tb=-3, rot=0

# Chekanov pair: two Legendrian m(5_2) with the same tb=1, rot=0
M52_A = [2, 3, 1, 3, 4, 5, 1, 2, 3, 4, 2, 4]
M52_B = [2, 3, 1, 3, 1, 2, 2, 3, 1, 3, 2]

# 2-component link (σ_1 and σ_3 act on disjoint strand pairs)
LINK_2 = [1, 3]

# Hopf link: maslov=[1,0] gives grading [0,0,1,1]; has Z/3 augmentations
HOPF = [2, 2]
HOPF_MASLOV = [1, 0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _aug_is_valid(augm: Augmentation) -> bool:
    """Re-verify ε∘∂ = 0 over Z/2 using the augmentation's parent DGA.
    Only correct for Z/2 augmentations: augm.data is List[int] there.
    For Z/n, data is a Dict and this helper silently gives wrong results."""
    d = augm.dga.differential
    eps_set = set(augm.data)   # List[int] for Z/2
    for poly in d:
        total = sum(1 for mono in poly if all(x in eps_set for x in mono))
        if total % 2 != 0:
            return False
    return True



# ---------------------------------------------------------------------------
# Section 0: Component counting and knot guard
# ---------------------------------------------------------------------------

class TestNumComponents:

    def test_trefoil_is_knot(self):
        assert Leg(TREFOIL).num_components == 1

    def test_fig_eight_is_knot(self):
        assert Leg(FIG_EIGHT).num_components == 1

    def test_m52_reps_are_knots(self):
        assert Leg(M52_A).num_components == 1
        assert Leg(M52_B).num_components == 1

    def test_two_component_link(self):
        assert Leg(LINK_2).num_components == 2

    def test_atlas_all_knots(self):
        """Every braid word in ATLAS should close to a knot (1 component)."""
        for name, braids in ATLAS.items():
            for b in braids:
                assert Leg(b).num_components == 1, \
                    f"Expected knot, got link for {name}: {b}"

    def test_z2_differential_works_for_link(self):
        # Z/2 differential is purely combinatorial and link-agnostic
        d = Leg(LINK_2, maslov=[0, 0]).dga().differential
        assert isinstance(d, list)

    def test_z2_differential_d_squared_zero_for_link(self):
        assert Leg(LINK_2, maslov=[0, 0]).dga().check_d_squared()

    def test_zn_differential_works_for_link(self):
        d = Leg(LINK_2, maslov=[0, 0]).dga('Z3').differential
        assert isinstance(d, list) and len(d) > 0

    def test_augmentations_z2_graded_works_for_link(self):
        # grading_mod=2: 2 | 2*rot_c = 2*1 always → valid for LINK_2
        result = Leg(LINK_2, maslov=[0, 0]).dga().augmentations(grading_mod=2)
        assert isinstance(result, list)

    def test_augmentations_raises_for_link_bad_grading(self):
        # grading_mod=0 (Z-graded) requires rot_c=0; LINK_2 has rot=(1,1)
        with pytest.raises(ValueError):
            Leg(LINK_2, maslov=[0, 0]).augmentations()

    def test_zlambda_diff_works_for_link(self):
        d = Leg(LINK_2, maslov=[0, 0]).dga('Zlambda').differential
        assert isinstance(d, list) and len(d) > 0

    def test_rulings_works_for_link(self):
        result = Leg(LINK_2, maslov=[0, 0]).rulings(grading_mod=1)
        assert isinstance(result, list)

    def test_rulings_graded_works_for_link(self):
        result = Leg(LINK_2, maslov=[0, 0]).rulings(grading_mod=2)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Section 1: Classical invariants
# ---------------------------------------------------------------------------

class TestClassicalInvariants:

    def test_trefoil_grading(self):
        assert Leg(TREFOIL).grading == [0, 0, 0, 1, 1]

    def test_trefoil_tb(self):
        assert Leg(TREFOIL).tb == 1

    def test_trefoil_rot(self):
        assert Leg(TREFOIL).rot == 0

    def test_fig_eight_grading(self):
        gr = Leg(FIG_EIGHT).grading
        assert len(gr) == 15
        assert gr[:4] == [0, 1, -1, 0]

    def test_fig_eight_tb(self):
        assert Leg(FIG_EIGHT).tb == -3

    def test_fig_eight_rot(self):
        assert Leg(FIG_EIGHT).rot == 0

    def test_chekanov_pair_classical_invariants_agree(self):
        assert Leg(M52_A).tb == Leg(M52_B).tb == 1
        assert Leg(M52_A).rot == Leg(M52_B).rot == 0

    def test_grading_length(self):
        # trefoil: 3 crossings + 2 right cusps = 5 generators
        assert len(Leg(TREFOIL).grading) == 5
        # figure-eight: 12 crossings + 3 right cusps = 15 generators
        assert len(Leg(FIG_EIGHT).grading) == 15

    def test_properties_are_cached(self):
        k = Leg(TREFOIL)
        assert k.grading is k.grading
        assert k.tb == k.tb  # value stable across accesses

    def test_rot_for_link_is_tuple(self):
        assert isinstance(Leg(LINK_2).rot, tuple)

    def test_rot_for_link_values(self):
        assert Leg(LINK_2, maslov=[0, 0]).rot == (1, 1)


# ---------------------------------------------------------------------------
# Section 1b: Maslov potentials and link grading
# ---------------------------------------------------------------------------

class TestMaslovPotentials:

    def test_trefoil_strand_potentials_default(self):
        # Left cusps give mu[0]=mu[1]+1 and mu[2]=mu[3]+1; right cusp
        # propagates mu[2]=mu[1] and mu[3]=mu[0]-2.  With seed 0: [1,0,0,-1].
        assert Leg(TREFOIL).strand_potentials == [1, 0, 0, -1]

    def test_trefoil_strand_potentials_shifted(self):
        assert Leg(TREFOIL, maslov=[5]).strand_potentials == [6, 5, 5, 4]

    def test_trefoil_grading_unchanged_by_maslov_shift(self):
        # Shifting the Maslov seed shifts all generator gradings uniformly;
        # for a knot the crossing grading is a *difference*, so it's unchanged.
        assert Leg(TREFOIL, maslov=[5]).grading == Leg(TREFOIL).grading

    def test_link_strand_potentials(self):
        # LINK_2 = [1,3]: two disjoint components {strands 0,1} and {strands 2,3}.
        # Left cusp constraints only: mu=[1,0,1,0].
        assert Leg(LINK_2, maslov=[0, 0]).strand_potentials == [1, 0, 1, 0]

    def test_link_grading(self):
        assert Leg(LINK_2, maslov=[0, 0]).grading == [1, 1, 1, 1]

    def test_link_grading_component_shift(self):
        # Shifting component 1's seed by k shifts that component's crossing by k.
        g0 = Leg(LINK_2, maslov=[0, 0]).grading
        g1 = Leg(LINK_2, maslov=[0, 3]).grading
        # Crossing 0 (gen=1) is on component 0 — unchanged.
        assert g1[0] == g0[0]
        # Crossing 1 (gen=3) is on component 1 — unchanged (difference of same-component potentials).
        assert g1[1] == g0[1]

    def test_maslov_wrong_length_raises(self):
        with pytest.raises(ValueError):
            Leg(TREFOIL, maslov=[0, 0]).strand_potentials  # knot needs length 1

    def test_maslov_wrong_length_link_raises(self):
        with pytest.raises(ValueError):
            Leg(LINK_2, maslov=[0]).strand_potentials  # link needs length 2


# ---------------------------------------------------------------------------
# Section 2: Differential
# ---------------------------------------------------------------------------

class TestDifferential:

    def test_trefoil_diff_crossings_zero(self):
        d = Leg(TREFOIL).dga().differential
        assert d[0] == []
        assert d[1] == []
        assert d[2] == []

    def test_trefoil_diff_cusp(self):
        """d(a4) = a1*a2*a3 + a3 + a1 + 1 (as a set of monomials, mod 2)."""
        d = Leg(TREFOIL).dga().differential
        terms = frozenset(tuple(m) for m in d[3])
        assert terms == frozenset([(1, 2, 3), (3,), (1,), ()])

    def test_trefoil_d_squared_zero(self):
        assert Leg(TREFOIL).dga().check_d_squared()

    def test_fig_eight_d_squared_zero(self):
        assert Leg(FIG_EIGHT).dga().check_d_squared()

    def test_m52a_d_squared_zero(self):
        assert Leg(M52_A).dga().check_d_squared()

    def test_m52b_d_squared_zero(self):
        assert Leg(M52_B).dga().check_d_squared()

    def test_diff_length(self):
        k = Leg(TREFOIL)
        assert len(k.dga().differential) == len(k.grading)

    def test_differential_is_cached(self):
        d = Leg(TREFOIL).dga()
        assert d.differential is d.differential


# ---------------------------------------------------------------------------
# Section 3: Augmentations
# ---------------------------------------------------------------------------

class TestAugmentations:

    def test_trefoil_aug_count(self):
        assert len(Leg(TREFOIL).augmentations()) == 5

    def test_trefoil_aug_set(self):
        augms = [frozenset(a.data) for a in Leg(TREFOIL).augmentations()]
        expected = [frozenset(s) for s in [[1], [3], [1, 2], [2, 3], [1, 2, 3]]]
        assert sorted(augms, key=sorted) == sorted(expected, key=sorted)

    def test_fig_eight_aug_count(self):
        assert len(Leg(FIG_EIGHT).augmentations()) == 4

    def test_aug_validity(self):
        """Every returned augmentation satisfies ε∘∂ = 0."""
        for augm in Leg(TREFOIL).augmentations():
            assert _aug_is_valid(augm)

    def test_aug_count_trefoil(self):
        assert Leg(TREFOIL).dga().aug_count() == 2.5

    def test_chekanov_pair_aug_counts_differ(self):
        assert len(Leg(M52_A).augmentations()) == 2
        assert len(Leg(M52_B).augmentations()) == 12

    def test_aug_only_grade_zero_generators(self):
        k = Leg(TREFOIL)
        gr = k.grading
        for augm in k.augmentations():
            for g in augm.data:
                assert gr[g - 1] == 0

    def test_augmentations_are_cached(self):
        d = Leg(TREFOIL).dga()
        assert d.augmentations() is d.augmentations()

    def test_augmentation_back_references(self):
        k = Leg(TREFOIL)
        d = k.dga()
        augm = d.augmentations()[0]
        assert augm.dga is d
        assert augm.dga.leg is k

    def test_trivial_augmentation_when_no_grade_zero_gens(self):
        # Hopf link: all generators in degree ±1; none in degree 0.
        # The trivial augmentation (all generators → 0) must still be found.
        hopf = Leg([2, 2], maslov=[0, 0])
        assert not any(x == 0 for x in hopf.grading), "precondition: no grade-0 gens"
        augs = hopf.augmentations()
        assert len(augs) == 1
        assert augs[0].data == []


# ---------------------------------------------------------------------------
# Section 4: Linearized homology
# ---------------------------------------------------------------------------

class TestLinearizedHomology:

    def test_trefoil_lin_hom_single(self):
        augm = Leg(TREFOIL).augmentations()[0]
        assert augm.lin_hom == {1: 1, 0: 2}

    def test_trefoil_lin_hom_all_same(self):
        polys = Leg(TREFOIL).all_lin_hom()
        assert len(polys) == 1
        assert polys[0] == {1: 1, 0: 2}

    def test_fig_eight_lin_hom(self):
        polys = Leg(FIG_EIGHT).all_lin_hom()
        assert len(polys) == 1
        assert polys[0] == {1: 2, -1: 1}

    def test_chekanov_pair_lin_hom_differ(self):
        assert Leg(M52_A).all_lin_hom() != Leg(M52_B).all_lin_hom()

    def test_chekanov_m52a_lin_hom(self):
        polys = Leg(M52_A).all_lin_hom()
        assert len(polys) == 1
        assert polys[0] == {2: 1, 1: 1, -2: 1}

    def test_chekanov_m52b_lin_hom(self):
        polys = Leg(M52_B).all_lin_hom()
        assert len(polys) == 1
        assert polys[0] == {1: 1, 0: 2}

    def test_lin_hom_is_cached(self):
        augm = Leg(TREFOIL).augmentations()[0]
        assert augm.lin_hom is augm.lin_hom


# ---------------------------------------------------------------------------
# Section 5: Z[lambda] differential
# ---------------------------------------------------------------------------

class TestZDiff:

    def test_trefoil_zdiff_length(self):
        k = Leg(TREFOIL)
        assert len(k.dga('Zlambda').differential) == len(k.grading)

    def test_trefoil_zdiff_crossings_zero(self):
        zd = Leg(TREFOIL).dga('Zlambda').differential
        assert zd[0] == {}
        assert zd[1] == {}
        assert zd[2] == {}

    def test_trefoil_zdiff_cusp_has_lambda(self):
        zd = Leg(TREFOIL).dga('Zlambda').differential
        # knot has one component (c=0), so kind is ('lambda', 0)
        has_lambda = any(isinstance(kind, tuple) and kind[0] == 'lambda' for (_, kind) in zd[-1])
        assert has_lambda

    def test_trefoil_zdiff_check_d_squared(self):
        assert Leg(TREFOIL).dga('Zlambda').check_d_squared()

    def test_link_zlambda_diff_works(self):
        d = Leg(LINK_2, maslov=[0, 0]).dga('Zlambda').differential
        assert isinstance(d, list) and len(d) > 0

    def test_link_zlambda_two_lambda_kinds(self):
        d = Leg(LINK_2, maslov=[0, 0]).dga('Zlambda').differential
        kinds = {kind for entry in d for (_, kind) in entry}
        assert ('lambda', 0) in kinds
        assert ('lambda', 1) in kinds

    def test_link_zlambda_check_d_squared(self):
        assert Leg(LINK_2, maslov=[0, 0]).dga('Zlambda').check_d_squared()

    def test_zn_differential_link_symbolic(self):
        # Z/n differential for links keeps lambda_c symbolic (not pre-substituted)
        d = Leg(LINK_2, maslov=[0, 0]).dga('Z3').differential
        kinds = {kind for entry in d for (_, kind) in entry}
        assert ('lambda', 0) in kinds
        assert ('lambda', 1) in kinds


# ---------------------------------------------------------------------------
# Section 5b: Z/n augmentations for links
# ---------------------------------------------------------------------------

class TestZnLinkAugmentations:

    def test_augmentations_returns_list(self):
        augs = Leg(HOPF, maslov=HOPF_MASLOV).dga('Z3').augmentations()
        assert isinstance(augs, list)

    def test_augmentations_nonempty(self):
        augs = Leg(HOPF, maslov=HOPF_MASLOV).dga('Z3').augmentations()
        assert len(augs) == 7

    def test_lambda_keys_in_data(self):
        augs = Leg(HOPF, maslov=HOPF_MASLOV).dga('Z3').augmentations()
        for a in augs:
            assert ('lambda', 0) in a.data
            assert ('lambda', 1) in a.data

    def test_lambda_values_are_units(self):
        from math import gcd
        n = 3
        augs = Leg(HOPF, maslov=HOPF_MASLOV).dga('Z3').augmentations()
        for a in augs:
            for c in (0, 1):
                v = a.data[('lambda', c)]
                assert gcd(v, n) == 1, f"lambda_{c}={v} is not a unit mod {n}"

    def test_knot_default_lambda_preserved(self):
        # For knots, default is lambda -> -1 = n-1 mod n
        augs = Leg(TREFOIL).dga('Z3').augmentations()
        assert len(augs) == 10
        assert all(a.data.get(('lambda', 0)) == 2 for a in augs)

    def test_knot_lambda_override_zero_augs(self):
        # lambda=1 contradicts d_squared=0 constraint for trefoil over Z/3
        augs = Leg(TREFOIL).dga('Z3').augmentations(lambda_values={0: 1})
        assert len(augs) == 0

    def test_lin_hom_works_for_link_augmentation(self):
        augs = Leg(HOPF, maslov=HOPF_MASLOV).dga('Z3').augmentations()
        lh = augs[0].lin_hom
        assert isinstance(lh, dict)

    def test_augmentations_cached(self):
        dga = Leg(HOPF, maslov=HOPF_MASLOV).dga('Z3')
        assert dga.augmentations() is dga.augmentations()

    def test_lambda_values_param_changes_result(self):
        dga = Leg(HOPF, maslov=HOPF_MASLOV).dga('Z3')
        default_augs = dga.augmentations()
        # Fix lambda_0=1 (a valid unit): result may differ from search-all
        fixed_augs = dga.augmentations(lambda_values={0: 1})
        assert isinstance(fixed_augs, list)
        # Verify all returned augs have lambda_0=1
        assert all(a.data[('lambda', 0)] == 1 for a in fixed_augs)

    def test_lin_hom_grading_mod_nonzero(self):
        # grading_mod=2 takes diff_mat_zp path; must handle new dict format and lambda factors
        dga = Leg(HOPF, maslov=HOPF_MASLOV).dga('Z3')
        augs = dga.augmentations(grading_mod=2)
        assert len(augs) == 7
        lh = augs[0].lin_hom
        assert isinstance(lh, dict)
        # The set of distinct lin_hom polynomials must agree regardless of grading_mod
        set_gm2 = {frozenset(a.lin_hom.items()) for a in augs}
        set_gm0 = {frozenset(a.lin_hom.items()) for a in dga.augmentations(grading_mod=0)}
        assert set_gm2 == set_gm0


# ---------------------------------------------------------------------------
# Section 6: Cup product
# ---------------------------------------------------------------------------

class TestCupProduct:

    def test_double_products_returns_square_table(self):
        augm = Leg(FIG_EIGHT).augmentations()[0]
        table, basis = augm.double_products
        n = len(basis)
        assert len(table) == n
        assert all(len(row) == n for row in table)

    def test_double_products_trefoil(self):
        augm = Leg(TREFOIL).augmentations()[0]
        table, basis = augm.double_products
        assert isinstance(table, list)

    def test_double_products_is_cached(self):
        augm = Leg(TREFOIL).augmentations()[0]
        assert augm.double_products is augm.double_products


# ---------------------------------------------------------------------------
# Section 7: Ruling invariant
# ---------------------------------------------------------------------------

class TestRulings:

    def test_trefoil_rulings_count(self):
        assert len(Leg(TREFOIL).rulings()) == 3

    def test_trefoil_ruling_invariant(self):
        assert Leg(TREFOIL).ruling_invariant() == {0: 2, 2: 1}

    def test_fig_eight_ruling_invariant(self):
        assert Leg(FIG_EIGHT).ruling_invariant() == {0: 1}

    def test_chekanov_pair_rulings_differ(self):
        assert Leg(M52_A).ruling_invariant() != Leg(M52_B).ruling_invariant()

    def test_ruling_implies_augmentation(self):
        """A knot with rulings has augmentations (Fuchs-Ishkhanov)."""
        for b in [TREFOIL, FIG_EIGHT, M52_A, M52_B]:
            k = Leg(b)
            if k.rulings():
                assert len(k.augmentations()) > 0

    def test_rulings_are_cached(self):
        k = Leg(TREFOIL)
        assert k.rulings() is k.rulings()

    def test_rulings_z_graded_raises_for_nonzero_rot_link(self):
        # LINK_2 has rot=(1,1); Z-graded rulings require rot=0
        with pytest.raises(ValueError):
            Leg(LINK_2, maslov=[0, 0]).rulings(grading_mod=0)

    def test_rulings_incompatible_mod_raises(self):
        # grading_mod=3 requires 3 | 2*rot_c; rot=(1,1) gives 2*1=2, 3∤2
        with pytest.raises(ValueError):
            Leg(LINK_2, maslov=[0, 0]).rulings(grading_mod=3)

    def test_ruling_invariant_propagates_guard(self):
        with pytest.raises(ValueError):
            Leg(LINK_2, maslov=[0, 0]).ruling_invariant(grading_mod=0)


# ---------------------------------------------------------------------------
# Section 8: check_d_squared
# ---------------------------------------------------------------------------

class TestCheckDSquared:

    def test_atlas_d_squared_zero(self):
        """Every braid in ATLAS satisfies d^2 = 0 over Z/2."""
        for name, braids in ATLAS.items():
            for b in braids:
                assert Leg(b).dga().check_d_squared(), \
                    f"d^2 != 0 for {name}: {b}"

    def test_noncommutativity_detected(self):
        """
        Replacing [7,2,4] with [4,2,7] in d(gen 8) of M52_A breaks d^2=0.
        The monomials differ only in order, which matters in the noncommutative algebra.
        """
        k = Leg(M52_A)
        d_bad = copy.deepcopy(k.dga().differential)
        poly = d_bad[7]
        poly[poly.index([7, 2, 4])] = [4, 2, 7]
        assert k.dga().check_d_squared(), "original should satisfy d^2 = 0"
        bad_dga = DGA(k, GroundRing.Z2)
        bad_dga._differential = d_bad
        assert not bad_dga.check_d_squared(), "permuted monomial should break d^2 = 0"


# ---------------------------------------------------------------------------
# Section 9: check_d_squared_z  (DGA over Z[λ])
# ---------------------------------------------------------------------------

class TestCheckDSquaredZ:

    def test_atlas_d_squared_z_zero(self):
        """Every braid in ATLAS satisfies d^2 = 0 over Z[λ]."""
        for name, braids in ATLAS.items():
            for b in braids:
                assert Leg(b).dga('Zlambda').check_d_squared(), \
                    f"d^2 != 0 over Z[λ] for {name}: {b}"

    def test_commuted_monomial_detected(self):
        """
        Commuting [3,1] → [1,3] in d(gen 7) of 4_1 breaks d^2=0.
        Over Z the noncommutative word order matters (unlike Z/2 set-based check).
        """
        k = Leg(FIG_EIGHT)
        zlam = k.dga('Zlambda')
        d_bad = copy.deepcopy(zlam.differential)
        d_bad[6][(1, 3), 'int'] = d_bad[6].pop(((3, 1), 'int'))
        assert zlam.check_d_squared(), "original should satisfy d^2 = 0"
        bad_dga = DGA(k, GroundRing.ZLAMBDA)
        bad_dga._differential = d_bad
        assert not bad_dga.check_d_squared(), "commuted word should break d^2 = 0"

    def test_wrong_sign_detected(self):
        """Flipping the sign of [11,8,7] in d(gen 12) of 4_1 from -1 to +1 breaks d^2=0."""
        k = Leg(FIG_EIGHT)
        zlam = k.dga('Zlambda')
        d_bad = copy.deepcopy(zlam.differential)
        d_bad[11][(11, 8, 7), 'int'] = +1   # was -1
        assert zlam.check_d_squared(), "original should satisfy d^2 = 0"
        bad_dga = DGA(k, GroundRing.ZLAMBDA)
        bad_dga._differential = d_bad
        assert not bad_dga.check_d_squared(), "wrong sign should break d^2 = 0"


# ---------------------------------------------------------------------------
# Section 10: OO-specific interface tests
# ---------------------------------------------------------------------------

class TestLegConstruction:

    def test_braid_word_stores_braid(self):
        k = Leg(TREFOIL)
        assert k.braid == TREFOIL

    def test_braid_word_default_name(self):
        k = Leg(TREFOIL)
        assert k.name == repr(TREFOIL)

    def test_braid_word_repr(self):
        k = Leg(TREFOIL)
        assert repr(k) == f"Leg({repr(TREFOIL)!r})"

    def test_atlas_lookup_braid(self):
        k = Leg('mK3_1')
        assert k.braid == ATLAS['mK3_1'][0]

    def test_atlas_lookup_name(self):
        k = Leg('mK3_1')
        assert k.name == 'mK3_1'

    def test_atlas_lookup_repr(self):
        assert repr(Leg('mK3_1')) == "Leg('mK3_1')"

    def test_atlas_explicit_index_zero(self):
        k = Leg('K4_1.0')
        assert k.braid == ATLAS['K4_1'][0]
        assert k.name == 'K4_1.0'

    def test_atlas_explicit_index_matches_plain(self):
        assert Leg('K4_1.0').braid == Leg('K4_1').braid

    def test_atlas_multi_rep_index_zero(self):
        k = Leg('mK5_2.0')
        assert k.braid == ATLAS['mK5_2'][0]

    def test_atlas_multi_rep_index_one(self):
        k = Leg('mK5_2.1')
        assert k.braid == ATLAS['mK5_2'][1]

    def test_atlas_multi_rep_braids_differ(self):
        assert Leg('mK5_2.0').braid != Leg('mK5_2.1').braid

    def test_atlas_multi_rep_no_index_raises(self):
        with pytest.raises(ValueError, match="Legendrian representative"):
            Leg('mK5_2')

    def test_atlas_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown knot name"):
            Leg('K99_99')

    def test_atlas_out_of_range_index_raises(self):
        with pytest.raises(IndexError):
            Leg('K4_1.5')

    def test_invalid_input_type_raises(self):
        with pytest.raises(TypeError):
            Leg(42)

    def test_custom_name_overrides_default(self):
        k = Leg(TREFOIL, name='my_trefoil')
        assert k.name == 'my_trefoil'
        assert repr(k) == "Leg('my_trefoil')"

    def test_custom_name_on_atlas_lookup(self):
        k = Leg('mK3_1', name='trefoil_mirror')
        assert k.name == 'trefoil_mirror'
        assert k.braid == ATLAS['mK3_1'][0]

    def test_braid_is_copied(self):
        b = [2, 2, 2]
        k = Leg(b)
        b.append(99)
        assert k.braid == [2, 2, 2]


class TestDGACaching:

    def test_dga_default_is_z2(self):
        k = Leg(TREFOIL)
        assert k.dga().ring == GroundRing.Z2

    def test_dga_same_ring_returns_same_object(self):
        k = Leg(TREFOIL)
        assert k.dga() is k.dga()

    def test_dga_z2_and_default_are_same_object(self):
        k = Leg(TREFOIL)
        assert k.dga(GroundRing.Z2) is k.dga()

    def test_dga_different_rings_are_distinct(self):
        k = Leg(TREFOIL)
        assert k.dga(GroundRing.Z2) is not k.dga(GroundRing.ZLAMBDA)

    def test_dga_string_ring_z2(self):
        k = Leg(TREFOIL)
        assert k.dga('Z2') is k.dga(GroundRing.Z2)

    def test_dga_string_ring_zlambda(self):
        k = Leg(TREFOIL)
        assert k.dga('Zlambda') is k.dga(GroundRing.ZLAMBDA)

    def test_dga_back_reference_to_leg(self):
        k = Leg(TREFOIL)
        assert k.dga().leg is k

    def test_dga_repr(self):
        k = Leg(TREFOIL)
        r = repr(k.dga())
        assert 'DGA' in r and 'GroundRing.Z2' in r


class TestGroundRing:

    def test_z2_equals_ground_ring_2(self):
        assert GroundRing.Z2 == GroundRing(2)

    def test_zlambda_equals_ground_ring_0(self):
        assert GroundRing.ZLAMBDA == GroundRing(0)

    def test_zn_factory(self):
        assert GroundRing.Zn(3) == GroundRing(3)

    def test_from_str_z2(self):
        assert GroundRing.from_str('Z2') == GroundRing.Z2

    def test_from_str_zlambda(self):
        assert GroundRing.from_str('Zlambda') == GroundRing.ZLAMBDA

    def test_from_str_zn(self):
        assert GroundRing.from_str('Z3') == GroundRing.Zn(3)

    def test_string_constructor_z2(self):
        assert GroundRing('Z2') == GroundRing.Z2

    def test_string_constructor_zlambda(self):
        assert GroundRing('Zlambda') == GroundRing.ZLAMBDA

    def test_hashable_as_dict_key(self):
        d = {GroundRing.Z2: 'a', GroundRing.ZLAMBDA: 'b', GroundRing.Zn(3): 'c'}
        assert d[GroundRing(2)] == 'a'
        assert d[GroundRing(0)] == 'b'
        assert d[GroundRing(3)] == 'c'

    def test_repr_z2(self):
        assert repr(GroundRing.Z2) == 'GroundRing.Z2'

    def test_repr_zlambda(self):
        assert repr(GroundRing.ZLAMBDA) == 'GroundRing.ZLAMBDA'

    def test_repr_zn(self):
        assert repr(GroundRing.Zn(3)) == 'GroundRing.Zn(3)'

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError):
            GroundRing.from_str('nonsense')

    def test_default_ground_ring_is_z2(self):
        assert DEFAULT_GROUND_RING == GroundRing.Z2


class TestNotImplemented:
    """Operations not yet supported for certain rings should raise NotImplementedError."""

    def test_augmentations_zlambda_raises(self):
        with pytest.raises(NotImplementedError):
            Leg(TREFOIL).dga('Zlambda').augmentations()

    def test_lin_hom_zlambda_raises(self):
        with pytest.raises(NotImplementedError):
            Leg(TREFOIL).dga('Zlambda').all_lin_hom()

    def test_aug_count_zlambda_raises(self):
        with pytest.raises(NotImplementedError):
            Leg(TREFOIL).dga('Zlambda').aug_count()

    def test_check_d_squared_zn_raises(self):
        with pytest.raises(NotImplementedError):
            Leg(TREFOIL).dga(GroundRing.Zn(3)).check_d_squared()

    def test_aug_lin_hom_zlambda_raises(self):
        # Manually construct an Augmentation over ZLAMBDA to test the guard
        k = Leg(TREFOIL)
        bad_dga = DGA(k, GroundRing.ZLAMBDA)
        augm = Augmentation([1], bad_dga)
        with pytest.raises(NotImplementedError):
            _ = augm.lin_hom

    def test_cohomology_basis_zn_raises(self):
        k = Leg(TREFOIL)
        bad_dga = DGA(k, GroundRing.Zn(3))
        augm = Augmentation({1: 1}, bad_dga)
        with pytest.raises(NotImplementedError):
            _ = augm.cohomology_basis


# ---------------------------------------------------------------------------
# Section 11: Tangle input
# ---------------------------------------------------------------------------

# Tangle for mK3_1: two LCs, then crossings 2,3,1,3,1,3,2 (1-indexed) → 0-indexed,
# then two RCs — all in plat order, so no commutation rules fire.
_TREFOIL_TANGLE_PLAT = (
    [('<', 0), ('<', 0)]
    + [('X', g - 1) for g in [2, 3, 1, 3, 1, 3, 2]]
    + [('>', 0), ('>', 0)]
)

# Non-plat: second LC sits after two crossings; rule A must fire twice.
# Derived by reverse-applying rule A (free case) to the plat LC(0)·LC(0)·X(2)·X(2)·...
# Reverse: LC(0)·X(2) → X(0)·LC(0), applied twice → LC(0)·X(0)·X(0)·LC(0)·RC(0)·RC(0)
_TANGLE_NON_PLAT = [('<', 0), ('X', 0), ('X', 0), ('<', 0), ('>', 0), ('>', 0)]


class TestTangleInput:

    # --- round-trips (no commutation rules fire) ---

    def test_lc_x_rc_braid(self):
        assert Leg([('<', 0), ('X', 0), ('>', 0)]).braid == [1]

    def test_lc_x_rc_matches_braid_tb(self):
        assert Leg([('<', 0), ('X', 0), ('>', 0)]).tb == Leg([1]).tb

    def test_lc_x_rc_matches_braid_rot(self):
        assert Leg([('<', 0), ('X', 0), ('>', 0)]).rot == Leg([1]).rot

    def test_lc_x_rc_num_cusps(self):
        assert Leg([('<', 0), ('X', 0), ('>', 0)]).num_cusps == 1

    def test_lc_x_rc_tangle_stored(self):
        t = [('<', 0), ('X', 0), ('>', 0)]
        assert Leg(t).tangle == t

    def test_unknot_empty_braid(self):
        assert Leg([('<', 0), ('>', 0)]).braid == []

    def test_unknot_num_cusps(self):
        assert Leg([('<', 0), ('>', 0)]).num_cusps == 1

    def test_trefoil_braid_matches_atlas(self):
        assert Leg(_TREFOIL_TANGLE_PLAT).braid == Leg('mK3_1').braid

    def test_trefoil_tb_matches_atlas(self):
        assert Leg(_TREFOIL_TANGLE_PLAT).tb == Leg('mK3_1').tb

    def test_trefoil_rot_matches_atlas(self):
        assert Leg(_TREFOIL_TANGLE_PLAT).rot == Leg('mK3_1').rot

    def test_trefoil_num_cusps(self):
        assert Leg(_TREFOIL_TANGLE_PLAT).num_cusps == 2

    # --- rule A LR-II: X(0)·LC(1) with h = j−1 triggers the LR-II move ---

    def test_rule_a_lrII_braid(self):
        # LC(0) X(0) LC(1) RC(0) RC(0) → LC(0) LC(0) X(2) X(1) X(0) RC(0) RC(0)
        assert Leg([('<', 0), ('X', 0), ('<', 1), ('>', 0), ('>', 0)]).braid == [3, 2, 1]

    def test_rule_a_lrII_num_cusps(self):
        assert Leg([('<', 0), ('X', 0), ('<', 1), ('>', 0), ('>', 0)]).num_cusps == 2

    def test_rule_a_lrII_tb_matches_ref(self):
        ra = Leg([('<', 0), ('X', 0), ('<', 1), ('>', 0), ('>', 0)])
        assert ra.tb == Leg([3, 2, 1]).tb

    def test_rule_a_lrII_rot_matches_ref(self):
        ra = Leg([('<', 0), ('X', 0), ('<', 1), ('>', 0), ('>', 0)])
        assert ra.rot == Leg([3, 2, 1]).rot

    # --- rule D: adjacent nested LC(0)·LC(1) → extra crossings introduced ---

    def test_rule_d_braid(self):
        # LC(0) LC(1) RC(0) RC(0) → LC(0) LC(2) X(1) X(0) RC(0) RC(0) → braid [2,1]
        assert Leg([('<', 0), ('<', 1), ('>', 0), ('>', 0)]).braid == [2, 1]

    def test_rule_d_num_cusps(self):
        assert Leg([('<', 0), ('<', 1), ('>', 0), ('>', 0)]).num_cusps == 2

    # --- non-plat: rule A fires twice (free commutation) ---

    def test_nonplat_braid(self):
        assert Leg(_TANGLE_NON_PLAT).braid == [3, 3]

    def test_nonplat_num_cusps(self):
        assert Leg(_TANGLE_NON_PLAT).num_cusps == 2

    def test_nonplat_tb_matches_ref(self):
        assert Leg(_TANGLE_NON_PLAT).tb == Leg([3, 3]).tb

    def test_nonplat_rot_is_tuple(self):
        # _TANGLE_NON_PLAT converts to braid [3,3], a 2-component link
        assert isinstance(Leg(_TANGLE_NON_PLAT).rot, tuple)

    # --- DGA ---

    def test_tangle_trefoil_d_squared_zero(self):
        assert Leg(_TREFOIL_TANGLE_PLAT).dga().check_d_squared()

    # --- name and repr ---

    def test_name_kwarg(self):
        assert Leg([('<', 0), ('>', 0)], name='trivial').name == 'trivial'

    def test_repr_named(self):
        assert repr(Leg([('<', 0), ('>', 0)], name='trivial')) == "Leg('trivial')"

    def test_auto_name_is_repr_of_tangle(self):
        t = [('<', 0), ('>', 0)]
        assert Leg(t).name == repr(t)

    # --- validation errors ---

    def test_validate_bad_op(self):
        with pytest.raises(ValueError):
            Leg([('Z', 0)])

    def test_validate_negative_height(self):
        with pytest.raises(ValueError):
            Leg([('<', -1)])

    def test_validate_unclosed(self):
        with pytest.raises(ValueError):
            Leg([('<', 0)])

    def test_validate_crossing_height_out_of_range(self):
        with pytest.raises(ValueError):
            Leg([('<', 0), ('X', 5), ('>', 0)])

    def test_validate_rc_height_out_of_range(self):
        with pytest.raises(ValueError):
            Leg([('<', 0), ('>', 5)])

    def test_validate_not_a_tuple(self):
        with pytest.raises(ValueError):
            Leg([('X',)])

    # --- draw ---

    def test_draw_plat_returns_figure(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig = Leg([2, 2, 2]).draw()
        assert hasattr(fig, 'savefig')
        plt.close(fig)

    def test_draw_plat_color_true(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig = Leg([2, 2, 2]).draw(color=True)
        assert hasattr(fig, 'savefig')
        plt.close(fig)

    def test_draw_tangle_returns_figure(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        t = [('<', 0), ('X', 0), ('>', 0)]
        fig = Leg(t).draw(method='tangle')
        assert hasattr(fig, 'savefig')
        plt.close(fig)

    def test_draw_tangle_no_tangle_raises(self):
        with pytest.raises(AttributeError):
            Leg([1]).draw(method='tangle')

    def test_draw_tangle_use_tangle_compat(self):
        """use_tangle=True is still accepted as a deprecated alias."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        t = [('<', 0), ('X', 0), ('>', 0)]
        fig = Leg(t).draw(use_tangle=True)
        assert hasattr(fig, 'savefig')
        plt.close(fig)

    def test_draw_grid_returns_figure(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig = Leg(([1, 0], [0, 1])).draw(method='grid')
        assert hasattr(fig, 'savefig')
        plt.close(fig)

    def test_draw_grid_no_grid_raises(self):
        with pytest.raises(AttributeError):
            Leg([2, 2, 2]).draw(method='grid')

    def test_draw_grid_color_true(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig = Leg(([1, 0], [0, 1])).draw(method='grid', color=True)
        assert hasattr(fig, 'savefig')
        plt.close(fig)


# ---------------------------------------------------------------------------
# Section 12: Grid diagram input
#
# Convention: Leg((X_perm, O_perm)) where X_perm[j] is the row of the X
# marker in column j, O_perm[j] is the row of the O marker in column j,
# rows are 0-indexed from the bottom, columns 0-indexed from the left.
# Vertical strands go over horizontal strands, and the front projection is
# obtained by rotating the grid 45° counter-clockwise.
#
# Test strategy
# -------------
# 1. Smallest known case: the 2×2 grid ([1,0],[0,1]) encodes the max-tb
#    Legendrian unknot (tb = -1, rot = 0, braid = []).
# 2. A 3×3 grid ([0,1,2],[1,2,0]) encodes a once-stabilised Legendrian
#    unknot (tb = -2, rot = 1).
# 3. Storage: Leg.grid and Leg.tangle are both set; Leg.tangle is the
#    intermediate tangle produced by _grid_to_tangle.
# 4. Naming: name kwarg overrides; auto-name is repr of the input tuple.
# 5. Validation: wrong-length, non-permutation, same-cell, and grid-size-1
#    inputs each raise ValueError.
# 6. d² = 0 confirms the converted DGA is valid.
# ---------------------------------------------------------------------------

# 2×2 grid for the Legendrian unknot (maximum tb = -1)
_UNKNOT_GRID_2x2 = ([1, 0], [0, 1])

# 3×3 grid for a once-stabilised Legendrian unknot (tb = -2, rot = 1)
_UNKNOT_GRID_3x3 = ([0, 1, 2], [1, 2, 0])


class TestGridInput:

    # --- classical invariants for the 2×2 unknot ---

    def test_2x2_unknot_nc(self):
        assert Leg(_UNKNOT_GRID_2x2).num_components == 1

    def test_2x2_unknot_tb(self):
        assert Leg(_UNKNOT_GRID_2x2).tb == -1

    def test_2x2_unknot_rot(self):
        assert Leg(_UNKNOT_GRID_2x2).rot == 0

    def test_2x2_unknot_braid(self):
        assert Leg(_UNKNOT_GRID_2x2).braid == []

    def test_2x2_unknot_num_cusps(self):
        assert Leg(_UNKNOT_GRID_2x2).num_cusps == 1

    def test_2x2_unknot_tangle(self):
        assert Leg(_UNKNOT_GRID_2x2).tangle == [('<', 0), ('>', 0)]

    # --- classical invariants for the 3×3 stabilised unknot ---

    def test_3x3_unknot_nc(self):
        assert Leg(_UNKNOT_GRID_3x3).num_components == 1

    def test_3x3_unknot_tb(self):
        assert Leg(_UNKNOT_GRID_3x3).tb == -2

    def test_3x3_unknot_rot(self):
        assert Leg(_UNKNOT_GRID_3x3).rot == 1

    # --- storage ---

    def test_grid_attribute_stored(self):
        k = Leg(_UNKNOT_GRID_2x2)
        assert k.grid == ([1, 0], [0, 1])

    def test_tangle_attribute_set_by_grid(self):
        # _grid_to_tangle should produce the intermediate tangle
        k = Leg(_UNKNOT_GRID_2x2)
        assert hasattr(k, 'tangle') and isinstance(k.tangle, list)

    # --- naming ---

    def test_name_kwarg_grid(self):
        k = Leg(_UNKNOT_GRID_2x2, name='unknot')
        assert k.name == 'unknot'

    def test_auto_name_is_repr_of_tuple(self):
        inp = ([1, 0], [0, 1])
        assert Leg(inp).name == repr(inp)

    # --- DGA self-consistency ---

    def test_2x2_unknot_d_squared_zero(self):
        # braid is empty so the DGA is trivial; check_d_squared returns True
        assert Leg(_UNKNOT_GRID_2x2).dga().check_d_squared()

    def test_3x3_d_squared_zero(self):
        assert Leg(_UNKNOT_GRID_3x3).dga().check_d_squared()

    # --- validation errors ---

    def test_validate_unequal_lengths(self):
        with pytest.raises(ValueError):
            Leg(([0, 1], [0, 1, 2]))

    def test_validate_not_permutation_X(self):
        with pytest.raises(ValueError):
            Leg(([0, 0, 2], [1, 2, 0]))

    def test_validate_not_permutation_O(self):
        with pytest.raises(ValueError):
            Leg(([0, 1, 2], [0, 0, 1]))

    def test_validate_grid_size_one(self):
        with pytest.raises(ValueError):
            Leg(([0], [0]))

    def test_validate_same_cell(self):
        # X[0] == O[0] is not allowed
        with pytest.raises(ValueError):
            Leg(([0, 1], [0, 2]))

    # --- Legendrian trefoil: mK3_1 ---
    # 1-indexed user input: X=[5,4,3,2,1], O=[2,1,5,4,3]
    # 0-indexed for code:   X=[4,3,2,1,0], O=[1,0,4,3,2]
    # Expected: nc=1, tb=1, rot=0

    _TREFOIL_MIRROR_GRID = ([4, 3, 2, 1, 0], [1, 0, 4, 3, 2])

    def test_trefoil_mirror_grid_nc(self):
        assert Leg(self._TREFOIL_MIRROR_GRID).num_components == 1

    def test_trefoil_mirror_grid_tb(self):
        assert Leg(self._TREFOIL_MIRROR_GRID).tb == 1

    def test_trefoil_mirror_grid_rot(self):
        assert Leg(self._TREFOIL_MIRROR_GRID).rot == 0

    # --- Legendrian trefoil: K3_1 ---
    # 1-indexed user input: X=[5,1,2,3,4], O=[2,3,4,5,1]
    # 0-indexed for code:   X=[4,0,1,2,3], O=[1,2,3,4,0]
    # Expected: nc=1, tb=-6, rot=1 (or rot=-1 by symmetry)

    _TREFOIL_GRID = ([4, 0, 1, 2, 3], [1, 2, 3, 4, 0])

    def test_trefoil_grid_nc(self):
        assert Leg(self._TREFOIL_GRID).num_components == 1

    def test_trefoil_grid_tb(self):
        assert Leg(self._TREFOIL_GRID).tb == -6

    def test_trefoil_grid_rot(self):
        assert abs(Leg(self._TREFOIL_GRID).rot) == 1


# ---------------------------------------------------------------------------
# Section 13: Grid-atlas round-trips
#
# All 138 grid representatives from GRID_ATLAS are tested against pre-computed
# expected invariants.  For rot = 0 cases the ruling polynomial is also
# checked; for rot != 0 only tb and rot are tested.
#
# Validation strength:
#   "fully validated"  — tb, rot, and ruling all agree with the braid-word
#                        ATLAS entry for the same knot name (~56 reps)
#   "partial"          — tb/rot match but rot != 0, or ruling differs
#                        (different Legendrian type in the same topological
#                        class, e.g. a stabilisation or a rot != 0 rep)
#   "regression"       — knot name newly added to ATLAS from grid data;
#                        no independent reference, but locks in behaviour
# ---------------------------------------------------------------------------

from auxiliary_data.grid_atlas import GRID_ATLAS as _GA


def _build_grid_cases():
    rows = []
    ids = []
    for name in sorted(_GA.keys()):
        for i, grid in enumerate(_GA[name]):
            g = Leg(grid)
            ruling = g.ruling_invariant() if g.rot == 0 else None
            rows.append((grid, g.num_components, g.tb, g.rot, ruling))
            ids.append(f"{name}.{i}")
    return rows, ids


_grid_cases_cache: list | None = None
_grid_ids_cache: list | None = None


def _get_grid_cases():
    global _grid_cases_cache, _grid_ids_cache
    if _grid_cases_cache is None:
        _grid_cases_cache, _grid_ids_cache = _build_grid_cases()
    return _grid_cases_cache, _grid_ids_cache


def pytest_generate_tests(metafunc):
    if metafunc.cls is TestGridAtlas:
        cases, ids = _get_grid_cases()
        ruling_cases = [c for c in cases if c[4] is not None]
        ruling_ids = [id_ for id_, c in zip(ids, cases) if c[4] is not None]
        if metafunc.function.__name__ == "test_ruling":
            metafunc.parametrize("grid,nc,tb,rot,ruling", ruling_cases, ids=ruling_ids)
        else:
            metafunc.parametrize("grid,nc,tb,rot,ruling", cases, ids=ids)


class TestGridAtlas:
    """Round-trip tests for every grid representative in GRID_ATLAS."""

    def test_nc(self, grid, nc, tb, rot, ruling):
        assert Leg(grid).num_components == nc

    def test_tb(self, grid, nc, tb, rot, ruling):
        assert Leg(grid).tb == tb

    def test_rot(self, grid, nc, tb, rot, ruling):
        assert Leg(grid).rot == rot

    def test_ruling(self, grid, nc, tb, rot, ruling):
        assert Leg(grid).ruling_invariant() == ruling


# ---------------------------------------------------------------------------
# Section 14: HOMFLY and Kauffman polynomial reality checks
#
# Rutherford (2006): for a max-tb Legendrian rep L of knot K with tb = tb(L),
#   R2(L)(z) = coefficient of a^{-tb-1} in HOMFLY(K; a, z)
#   R1(L)(z) = coefficient of a^{-tb-1} in Dubrovnik(K; a, z)
# where Dubrovnik is obtained from the KnotInfo Kauffman polynomial by
# substituting a -> -i*a, z -> i*z (then taking absolute values of
# coefficients to reconcile z-sign conventions).
#
# Expected values below were extracted from KnotInfo polynomial data and
# verified against the atlas representatives; (R2, R1) are dicts mapping
# z-degree to coefficient.
# ---------------------------------------------------------------------------

_POLY_EXPECTED = {
    'mK3_1': ({0: 2, 2: 1}, {0: 2, 2: 1}),
    'K4_1': ({0: 1}, {0: 1, 2: 1}),
    'mK5_1': ({0: 3, 2: 4, 4: 1}, {0: 3, 2: 4, 4: 1}),
    'mK5_2.0': ({0: 1, 2: 1}, {0: 1, 2: 1}),
    'mK5_2.1': ({0: 1, 2: 1}, {0: 1, 2: 1}),
    'K6_1': ({0: 1}, {0: 1, 2: 3, 4: 1}),
    'mK6_1.0': ({0: 1}, {0: 1, 2: 1}),
    'mK6_1.1': ({0: 1}, {0: 1, 2: 1}),
    'K6_2': ({}, {2: 1}),
    'mK6_2': ({0: 2, 2: 1}, {0: 2, 2: 3, 4: 1}),
    'mK7_1': ({0: 4, 2: 10, 4: 6, 6: 1}, {0: 4, 2: 10, 4: 6, 6: 1}),
    'mK7_2.0': ({0: 1, 2: 1}, {0: 1, 2: 1}),
    'mK7_2.1': ({0: 1, 2: 1}, {0: 1, 2: 1}),
    'mK7_2.2': ({0: 1, 2: 1}, {0: 1, 2: 1}),
    'mK7_2.3': ({0: 1, 2: 1}, {0: 1, 2: 1}),
    'K7_3.0': ({0: 1, 2: 3, 4: 1}, {0: 1, 2: 3, 4: 1}),
    'K7_3.1': ({0: 1, 2: 3, 4: 1}, {0: 1, 2: 3, 4: 1}),
    'K7_4.0': ({2: 1}, {2: 1}),
    'K7_4.1': ({2: 1}, {2: 1}),
    'K7_4.2': ({2: 1}, {2: 1}),
    'K7_4.3': ({2: 1}, {2: 1}),
    'mK7_5.0': ({0: 2, 2: 3, 4: 1}, {0: 2, 2: 3, 4: 1}),
    'mK7_5.1': ({0: 2, 2: 3, 4: 1}, {0: 2, 2: 3, 4: 1}),
    'mK7_6.0': ({0: 1, 2: 1}, {0: 1, 2: 2, 4: 1}),
    'mK7_6.1': ({0: 1, 2: 1}, {0: 1, 2: 2, 4: 1}),
    'mK7_6.2': ({0: 1, 2: 1}, {0: 1, 2: 2, 4: 1}),
    'mK7_7.0': ({0: 1}, {0: 1, 2: 2, 4: 1}),
    'mK7_7.1': ({0: 1}, {0: 1, 2: 2, 4: 1}),
    'K8_19': ({0: 5, 2: 10, 4: 6, 6: 1}, {0: 5, 2: 10, 4: 6, 6: 1}),
    'K8_21.0': ({}, {2: 2, 4: 1}),
    'K8_21.1': ({}, {2: 2, 4: 1}),
    'mK8_21': ({0: 3, 2: 2}, {0: 3, 2: 3}),
    'K9_42': ({0: 2, 2: 1}, {0: 2, 2: 6, 4: 5, 6: 1}),
    'mK9_42': ({}, {}),
    'K9_43': ({0: 3, 2: 4, 4: 1}, {0: 3, 2: 7, 4: 5, 6: 1}),
    'mK9_44': ({0: 1}, {0: 1, 2: 1}),
    'mK9_45.0': ({0: 2, 2: 2}, {0: 2, 2: 3}),
    'mK9_45.1': ({0: 2, 2: 2}, {0: 2, 2: 3}),
    'K9_46': ({0: 1}, {0: 1, 2: 6, 4: 5, 6: 1}),
    'mK9_46': ({0: 2}, {0: 2}),
    'mK9_47': ({0: 1}, {0: 1, 2: 3}),
    'K9_48.0': ({2: 1}, {2: 1, 4: 1}),
    'K9_48.1': ({2: 1}, {2: 1, 4: 1}),
    'K9_48.2': ({2: 1}, {2: 1, 4: 1}),
    'K9_48.3': ({2: 1}, {2: 1, 4: 1}),
    'K9_49.0': ({2: 2, 4: 1}, {2: 2, 4: 1}),
    'K9_49.1': ({2: 2, 4: 1}, {2: 2, 4: 1}),
    'K10_124': ({0: 7, 2: 21, 4: 21, 6: 8, 8: 1}, {0: 7, 2: 21, 4: 21, 6: 8, 8: 1}),
    'K10_128.0': ({0: 2, 2: 6, 4: 5, 6: 1}, {0: 2, 2: 6, 4: 5, 6: 1}),
    'K10_128.1': ({0: 2, 2: 6, 4: 5, 6: 1}, {0: 2, 2: 6, 4: 5, 6: 1}),
    'K10_132.0': ({}, {}),
    'K10_132.1': ({}, {}),
    'K10_136.0': ({0: 1, 2: 1}, {0: 1, 2: 3, 4: 4, 6: 1}),
    'K10_136.1': ({0: 1, 2: 1}, {0: 1, 2: 3, 4: 4, 6: 1}),
    'K10_136.2': ({0: 1, 2: 1}, {0: 1, 2: 3, 4: 4, 6: 1}),
    'K10_136.3': ({0: 1, 2: 1}, {0: 1, 2: 3, 4: 4, 6: 1}),
    'K10_139': ({0: 6, 2: 21, 4: 21, 6: 8, 8: 1}, {0: 6, 2: 21, 4: 21, 6: 8, 8: 1}),
    'mK10_140.0': ({0: 1}, {0: 1}),
    'mK10_140.1': ({0: 1}, {0: 1}),
    'K10_142.0': ({0: 1, 2: 6, 4: 5, 6: 1}, {0: 1, 2: 6, 4: 5, 6: 1}),
    'K10_142.1': ({0: 1, 2: 6, 4: 5, 6: 1}, {0: 1, 2: 6, 4: 5, 6: 1}),
    'mK10_145': ({0: 2, 2: 4, 4: 1}, {0: 2, 2: 4, 4: 1}),
    'K10_160.0': ({0: 1, 2: 3, 4: 1}, {0: 1, 2: 4, 4: 4, 6: 1}),
    'K10_160.1': ({0: 1, 2: 3, 4: 1}, {0: 1, 2: 4, 4: 4, 6: 1}),
    'mK10_161': ({0: 3, 2: 9, 4: 6, 6: 1}, {0: 3, 2: 9, 4: 6, 6: 1}),
    'K3_1': ({}, {1: 1}),
    'K5_1': ({}, {1: 1}),
    'K5_2': ({}, {1: 2, 3: 1}),
    'K6_3': ({}, {1: 1, 3: 1}),
    'K7_1': ({}, {1: 1}),
    'K7_2': ({}, {1: 3, 3: 4, 5: 1}),
    'K7_5': ({}, {1: 1, 3: 1}),
    'K7_6': ({}, {1: 1, 3: 1}),
    'K7_7': ({}, {3: 1}),
    'K8_20': ({}, {1: 3, 3: 4, 5: 1}),
    'K9_44': ({}, {3: 3, 5: 1, 1: 1}),
    'K9_45': ({}, {1: 2, 3: 3, 5: 1}),
    'K9_47': ({}, {3: 2, 5: 1}),
    'K10_140': ({}, {1: 4, 3: 10, 5: 6, 7: 1}),
    'K10_145': ({}, {1: 5, 3: 10, 5: 6, 7: 1}),
    'K10_161': ({}, {1: 3, 3: 4, 5: 1}),
    'mK7_3': ({}, {1: 2, 3: 1}),
    'mK7_4': ({}, {1: 4, 3: 4, 5: 1}),
    'mK8_19': ({}, {}),
    'mK8_20': ({}, {1: 1}),
    'mK9_43': ({}, {1: 1}),
    'mK9_48': ({}, {1: 4, 3: 3}),
    'mK9_49': ({}, {1: 4, 3: 3}),
    'mK10_124': ({}, {}),
    'mK10_128': ({}, {}),
    'mK10_132': ({}, {}),
    'mK10_136': ({}, {}),
    'mK10_139': ({}, {1: 2, 3: 1}),
    'mK10_142': ({}, {1: 2}),
    'mK10_160': ({}, {1: 2}),
}

_POLY_CASES = [(rep, r2, r1) for rep, (r2, r1) in _POLY_EXPECTED.items()]
_POLY_IDS = list(_POLY_EXPECTED.keys())


class TestPolynomialRealityCheck:
    """
    Verify ruling polynomials against HOMFLY and Kauffman polynomial data
    (Rutherford 2006).
    """

    @pytest.mark.parametrize('rep,expected_r2,expected_r1', _POLY_CASES, ids=_POLY_IDS)
    def test_r2_ruling_polynomial(self, rep, expected_r2, expected_r1):
        k = Leg(rep)
        assert k.ruling_invariant(grading_mod=2) == expected_r2

    @pytest.mark.parametrize('rep,expected_r2,expected_r1', _POLY_CASES, ids=_POLY_IDS)
    def test_r1_ruling_polynomial(self, rep, expected_r2, expected_r1):
        k = Leg(rep)
        assert k.ruling_invariant(grading_mod=1) == expected_r1


# ---------------------------------------------------------------------------
# Section 15: Link grid atlas — structural and DGA sanity checks
#
# Tests that do not require pre-computed invariant values.  Pre-computed
# tb, rot, and ruling data will be added in a later pass.
#
# Test strategy
# -------------
# 1. nc == 2         — every entry in LINK_GRID_ATLAS should be a 2-component link
# 2. d² = 0          — Z/2 DGA correctness; maslov not needed (differential is grade-agnostic)
# 3. rot is a tuple  — regression on multi-component rot
# 4. Z/2 rulings     — rulings(grading_mod=2) never raises (2 | 2r for any r)
# 5. Z graded guard  — rulings(grading_mod=0) raises iff any rot_c ≠ 0
# ---------------------------------------------------------------------------

from auxiliary_data.link_grid_atlas import LINK_GRID_ATLAS as _LGA


def _build_link_cases():
    rows, ids = [], []
    for name in sorted(_LGA.keys()):
        for i, grid in enumerate(_LGA[name]):
            rows.append((name, i, grid))
            ids.append(f"{name}.{i}")
    return rows, ids


_LINK_CASES, _LINK_IDS = _build_link_cases()


class TestLinkGridAtlas:
    """Structural and DGA sanity checks for every link in LINK_GRID_ATLAS."""

    @pytest.mark.parametrize("name,i,grid", _LINK_CASES, ids=_LINK_IDS)
    def test_nc(self, name, i, grid):
        assert Leg(grid).num_components == 2

    @pytest.mark.parametrize("name,i,grid", _LINK_CASES, ids=_LINK_IDS)
    def test_d_squared_zero(self, name, i, grid):
        # Z/2 differential is grade-agnostic; no maslov needed
        assert Leg(grid).dga().check_d_squared()

    @pytest.mark.parametrize("name,i,grid", _LINK_CASES, ids=_LINK_IDS)
    def test_rot_is_tuple(self, name, i, grid):
        assert isinstance(Leg(grid).rot, tuple)

    @pytest.mark.parametrize("name,i,grid", _LINK_CASES, ids=_LINK_IDS)
    def test_z2_rulings_works(self, name, i, grid):
        # grading_mod=2: 2 | 2r_c for every integer r_c, so always valid
        result = Leg(grid, maslov=[0, 0]).rulings(grading_mod=2)
        assert isinstance(result, list)

    @pytest.mark.parametrize("name,i,grid", _LINK_CASES, ids=_LINK_IDS)
    def test_z_graded_rulings_guard(self, name, i, grid):
        leg = Leg(grid, maslov=[0, 0])
        if any(r != 0 for r in leg.rot):
            with pytest.raises(ValueError):
                leg.rulings(grading_mod=0)
        else:
            assert isinstance(leg.rulings(grading_mod=0), list)

    @pytest.mark.parametrize("name,i,grid", _LINK_CASES, ids=_LINK_IDS)
    def test_zlambda_d_squared_zero(self, name, i, grid):
        assert Leg(grid, maslov=[0, 0]).dga('Zlambda').check_d_squared()
