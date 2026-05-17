"""
test_legendrian.py

Tests for legendrian.py (the object-oriented interface).

Run with:  pytest test_legendrian.py
Single:    pytest test_legendrian.py::TestClassicalInvariants::test_trefoil_tb
"""

import copy
from collections import Counter
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _aug_is_valid(augm: Augmentation) -> bool:
    """Re-verify ε∘∂ = 0 over Z/2 using the augmentation's parent DGA."""
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

    def test_differential_raises_for_link(self):
        with pytest.raises(ValueError):
            _ = Leg(LINK_2).dga().differential

    def test_augmentations_raises_for_link(self):
        with pytest.raises(ValueError):
            Leg(LINK_2).augmentations()

    def test_zlambda_diff_raises_for_link(self):
        with pytest.raises(ValueError):
            _ = Leg(LINK_2).dga('Zlambda').differential

    def test_rulings_raises_for_link(self):
        with pytest.raises(ValueError):
            Leg(LINK_2).rulings()

    def test_rulings_graded_raises_for_link(self):
        with pytest.raises(ValueError):
            Leg(LINK_2).rulings(grading_mod=2)


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
        has_lambda = any(kind == 'lambda' for (_, kind) in zd[-1])
        assert has_lambda


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
        assert Leg(TREFOIL).ruling_invariant == {0: 2, 2: 1}

    def test_fig_eight_ruling_invariant(self):
        assert Leg(FIG_EIGHT).ruling_invariant == {0: 1}

    def test_chekanov_pair_rulings_differ(self):
        assert Leg(M52_A).ruling_invariant != Leg(M52_B).ruling_invariant

    def test_ruling_implies_augmentation(self):
        """A knot with rulings has augmentations (Fuchs-Ishkhanov)."""
        for b in [TREFOIL, FIG_EIGHT, M52_A, M52_B]:
            k = Leg(b)
            if k.rulings():
                assert len(k.augmentations()) > 0

    def test_rulings_are_cached(self):
        k = Leg(TREFOIL)
        assert k.rulings() is k.rulings()
        assert k.ruling_invariant is k.ruling_invariant


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
# Section 11: OO-specific interface tests
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
# Section 12: Tangle input
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

    def test_nonplat_rot_matches_ref(self):
        assert Leg(_TANGLE_NON_PLAT).rot == Leg([3, 3]).rot

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

    def test_draw_use_tangle_returns_figure(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        t = [('<', 0), ('X', 0), ('>', 0)]
        fig = Leg(t).draw(use_tangle=True)
        assert hasattr(fig, 'savefig')
        plt.close(fig)

    def test_draw_use_tangle_no_tangle_raises(self):
        with pytest.raises(AttributeError):
            Leg([1]).draw(use_tangle=True)
