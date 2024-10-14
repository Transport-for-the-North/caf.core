# -*- coding: utf-8 -*-
"""
to test:
from pure enum inputs
from enum+custom inputs
subsets
exclusions
load
save
add segmentations
"""
# Built-Ins

# Third Party
import pytest
import pandas as pd
import numpy as np
from caf.base import segmentation

# Local Imports
# pylint: disable=import-error,wrong-import-position
# Local imports here
# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #


# # # CLASSES # # #
@pytest.fixture(scope="session", name="vanilla_seg")
def fix_vanilla_segmentation():
    input = segmentation.SegmentationInput(
        enum_segments=["ca", "m", "gender_3"],
        naming_order=["ca", "m", "gender_3"],
    )
    return segmentation.Segmentation(input)


@pytest.fixture(scope="session", name="expected_vanilla_ind")
def fix_exp_vanilla_ind():
    ca = [1, 2]
    g = [1, 2, 3]
    m = [1, 2, 3, 4, 5, 6]
    ind = pd.MultiIndex.from_product([ca, m, g], names=["ca", "m", "gender_3"])
    return ind


@pytest.fixture(scope="session", name="nam_ord_seg")
def fix_nam_ord_seg():
    input = segmentation.SegmentationInput(
        enum_segments=["ca", "m", "gender_3"],
        naming_order=["ca", "gender_3", "m"],
    )
    return segmentation.Segmentation(input)


@pytest.fixture(scope="session", name="exp_nam_ord")
def fix_exp_nam_ord():
    ca = [1, 2]
    g = [1, 2, 3]
    m = [1, 2, 3, 4, 5, 6]
    ind = pd.MultiIndex.from_product([ca, g, m], names=["ca", "gender_3", "m"])
    return ind


@pytest.fixture(scope="session", name="seg_with_excl")
def fix_seg_with_excl():
    conf = segmentation.SegmentationInput(
        enum_segments=["gender_3", "soc", "ca"], naming_order=["gender_3", "soc", "ca"]
    )
    return segmentation.Segmentation(conf)


@pytest.fixture(scope="session", name="expected_excl")
def fix_exp_excl():
    return pd.MultiIndex.from_tuples(
        [
            (1, 4, 1),
            (1, 4, 2),
            (2, 1, 1),
            (2, 1, 2),
            (2, 2, 1),
            (2, 2, 2),
            (2, 3, 1),
            (2, 3, 2),
            (2, 4, 1),
            (2, 4, 2),
            (3, 1, 1),
            (3, 1, 2),
            (3, 2, 1),
            (3, 2, 2),
            (3, 3, 1),
            (3, 3, 2),
            (3, 4, 1),
            (3, 4, 2),
        ]
    )


@pytest.fixture(scope="session", name="subset_seg")
def fix_subset_seg():
    conf = segmentation.SegmentationInput(
        enum_segments=["p", "gender_3", "ns_sec"],
        subsets={"p": list(range(1, 9))},
        naming_order=["p", "gender_3", "ns_sec"],
    )
    return segmentation.Segmentation(conf)


@pytest.fixture(scope="session", name="exp_subset")
def fix_exp_sub():
    p = [1, 2, 3, 4, 5, 6, 7, 8]
    g = [1, 2, 3]
    ns = [1, 2, 3, 4, 5]
    return pd.MultiIndex.from_product([p, g, ns], names=["p", "gender_3", "ns_sec"])


@pytest.fixture(scope="session", name="exp_add")
def fix_add_exp():
    conf = segmentation.SegmentationInput(
        enum_segments=["gender_3", "soc", "ca", "p", "ns_sec"],
        subsets={"p": list(range(1, 9))},
        naming_order=["gender_3", "soc", "ca", "p", "ns_sec"],
    )
    return segmentation.Segmentation(conf)


class TestInd:
    def test_vanilla_ind(self, vanilla_seg, expected_vanilla_ind):
        assert expected_vanilla_ind.equal_levels(vanilla_seg.ind())

    def test_name_order(self, nam_ord_seg, exp_nam_ord):
        assert exp_nam_ord.equal_levels(nam_ord_seg.ind())

    # @pytest.mark.parametrize("segmentation", ["excl_segmentation", "excl_segmentation_rev"])

    def test_exclusions(self, seg_with_excl, expected_excl):
        assert seg_with_excl.ind().equal_levels(expected_excl)

    def test_subset(self, subset_seg, exp_subset):
        assert exp_subset.equal_levels(subset_seg.ind())

    @pytest.mark.parametrize(
        "seg_str", ["subset_seg", "seg_with_excl", "nam_ord_seg", "vanilla_seg"]
    )
    def test_io(self, seg_str, main_dir, request):
        """Check that segmentation objects can be saved and loaded"""
        seg = request.getfixturevalue(seg_str)
        seg.save(main_dir / "meta.yml", "yaml")
        read = segmentation.Segmentation.load(main_dir / "meta.yml", "yaml")
        assert read == seg

    def test_add(self, seg_with_excl, subset_seg, exp_add):
        added = seg_with_excl + subset_seg
        assert added == exp_add

    def test_agg(self, vanilla_seg):
        aggregated = vanilla_seg.aggregate(["ca", "m"])
        conf = segmentation.SegmentationInput(
            enum_segments=["ca", "m"], naming_order=["ca", "m"]
        )
        assert aggregated == segmentation.Segmentation(conf)
