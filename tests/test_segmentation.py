# -*- coding: utf-8 -*-
"""
Created on: 19/09/2023
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:

"""
# Built-Ins

# Third Party
import pytest
import pandas as pd
import numpy as np
from caf.core import segmentation

# Local Imports
# pylint: disable=import-error,wrong-import-position
# Local imports here
# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #


# # # CLASSES # # #
@pytest.fixture(scope="session", name="vanilla_seg")
def fix_vanilla_segmentation(seg_1, seg_2, seg_3, seg_4):
    input = segmentation.SegmentationInput(
        segments=[seg_1, seg_2, seg_3, seg_4],
        naming_order=["test seg 1", "test seg 2", "test seg 3", "test seg 4"],
    )
    return segmentation.Segmentation(input)


@pytest.fixture(scope="session", name="expected_vanilla_ind")
def fix_exp_vanilla_ind():
    a = [1, 2, 3, 4]
    b = [1, 2, 3]
    ind = pd.MultiIndex.from_product(
        [a, b, a, b], names=["test seg 1", "test seg 2", "test seg 3", "test seg 4"]
    )
    return ind


@pytest.fixture(scope="session", name="build_basic")
def fix_build_basic():
    return segmentation.Segmentation._build_seg(segs=["p", "g", "m"], naming_order=["p", "g", "m"])


@pytest.fixture(scope="session", name="build_naming_order")
def fix_build_naming_order():
    return segmentation.Segmentation._build_seg(segs=["p", "g", "m"], naming_order=["g", "p", "m"])


@pytest.fixture(scope="session", name="expected_basic_build")
def fix_expected_basic(purpose_seg, gender_seg, mode_seg):
    input = segmentation.SegmentationInput(segments=[purpose_seg, gender_seg, mode_seg],
                                           naming_order=["g", "p", "m"])
    return segmentation.Segmentation(input)

@pytest.fixture(scope="session", name="expected_name_build")
def fix_expected_name(purpose_seg, gender_seg, mode_seg):
    input = segmentation.SegmentationInput(segments=[purpose_seg, gender_seg, mode_seg],
                                           naming_order=["p", "g", "m"])
    return segmentation.Segmentation(input)


class TestInd:
    def test_vanilla_ind(self, vanilla_seg, expected_vanilla_ind):
        assert expected_vanilla_ind.equal_levels(vanilla_seg.ind)

    @pytest.mark.parametrize("segmentation", ["excl_segmentation", "excl_segmentation_rev"])
    def test_exclusions(self, segmentation, expected_excl_ind, request):
        seg = request.getfixturevalue(segmentation)
        print(seg.ind)
        assert seg.ind.equals(expected_excl_ind)


class TestBuild:
    @pytest.mark.parametrize("built,expected",
                             [("build_basic", "expected_basic_build"),
                              ("build_naming_order", "expected_name_build")])
    def test_build(self, built, expected, request):
        built = request.getfixturevalue(built)
        expected = request.getfixturevalue(expected)
        assert built == expected



# # # FUNCTIONS # # #
