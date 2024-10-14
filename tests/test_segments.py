# -*- coding: utf-8 -*-
"""
Created on: 08/09/2023
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

# Local Imports
# pylint: disable=import-error,wrong-import-position
# Local imports here
# pylint: enable=import-error,wrong-import-position
from caf.base import segments

# # # CONSTANTS # # #


# # # CLASSES # # #
@pytest.fixture(scope="session", name="multi-index")
def fix_mult():
    # Define the index levels
    level_a = ["A", "B", "C", "D", "E", "F"]
    level_b = ["G", "H", "I", "J", "K", "L"]
    level_c = ["M", "N", "O", "P", "Q", "R"]
    level_d = ["S", "T", "U", "V", "W", "X"]

    # Create a MultiIndex
    index = pd.MultiIndex.from_tuples(
        [(a, b, c, d) for a, b, c, d in zip(level_a, level_b, level_c, level_d)],
        names=["a", "b", "c", "d"],
    )

    # Create a DataFrame with random data
    data = np.random.rand(6, 1)

    df = pd.DataFrame(data, index=index, columns=["RandomData"])


@pytest.fixture(scope="session", name="expected_excl_ind")
def fix_excl_ind():
    return pd.MultiIndex.from_tuples(
        [(1, 1), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3), (4, 1), (4, 2), (4, 3)],
        names=["test seg 1", "test seg 2"],
    )


@pytest.fixture(scope="session", name="get_gender_seg")
def fix_gender_seg():
    return segments.SegmentsSuper("gender_3").get_segment()


@pytest.fixture(scope="session", name="exp_gender_seg")
def fix_exp_gen():
    return segments.Segment(
        name="gender_3",
        values={1: "Child", 2: "Male", 3: "Female"},
        exclusions=[
            segments.Exclusion(
                other_name=segments.SegmentsSuper.SOC.value, exclusions={1: [1, 2, 3]}
            )
        ],
    )


@pytest.fixture(scope="session", name="get_hb_purpose")
def fix_hb_purpose():
    return segments.SegmentsSuper("p").get_segment(subset=list(range(1, 9)))


@pytest.fixture(scope="session", name="expected_hb_purpose")
def fix_exp_hb_purpose():
    return segments.Segment(
        name="p",
        values={
            1: "HB Work",
            2: "HB Employers Business (EB)",
            3: "HB Education",
            4: "HB Shopping",
            5: "HB Personal Business (PB)",
            6: "HB Recreation / Social",
            7: "HB Visiting friends and relatives",
            8: "HB Holiday / Day trip",
        },
    )


class TestSegmentsSuper:
    def test_get(self, get_gender_seg, exp_gender_seg):
        assert get_gender_seg.values == exp_gender_seg.values

    def test_get_subset(self, get_hb_purpose, expected_hb_purpose):
        assert get_hb_purpose == expected_hb_purpose
