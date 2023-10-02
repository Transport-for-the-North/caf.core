# -*- coding: utf-8 -*-
"""
Created on: 13/09/2023
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:

"""
# Built-Ins
from pathlib import Path

# Third Party
import pytest

# Local Imports
# pylint: disable=import-error,wrong-import-position
# Local imports here
from caf.core import segmentation, segments

# pylint: enable=import-error,wrong-import-position


# # # CONSTANTS # # #
@pytest.fixture(name="main_dir", scope="session")
def fixture_main_dir(tmp_path_factory) -> Path:
    """
    Temporary path for I/O.

    Parameters
    ----------
    tmp_path_factory

    Returns
    -------
    Path: file path used for all saving and loading of files within the tests
    """
    path = tmp_path_factory.mktemp("main")
    return path


@pytest.fixture(scope="session", name="gender_seg")
def fix_gender():
    return segments.Segment(
        name="g",
        values={1: "Child", 2: "Male", 3: "Female"},
        exclusions=[segments.Exclusion(seg_name="soc", own_val=1, other_vals=[1, 2, 3])],
    )


@pytest.fixture(scope="session", name="soc_seg")
def fix_soc():
    return segments.Segment(
        name="soc",
        values={
            1: "High Skilled",
            2: "High Skilled",
            3: "High Skilled",
            4: "Skilled",
            5: "Skilled",
            6: "Skilled",
            7: "Low Skilled",
            8: "Low Skilled",
            9: "Low Skilled",
        },
    )


@pytest.fixture(scope="session", name="mode_seg")
def fix_mode():
    return segments.Segment(
        name="m",
        values={
            1: "Walk",
            2: "Cycle",
            3: "Car driver",
            4: "Car passenger",
            5: "Bus / Coach",
            6: "Rail / underground",
        },
    )


@pytest.fixture(scope="session", name="tp_seg")
def fix_tp():
    return segments.Segment(
        name="tp",
        values={
            1: "Weekday AM peak period (0700 - 0959)",
            2: "Weekday Inter peak period (1000 - 1559)",
            3: "Weekday PM peak period (1600 - 1859)",
            4: "Weekday Off peak (0000 - 0659 and 1900 - 2359)",
            5: "Saturdays (all times of day)",
            6: "Sundays (all times of day)",
            7: "Average Weekday",
            8: "Average Day",
        },
    )


@pytest.fixture(scope="session", name="purpose_seg")
def fix_purpose():
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
            11: "NHB Work",
            12: "NHB Employers Business (EB)",
            13: "NHB Education",
            14: "NHB Shopping",
            15: "NHB Personal Business (PB)",
            16: "NHB Recreation / Social",
            18: "NHB Holiday / Day trip",
        },
    )


@pytest.fixture(name="seg_1", scope="session")
def fixture_seg_1():
    return segmentation.Segment(name="test seg 1", values={1: "A", 2: "B", 3: "C", 4: "D"})


@pytest.fixture(name="seg_2", scope="session")
def fixture_seg_2():
    return segmentation.Segment(name="test seg 2", values={1: "X", 2: "Y", 3: "Z"})


@pytest.fixture(name="seg_3", scope="session")
def fixture_seg_3():
    return segmentation.Segment(name="test seg 3", values={1: "E", 2: "F", 3: "G", 4: "H"})


@pytest.fixture(name="seg_4", scope="session")
def fixture_seg_4():
    return segmentation.Segment(name="test seg 4", values={1: "I", 2: "J", 3: "K"})


@pytest.fixture(name="basic_segmentation", scope="session")
def fixture_basic_segmentation(seg_1, seg_2):
    segs = [seg_1, seg_2]
    order = ["test seg 2", "test seg 1"]
    return segmentation.Segmentation(segments=segs, naming_order=order)


@pytest.fixture(name="excl_segmentation", scope="session")
def fixture_excl_segmentation(seg_1, seg_2):
    seg_excl = seg_1.copy()
    seg_excl.exclusions = [
        segmentation.Exclusion(seg_name="test seg 2", own_val=1, other_vals=[2, 3])
    ]
    order = ["test seg 1", "test seg 2"]
    return segmentation.Segmentation(segments=[seg_excl, seg_2], naming_order=order)


@pytest.fixture(name="excl_segmentation_rev", scope="session")
def fixture_excl_segmentation_rev(seg_1, seg_2):
    seg_excl = seg_2.copy()
    seg_excl.exclusions = [
        segmentation.Exclusion(seg_name="test seg 1", own_val=2, other_vals=[1]),
        segmentation.Exclusion(seg_name="test seg 1", own_val=3, other_vals=[1]),
    ]
    order = ["test seg 1", "test seg 2"]
    return segmentation.Segmentation(segments=[seg_1, seg_excl], naming_order=order)


# # # CLASSES # # #

# # # FUNCTIONS # # #
