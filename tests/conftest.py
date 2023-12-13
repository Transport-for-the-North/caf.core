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
import pandas as pd

# Local Imports
# pylint: disable=import-error,wrong-import-position
# Local imports here
from caf.core import segmentation, zoning

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


@pytest.fixture(name="min_zoning", scope="session")
def fix_min_zoning():
    data = {"zone_id": [1, 2, 3, 4, 5], "zone_name": ["a", "b", "c", "d", "e"]}
    name = "zone_1"
    meta = zoning.ZoningSystemMetaData(name=name)
    return zoning.ZoningSystem(name=name, unique_zones=pd.DataFrame(data), metadata=meta)


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


@pytest.fixture(name="basic_segmentation_1", scope="session")
def fixture_basic_segmentation_1():
    conf = segmentation.SegmentationInput(enum_segments=["g", "m"], naming_order=["g", "m"])
    return segmentation.Segmentation(conf)


@pytest.fixture(name="basic_segmentation_2", scope="session")
def fixture_basic_segmentation_2():
    conf = segmentation.SegmentationInput(
        enum_segments=["g", "soc"], naming_order=["g", "soc"]
    )
    return segmentation.Segmentation(conf)


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


@pytest.fixture(name="test_trans", scope="session")
def fix_test_trans(main_dir):
    data = {
        "zone_1_id": ["a", "b", "c", "d", "e"],
        "zone_2_id": ["w", "x", "y", "z", "z"],
        "zone_1_to_zone_2": [1, 1, 1, 1, 1],
        "zone_2_to_zone_1": [1, 1, 1, 0.5, 0.5],
    }
    save_path = main_dir / "zone_1_zone_2"
    save_path.mkdir(exist_ok=True, parents=True)
    pd.DataFrame(data).to_csv(save_path / "zone_1_to_zone_2_spatial.csv", index=False)
    return pd.DataFrame(data)


@pytest.fixture(name="min_zoning_2", scope="session")
def fix_min_zoning_2():
    data = {"zone_id": [1, 2, 3, 4], "zone_name": ["w", "x", "y", "z"]}
    name = "zone_2"
    meta = zoning.ZoningSystemMetaData(name=name)
    return zoning.ZoningSystem(name=name, unique_zones=pd.DataFrame(data), metadata=meta)


# # # CLASSES # # #

# # # FUNCTIONS # # #
