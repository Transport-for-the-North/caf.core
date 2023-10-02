# -*- coding: utf-8 -*-
"""
To test:

build from dataframe
build from old format
save
load
add
subtract
mul
div
aggregate
translate

"""
# Built-Ins
from pathlib import Path
import pytest

# Third Party
from caf.core import data_structures, segmentation
import pandas as pd
import numpy as np

# Local Imports
# pylint: disable=import-error,wrong-import-position
# Local imports here
# pylint: enable=import-error,wrong-import-position


# # # CONSTANTS # # #


@pytest.fixture(name="dvec_data_1", scope="session")
def fix_data_1(basic_segmentation_1, min_zoning):
    return pd.DataFrame(
        data=np.random.rand(18, 5),
        index=basic_segmentation_1.ind,
        columns=min_zoning.unique_zones["zone_name"],
    )


@pytest.fixture(name="dvec_data_2", scope="session")
def fix_data_2(basic_segmentation_2, min_zoning):
    return pd.DataFrame(
        data=np.random.rand(9, 5),
        index=basic_segmentation_2.ind,
        columns=min_zoning.unique_zones["zone_name"],
    )


@pytest.fixture(name="basic_dvec_1", scope="session")
def fix_basic_dvec_1(min_zoning, basic_segmentation_1, dvec_data_1):
    return data_structures.DVector(
        segmentation=basic_segmentation_1, zoning_system=min_zoning, import_data=dvec_data_1
    )


@pytest.fixture(name="basic_dvec_2", scope="session")
def fix_basic_dvec_2(min_zoning, basic_segmentation_2, dvec_data_2):
    return data_structures.DVector(
        segmentation=basic_segmentation_2, zoning_system=min_zoning, import_data=dvec_data_2
    )


# # # CLASSES # # #


# # # FUNCTIONS # # #
class TestDvec:
    def test_io(self, basic_dvec_1, main_dir):
        basic_dvec_1.save(main_dir / "dvector.h5")
        read_dvec = data_structures.DVector.load(main_dir / "dvector.h5")
        assert read_dvec == basic_dvec_1

    def test_add(self, basic_dvec_1, basic_dvec_2, dvec_data_1, dvec_data_2):
        added_dvec = basic_dvec_2 + basic_dvec_1
        added_df = dvec_data_2 + dvec_data_1
        assert added_dvec.data.equals(added_df)

    def test_sub(self, basic_dvec_1, basic_dvec_2, dvec_data_1, dvec_data_2):
        added_dvec = basic_dvec_2 - basic_dvec_1
        added_df = dvec_data_2 - dvec_data_1
        assert added_dvec.data.equals(added_df)

    def test_mul(self, basic_dvec_1, basic_dvec_2, dvec_data_1, dvec_data_2):
        added_dvec = basic_dvec_2 * basic_dvec_1
        added_df = dvec_data_2 * dvec_data_1
        assert added_dvec.data.equals(added_df)

    def test_div(self, basic_dvec_1, basic_dvec_2, dvec_data_1, dvec_data_2):
        added_dvec = basic_dvec_2 / basic_dvec_1
        added_df = dvec_data_2 / dvec_data_1
        assert added_dvec.data.equals(added_df)
