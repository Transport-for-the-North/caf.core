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
from math import isclose

# Third Party
from caf.base import data_structures, segmentation
from caf.base.segments import SegmentsSuper
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
        index=basic_segmentation_1.ind(),
        columns=min_zoning.zone_ids,
    )


@pytest.fixture(name="dvec_data_2", scope="session")
def fix_data_2(basic_segmentation_2, min_zoning):
    return pd.DataFrame(
        data=np.random.rand(9, 5),
        index=basic_segmentation_2.ind(),
        columns=min_zoning.zone_ids,
    )


@pytest.fixture(name="single_seg_dvec", scope="session")
def fix_single_seg(min_zoning):
    seg_conf = segmentation.SegmentationInput(enum_segments=["p"], naming_order=["p"])
    seg = segmentation.Segmentation(seg_conf)
    data = pd.DataFrame(
        data=np.random.rand(15, 5), index=seg.ind(), columns=min_zoning.zone_ids
    )
    return data_structures.DVector(
        segmentation=seg, import_data=data, zoning_system=min_zoning
    )


@pytest.fixture(name="no_zone_dvec_1", scope="session")
def fix_no_zone_1(basic_segmentation_1):
    data = pd.Series(
        np.random.rand(
            18,
        ),
        index=basic_segmentation_1.ind(),
    )
    return data_structures.DVector(segmentation=basic_segmentation_1, import_data=data)


@pytest.fixture(name="no_zone_dvec_2", scope="session")
def fix_no_zone_2(basic_segmentation_2):
    data = pd.Series(
        np.random.rand(
            9,
        ),
        index=basic_segmentation_2.ind(),
    )
    return data_structures.DVector(segmentation=basic_segmentation_2, import_data=data)


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


@pytest.fixture(name="expected_trans", scope="session")
def fix_exp_trans(basic_dvec_1, min_zoning_2):
    orig_data = basic_dvec_1.data
    trans_data = pd.DataFrame(
        index=orig_data.index,
        data={
            1: orig_data[1],
            2: orig_data[2],
            3: orig_data[3],
            4: orig_data[4] + orig_data[5],
        },
    )
    return data_structures.DVector(
        segmentation=basic_dvec_1.segmentation,
        zoning_system=min_zoning_2,
        import_data=trans_data,
    )


# # # CLASSES # # #


# # # FUNCTIONS # # #
class TestDvec:
    @pytest.mark.parametrize("dvec", ["basic_dvec_1", "basic_dvec_2"])
    @pytest.mark.parametrize("subset", [None, [1, 2, 3]])
    @pytest.mark.parametrize("method", ["split", "duplicate"])
    def test_add_segments(self, dvec, subset, method, request):
        dvec_arg = request.getfixturevalue(dvec).copy()
        segment = SegmentsSuper("tp").get_segment(subset=subset)
        out_dvec = dvec_arg.add_segments([segment], split_method=method)
        new_seg_len = len(segment)
        if method == "split":
            assert isclose(out_dvec.data.values.sum(), dvec_arg.data.values.sum())
        else:
            assert isclose(
                out_dvec.data.values.sum(), dvec_arg.data.values.sum() * new_seg_len
            )

    def test_add_segment_exclusion(self, basic_dvec_1):
        segment = SegmentsSuper("soc").get_segment()
        out_dvec = basic_dvec_1.add_segments([segment], split_method="split")
        assert isclose(out_dvec.data.values.sum(), basic_dvec_1.data.values.sum())

    @pytest.mark.parametrize("dvec", ["basic_dvec_1", "basic_dvec_2", "single_seg_dvec"])
    def test_io(self, dvec, main_dir, request):
        dvec = request.getfixturevalue(dvec)
        dvec.save(main_dir / "dvector.h5")
        read_dvec = data_structures.DVector.load(main_dir / "dvector.h5")
        assert read_dvec == dvec

    @pytest.mark.parametrize(
        "dvec_1_str", ["basic_dvec_1", "basic_dvec_2", "no_zone_dvec_1", "no_zone_dvec_2"]
    )
    @pytest.mark.parametrize(
        "dvec_2_str", ["basic_dvec_1", "basic_dvec_2", "no_zone_dvec_1", "no_zone_dvec_2"]
    )
    def test_add(self, dvec_1_str, dvec_2_str, request):
        dvec_1 = request.getfixturevalue(dvec_1_str)
        dvec_2 = request.getfixturevalue(dvec_2_str)
        added_dvec = dvec_1 + dvec_2
        dvec_1_data = dvec_1.data
        dvec_2_data = dvec_2.data
        try:
            added_df = dvec_1_data.add(dvec_2_data, axis="index")
        except:
            added_df = dvec_2_data.add(dvec_1_data, axis="index")
        if added_df.index.names != added_dvec.segmentation.naming_order:
            added_df.index = added_df.index.reorder_levels(
                added_dvec.segmentation.naming_order
            )
        assert added_dvec.data.sort_index().equals(added_df.sort_index())

    @pytest.mark.parametrize(
        "dvec_1_str", ["basic_dvec_1", "basic_dvec_2", "no_zone_dvec_1", "no_zone_dvec_2"]
    )
    @pytest.mark.parametrize(
        "dvec_2_str", ["basic_dvec_1", "basic_dvec_2", "no_zone_dvec_1", "no_zone_dvec_2"]
    )
    def test_sub(self, dvec_1_str, dvec_2_str, request):
        dvec_1 = request.getfixturevalue(dvec_1_str)
        dvec_2 = request.getfixturevalue(dvec_2_str)
        added_dvec = dvec_1 - dvec_2
        dvec_1_data = dvec_1.data
        dvec_2_data = dvec_2.data
        try:
            added_df = dvec_1_data.sub(dvec_2_data, axis="index")
        except:
            added_df = dvec_2_data.sub(dvec_1_data, axis="index")
        if added_df.index.names != added_dvec.segmentation.naming_order:
            added_df.index = added_df.index.reorder_levels(
                added_dvec.segmentation.naming_order
            )
        assert added_dvec.data.sort_index().equals(added_df.sort_index())

    @pytest.mark.parametrize(
        "dvec_1_str", ["basic_dvec_1", "basic_dvec_2", "no_zone_dvec_1", "no_zone_dvec_2"]
    )
    @pytest.mark.parametrize(
        "dvec_2_str", ["basic_dvec_1", "basic_dvec_2", "no_zone_dvec_1", "no_zone_dvec_2"]
    )
    def test_mul(self, dvec_1_str, dvec_2_str, request):
        dvec_1 = request.getfixturevalue(dvec_1_str)
        dvec_2 = request.getfixturevalue(dvec_2_str)
        added_dvec = dvec_1 * dvec_2
        dvec_1_data = dvec_1.data
        dvec_2_data = dvec_2.data
        try:
            added_df = dvec_1_data.mul(dvec_2_data, axis="index")
        except:
            added_df = dvec_2_data.mul(dvec_1_data, axis="index")
        if added_df.index.names != added_dvec.segmentation.naming_order:
            added_df.index = added_df.index.reorder_levels(
                added_dvec.segmentation.naming_order
            )
        assert added_dvec.data.sort_index().equals(added_df.sort_index())

    @pytest.mark.parametrize(
        "dvec_1_str", ["basic_dvec_1", "basic_dvec_2", "no_zone_dvec_1", "no_zone_dvec_2"]
    )
    @pytest.mark.parametrize(
        "dvec_2_str", ["basic_dvec_1", "basic_dvec_2", "no_zone_dvec_1", "no_zone_dvec_2"]
    )
    def test_div(self, dvec_1_str, dvec_2_str, request):
        dvec_1 = request.getfixturevalue(dvec_1_str)
        dvec_2 = request.getfixturevalue(dvec_2_str)
        added_dvec = dvec_1 / dvec_2
        dvec_1_data = dvec_1.data
        dvec_2_data = dvec_2.data
        try:
            added_df = dvec_1_data.div(dvec_2_data, axis="index")
        except:
            added_df = dvec_2_data.div(dvec_1_data, axis="index")
        if added_df.index.names != added_dvec.segmentation.naming_order:
            added_df.index = added_df.index.reorder_levels(
                added_dvec.segmentation.naming_order
            )
        assert added_dvec.data.sort_index().equals(added_df.sort_index())

    def test_trans(self, basic_dvec_1, test_trans, min_zoning_2, expected_trans, main_dir):
        translation = basic_dvec_1.translate_zoning(min_zoning_2, cache_path=main_dir)
        back_trans = translation.translate_zoning(
            basic_dvec_1.zoning_system, cache_path=main_dir
        )
        assert translation == expected_trans

    def test_agg(self, basic_dvec_1):
        aggregated = basic_dvec_1.aggregate(["gender_3"])
        grouped = basic_dvec_1.data.groupby(level="gender_3").sum()
        assert grouped.equals(aggregated.data)
