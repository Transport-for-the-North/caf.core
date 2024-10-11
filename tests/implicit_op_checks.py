# -*- coding: utf-8 -*-
"""
Example tests cases for implicit operations for Segmentation levels
"""
from __future__ import annotations

import enum
import dataclasses

from typing import Any
from typing import Optional

import pandas as pd


@dataclasses.dataclass
class InputAndResults:
    """Holds the input and expected output for each 'test'."""

    df1: pd.DataFrame
    df2: pd.DataFrame

    expected_output: Optional[pd.DataFrame] = None
    expect_fail: bool = False


@dataclasses.dataclass
class Segment:
    """Example segment class

    I'm using strings here as a unique ID with `name`, but imagine there's
    something better.
    """

    name: str
    vals: list[Any]

    def get_subset_vals(self, vals: list[Any]) -> list[Any]:
        """Get a subset of this segmentations"""
        for x in vals:
            if x not in self.vals:
                raise ValueError(f"Not valid value for {self.name}")

        return vals


@enum.unique
class SegmentDef(enum.Enum):
    PURPOSE = "p"
    MODE = "m"
    GENDER = "g"
    SOC = "soc"

    def get(self) -> Segment:
        if self == SegmentDef.PURPOSE:
            vals = range(1, 9)
            # vals = range(1, 5)
        elif self == SegmentDef.MODE:
            vals = range(1, 7)
            # vals = range(1, 3)
        elif self == SegmentDef.MODE:
            vals = range(1, 4)
        elif self == SegmentDef.MODE:
            vals = range(1, 5)
        else:
            raise ValueError(f"No definition exists for {self} SegmentDef")

        return Segment(
            name=self.value,
            vals=vals,
        )


def index_to_df(pd_index: pd.Index | pd.MultiIndex) -> pd.DataFrame:
    return pd.DataFrame(index=pd_index).reset_index()


def df_from_segment_product(s1: Segment, s2: Segment) -> pd.DataFrame:
    idx = pd.MultiIndex.from_product((s1.vals, s2.vals), names=(s1.name, s2.name))
    return index_to_df(idx)


def simple_all_combinations() -> InputAndResults:
    """Simple example where all combinations are ok."""
    purpose = SegmentDef.PURPOSE.get()
    mode = SegmentDef.MODE.get()
    df = df_from_segment_product(purpose, mode)

    return InputAndResults(
        df1=df,
        df2=df,
        expected_output=df,
    )


def simple_all_combinations_perm() -> InputAndResults:
    """Simple example where all combinations are ok."""
    purpose = SegmentDef.PURPOSE.get()
    mode = SegmentDef.MODE.get()
    df = df_from_segment_product(purpose, mode)
    df2 = df_from_segment_product(mode, purpose)

    return InputAndResults(
        df1=df,
        df2=df2,
        expected_output=df,
    )


def simple_subset() -> InputAndResults:
    """Simple example where all combinations are ok."""
    purpose = SegmentDef.PURPOSE.get()
    mode = SegmentDef.MODE.get()
    sub_mode = mode.get_subset_vals([1, 2])
    df = df_from_segment_product(purpose, mode)
    df1 = df_from_segment_product(purpose, sub_mode)

    # TODO: Is this compatible? What would we expect the result to be?
    #  Would it almost do the opposite of numpy and subset the larger dataset to match the subset?
    #  Would this cause any implicit problems?
    #  If we don't allow this, how do we define subsets?
    #    They would have to be a different class with no relationship back to its parent.

    return InputAndResults(
        df1=df,
        df2=df1,
        expect_fail=True,
    )


def non_product_segmentations():
    # At Tfn, g segment cannot be defined with anything other than soc 4
    df = pd.DataFrame(
        [
            {"g": 1, "soc": 4},
            {"g": 2, "soc": 1},
            {"g": 3, "soc": 1},
            {"g": 2, "soc": 2},
            {"g": 3, "soc": 2},
            {"g": 2, "soc": 3},
            {"g": 3, "soc": 3},
            {"g": 2, "soc": 4},
            {"g": 3, "soc": 4},
        ]
    )

    # If we merge with anything that contains JUST soc or g, there's no problem

    # But if we merge on something that contains both, there's a potential issue
    # df2 and 3 below are fine to combine with, they follow the TfN rules

    df2 = pd.DataFrame(
        [
            {"g": 1, "soc": 4},
            {"g": 2, "soc": 4},
            {"g": 3, "soc": 4},
        ]
    )

    df3 = pd.DataFrame(
        [
            {"g": 2, "soc": 3},
            {"g": 3, "soc": 3},
        ]
    )

    # TODO: What do we do if a new segment comes along that doesn't follow this rule?
    #  TfN might update it's segments, or another person might have different rules.
    #  See DF4 for an example
    #  .
    #  Really, we'd want this to throw an error, as they are incompatible
    #  But how would we detect this?
    #  .
    #  Individually, the soc and gender segments are valid - they only contain valid numbers.
    #    They are also both the same SOC and GENDER segments
    #      There's nothing fundamentally different about them. Thy contain the same values.
    #      We could give it a different name, but the difference then is just semantics
    #      We would be ignoring the relationship that makes it unique
    #      There's no way of saying "In this segmentation, G1 only uses Soc4"
    #  .
    #  We could do an outer merge and check for NA values. This wouldn't work when only those two segments are being used.
    #  An exhaustive check (of every row) every time would be too expensive, especially for large segments.

    df4 = pd.DataFrame(
        [
            {"g": 1, "soc": 1},
            {"g": 1, "soc": 2},
            {"g": 1, "soc": 3},
            {"g": 1, "soc": 4},
            {"g": 2, "soc": 1},
            {"g": 2, "soc": 2},
            {"g": 2, "soc": 3},
            {"g": 2, "soc": 4},
        ]
    )


def simple_merge_method():
    """Works for all combinations, but fails where subsets."""

    def combine_fn(io: InputAndResults) -> pd.DataFrame:
        return pd.merge(io.df1, io.df2)

    # Fine
    data = [simple_all_combinations(), simple_all_combinations_perm()]
    print("---" * 20)
    for io in data:
        df = combine_fn(io)
        pd.testing.assert_frame_equal(io.expected_output, df)


def main():
    simple_merge_method()


if __name__ == "__main__":
    main()
