# -*- coding: utf-8 -*-
"""
Created on: 07/09/2023
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:

"""
# Built-Ins
import warnings
from typing import Union
# Third Party
import pandas as pd
from caf.core.config_base import BaseConfig
import numpy as np
from pathlib import Path

# Local Imports
# pylint: disable=import-error,wrong-import-position
# Local imports here
from caf.core.segments import Segment, SegmentsSuper

# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #


# # # CLASSES # # #
class SegmentationInput(BaseConfig):
    """
    segments: list of segments, stored as Segment classes.
    naming_order: The order segment names will appear in segmentations
    """
    segments: list[Segment]
    naming_order: list[str]


class Segmentation:
    """
    Segmentation class for handling segmentation objects.

    Parameters
    ----------
    input: Instance of SegmentationInput. See that class for details.
    """
    def __init__(self, input: SegmentationInput):
        self.segments = input.segments
        self.naming_order = input.naming_order

    @property
    def seg_dict(self):
        """
        Method to access segments in dict form.
        """
        return {seg.name: seg for seg in self.segments}

    @property
    def names(self):
        """
        Returns the names of all segments.
        """
        return [seg.name for seg in self.segments]

    @property
    def seg_descriptions(self):
        """
        Returns a list of segment descriptions.
        """
        return [seg.values.values() for seg in self.segments]

    @property
    def seg_vals(self):
        return [seg.values.keys() for seg in self.segments]

    @property
    def ind(self):
        """
        Returns a pandas MultiIndex of the segmentation. This is by default just a product
        of all segments given, taking exclusions into account if any exist between segments.
        """
        index = pd.MultiIndex.from_product(self.seg_vals, names=self.names)
        df = pd.DataFrame(index=index)
        drop_iterator = self.naming_order.copy()
        for own_seg in self.segments:
            for other_seg in drop_iterator:
                if other_seg == own_seg.name:
                    pass
                if own_seg.exclusion_segs:
                    if other_seg in own_seg.exclusion_segs:
                        dropper = own_seg.drop_indices(other_seg)
                        df = df.reset_index().set_index([own_seg.name, other_seg])
                        mask = ~df.index.isin(dropper)
                        df = df[mask]
        df = df.reorder_levels(self.naming_order)
        return df.index

    @classmethod
    def _build_seg(
            cls,
            segs: list[str],
            naming_order: list[str],
            custom_segments: list[Segment] = None,
    ):
        """
        Internal method to build a segmentation from inputs.

        Parameters
        ----------
        segs: list of segment names.
        naming_order: Ordered list of segment names.
        custom_segments: List of fully defined segments which don't exist in the SegmentsSuper
        enum class

        Returns
        -------
        A segmentation built from these inputs.
        """
        segments = []
        for seg in segs:
            try:
                segments.append(SegmentsSuper(seg).get_segment())
            except ValueError:
                raise ValueError(
                    f"{seg} isn't defined in this package. Either define it there or"
                    f"create your own segment and pass it in as a custom segment."
                )
        if custom_segments:
            segments += custom_segments
        segminput = SegmentationInput(segments=segments, naming_order=naming_order)
        return cls(segminput)

    @classmethod
    def load_segmentation(cls, source: Union[Path, pd.DataFrame], segs: list[str] = None, naming_order: list[str] = None, custom_segs=None):
        """
        Load a segmentation from either a path to a csv, or a dataframe. This could either be
        purely a segmentation, or data with a segmentation index.

        Parameters
        ----------
        source: Either a path to a csv containing a segmentation or a dataframe containing a segmentation.
        If source is a dataframe the segmentation should not form the index.
        segs: A list of strings, which must match enumerations in SegmentsSuper. If this isn't
        provided then it will default to the column names in 'source'
        naming_order: The order for the segmentation. This will default to segs if not provided.
        custom_segs: Optional list of Segment objects if segments not in SegmentsSuper

        Returns
        -------
        Segmentation class
        """
        if isinstance(source, Path):
            df = pd.read_csv(source)
        else:
            df = source
        if segs is None:
            segs = list(df.columns)
        if naming_order is None:
            naming_order = segs
        built_segmentation = cls._build_seg(segs, naming_order, custom_segs)
        read_index = df.set_index(naming_order).index
        built_index = built_segmentation.ind
        if built_index.names != read_index.names:
            raise ValueError("The read in segmentation does not match the given parameters")
        if read_index.equal_levels(built_index):
            return built_segmentation
        for name in built_index.names:
            built_level = set(built_index.get_level_values(name))
            read_level = set(read_index.get_level_values(name))
            if read_level == built_level:
                continue
            if read_level.issubset(built_level):
                warnings.warn(f"Read in level {name} is a subset of the segment. If this was not"
                              f" expected check the input segmentation.")
                built_segmentation.seg_dict[name].values = {i: j for i, j in built_segmentation.seg_dict[name].values.items() if i in read_level}
                temp_df = pd.DataFrame(index = built_index, columns=[0]).reset_index().set_index(name)
                temp_df = temp_df.loc[read_level]
                built_index = temp_df.reset_index().set_index(naming_order).index
            else:
                raise ValueError(f'The segment for {name} does not match the inbuilt definition.'
                                 f'Check for mistakes in the read in segmentation, or redefine the'
                                 f'segment with a different name.')
        if read_index.equal_levels(built_index):
            return built_segmentation
        else:
            raise ValueError("The read in segmentation does not match the given parameters. The segment names"
                             " are correct, but segment values don't match. This could be due to an incompatibility"
                             " between segments which isn't reflected in the loaded in segmentation, or it could be"
                             " an out of date in built segmentation in the caf.core package.")


    def __copy__(self):
        """Returns a copy of this class"""
        return self.copy()

    def __eq__(self, other) -> bool:
        """Overrides the default implementation"""
        if not isinstance(other, Segmentation):
            return False

        if set(self.naming_order) != set(other.naming_order):
            return False

        if set(self.names) != set(other.names):
            return False

        return True

    def __ne__(self, other) -> bool:
        """Overrides the default implementation"""
        return not self.__eq__(other)

    def _mul_div_join(self, other, data_self: np.array, data_other: np.array):
        self_col_length = data_self.shape[1]
        other_col_length = data_other.shape[1]
        overlap = [i for i in other.segments if i in self.segments]
        merge_cols = [seg.name for seg in overlap]
        combindex = set(self.names + other.names)
        if (
            (self_col_length != other_col_length)
            & (self_col_length != 1)
            & (other_col_length != 1)
        ):
            raise ValueError("Different numbers of columns cannot be multiplied")
        if (self_col_length == 1) & (other_col_length != 1):
            df_self = pd.DataFrame(
                data=data_self, index=self.ind, columns=["data"]
            ).reset_index()
            df_other = pd.DataFrame(
                data=data_other, index=other.ind, columns=range(other_col_length)
            ).reset_index()
            joined = pd.merge([df_self, df_other], on=merge_cols, how="inner")
            joined.set_index(combindex, inplace=True)
            df_self = joined["data"]
            df_other = joined.drop("data", axis=1)
        elif (self_col_length != 1) & (other_col_length == 1):
            df_self = pd.DataFrame(
                data=data_self, index=self.ind, columns=range(self_col_length)
            ).reset_index()
            df_other = pd.DataFrame(
                data=data_other, index=other.ind, columns=["data"]
            ).reset_index()
            joined = pd.merge([df_self, df_other], on=merge_cols, how="inner")
            joined.set_index(combindex, inplace=True)
            df_other = joined["data"]
            df_self = joined.drop("data", axis=1)
        else:
            df_self = pd.DataFrame(
                data=data_self, index=self.ind, columns=range(self_col_length)
            ).reset_index()
            df_other = pd.DataFrame(
                data=data_other, index=other.ind, columns=range(other_col_length)
            ).reset_index()
            joined = pd.merge(
                [df_self, df_other], on=merge_cols, how="inner", suffixes=["_self", "_other"]
            )
            joined.set_index(combindex, inplace=True)
            df_self = joined[[col for col in joined.columns if col.endswith("_self")]]
            df_other = joined[[col for col in joined.columns if col.endswith("_other")]]

        return df_self, df_other

    def __mul__(self, other, data_self, data_other):
        df_self, df_other = self._mul_div_join(self, other, data_self, data_other)
        return df_self * df_other















# # # FUNCTIONS # # #
