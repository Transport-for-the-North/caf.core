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
from dataclasses import dataclass
# Third Party
import pandas as pd
from caf.core.config_base import BaseConfig
from pydantic import validator
import numpy as np
# Local Imports
# pylint: disable=import-error,wrong-import-position
# Local imports here
# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #

# # # CLASSES # # #
@dataclass
class Exclusion:
    seg_name: str
    own_val: int
    other_vals: set[int]


class Segment(BaseConfig):
    name: str
    values: dict[str, int]
    exclusions: list[Exclusion] = None

    @property
    def exclusion_segs(self):
        if self.exclusions:
            return [seg.seg_name for seg in self.exclusions]
        else:
            return None

    @property
    def _exclusions(self):
        return {excl.seg_name: excl for excl in self.exclusions}

    def drop_indices(self, other_seg: str):
        if other_seg not in self.exclusion_segs:
            return None
        else:
            ind_tuples = []
            for excl in self.exclusions:
                if excl.seg_name == other_seg:
                    for other in excl.other_vals:
                        ind_tuples.append((excl.own_val, other))
            drop_ind = pd.MultiIndex.from_tuples(ind_tuples)
            return drop_ind


class Segmentation(BaseConfig):
    segments: list[Segment]
    naming_order: list[str]

    @property
    def names(self):
        return [seg.name for seg in self.segments]

    @property
    def seg_vals(self):
        return [seg.values.values() for seg in self.segments]

    @property
    def seg_keys(self):
        return [seg.values.keys() for seg in self.segments]

    # @validator('naming_order')
    # def names_in_segments(cls, v: set[str]):
    #     if v != cls.names:
    #         raise ValueError(f"The names in naming_order do not match the names of the segments.")
    #     return v

    @property
    def ind(self):
        index = pd.MultiIndex.from_product(self.seg_vals, names=self.names)
        df = pd.DataFrame(index=index)
        drop_iterator = self.naming_order.copy()
        for own_seg in self.segments:
            for other_seg in drop_iterator:
                if other_seg==own_seg.name:
                    pass
                if own_seg.exclusion_segs:
                    if other_seg in own_seg.exclusion_segs:
                        dropper = own_seg.drop_indices(other_seg)
                        df = df.reset_index().set_index([own_seg.name, other_seg])
                        mask = ~df.index.isin(dropper)
                        df = df[mask]
        df = df.reorder_levels(self.naming_order)
        return df.index

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

    # def __mul__(self, other: Segmentation) -> Segmentation:
    #     overlap = [i for i in other.segments if i in self.segments]
    #     merge_cols = [seg.name for seg in overlap]
    #     joined = pd.merge(self.df.reset_index(), other.df.reset_index(), on=merge_cols, suffixes=['_left', '_right'], how='inner')
    #     joined['data'] = joined['data_left'] * joined['data_right']



# # # FUNCTIONS # # #
