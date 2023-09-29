# -*- coding: utf-8 -*-
"""
Created on: 11/09/2023
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:

"""
import enum
import pandas as pd
from caf.toolkit import BaseConfig
import numpy as np
from pathlib import Path
from dataclasses import dataclass


# # # CONSTANTS # # #
# # # CLASSES # # #
@dataclass
class Exclusion:
    """
    seg_name: Name of the other segment this exclusion applies to
    own_val: The value for self segmentation which has exclusions in other
    other_vals: Values in other segmentation incompatible with 'own_val'.
    """

    seg_name: str
    own_val: int
    other_vals: set[int]

    def build_index(self):
        """
        Returns an index formed of the exclusions.
        """
        tups = [(self.own_val, other) for other in self.other_vals]
        return pd.MultiIndex.from_tuples(tups)


class Segment(BaseConfig):
    """
    Class containing info on a Segment, which combined with other Segments form
    a segmentation.

    Parameters
    ----------
    name: the name of the segmentation. Generally this is short form (e.g. 'p'
    instead of 'purpose')
    values: The values forming the segment. Keys are the values, and values are
    descriptions, e.g. for 'p', 1: 'HB work'. Descriptions don't tend to get used
    in DVectors so can be as verbose as desired for clarity.
    exclusions: Define incompatibilities between segments. See Exclusion class
    """

    name: str
    values: dict[int, str]
    exclusions: list[Exclusion] = None

    class Config:
        arbitrary_types_allowed = True

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


class SegmentsSuper(enum.Enum):
    """
    Getter for predefined segments. This should be where segments forming
    segmentations come from. In most cases if a segment is not defined here it
    should be added, rather than defined as a custom segment in a Segmentation
    """
    PURPOSE = "p"
    TIMEPERIOD = "tp"
    MODE = "m"
    GENDER = "g"
    SOC = "soc"
    SIC = "sic"
    CA = "ca"
    TFN_AT = "tfn_at"
    USERCLASS = "uc"
    NS = "ns"

    @classmethod
    def values(cls):
        return [e.value for e in cls]

    def get_segment(self, subset: list[int] = None):
        """
        method to get a segments.

        Parameters
        ----------
        subset: Define a subset of the segment being got. The integers in subset
        must appear in the asked for segment.
        """
        match self:
            case SegmentsSuper.PURPOSE:
                segmentation = Segment(
                    name=self.value,
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
            case SegmentsSuper.TIMEPERIOD:
                segmentation = Segment(
                    name=self.value,
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
            case SegmentsSuper.MODE:
                segmentation = Segment(
                    name=self.value,
                    values={
                        1: "Walk",
                        2: "Cycle",
                        3: "Car driver",
                        4: "Car passenger",
                        5: "Bus / Coach",
                        6: "Rail / underground",
                    },
                )
            case SegmentsSuper.GENDER:
                segmentation = Segment(
                    name=self.value,
                    values={1: "Child", 2: "Male", 3: "Female"},
                    exclusions=[
                        Exclusion(
                            seg_name=SegmentsSuper.SOC.value, own_val=1, other_vals=[1, 2, 3]
                        )
                    ],
                )
            case SegmentsSuper.SOC:
                segmentation = Segment(
                    name=self.value,
                    values={
                        1: "High Skilled",
                        2: "High Skilled",
                        3: "High Skilled",
                        4: "Skilled",
                    },
                )
            case SegmentsSuper.CA:
                segmentation = Segment(name=self.value, values={1: "dummy", 2: "dummy"})
            case SegmentsSuper.NS:
                segmentation = Segment(
                    name=self.value,
                    values={1: "dummy", 2: "dummy", 3: "dummy", 4: "dummy", 5: "dummy"},
                )

        if subset:
            segmentation.values = {i: j for i, j in segmentation.values.items() if i in subset}
        return segmentation


# # # FUNCTIONS # # #
