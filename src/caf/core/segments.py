# -*- coding: utf-8 -*-
"""Module defining Segments class and enumeration."""
import enum
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import pydantic
from caf.toolkit import BaseConfig
from pydantic import ConfigDict


# # # CONSTANTS # # #
# # # CLASSES # # #
@dataclass
class Exclusion:
    """
    Class to define exclusions between segments.

    Parameters
    ----------
    seg_name: str
        Name of the other segment this exclusion applies to
    own_val: int
        The value for self segmentation which has exclusions in other
    other_vals: set[int]
        Values in other segmentation incompatible with 'own_val'.
    """

    seg_name: str
    own_val: int
    other_vals: set[int]

    def build_index(self):
        """Return an index formed of the exclusions."""
        tups = [(self.own_val, other) for other in self.other_vals]
        return pd.MultiIndex.from_tuples(tups)


class Segment(BaseConfig):
    """
    Class containing info on a Segment, which combined with other Segments form a segmentation.

    Parameters
    ----------
    name: str
        The name of the segmentation. Generally this is short form (e.g. 'p'
        instead of 'purpose')
    values: dict[int, str]
        The values forming the segment. Keys are the values, and values are
        descriptions, e.g. for 'p', 1: 'HB work'. Descriptions don't tend to
        get used in DVectors so can be as verbose as desired for clarity.
    exclusions: list[Exclusion]
        Define incompatibilities between segments. See Exclusion class
    """

    name: str
    values: dict[int, str]
    exclusions: list[Exclusion] = pydantic.Field(default_factory=list)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # pylint: disable=too-few-public-methods

    @property
    def _exclusion_segs(self):
        return [seg.seg_name for seg in self.exclusions]

    def _drop_indices(self, other_seg: str):
        if other_seg not in self._exclusion_segs:
            return None
        ind_tuples = []
        for excl in self.exclusions:
            if excl.seg_name == other_seg:
                for other in excl.other_vals:
                    ind_tuples.append((excl.own_val, other))
        drop_ind = pd.MultiIndex.from_tuples(ind_tuples)
        return drop_ind


class SegmentsSuper(enum.Enum):
    """
    Getter for predefined segments.

    This should be where segments forming segmentations come from. In most
    cases if a segment is not defined here it should be added, rather than
    defined as a custom segment in a Segmentation.
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
        """Return values from class."""
        return [e.value for e in cls]

    def get_segment(self, subset: Optional[list[int]] = None):
        """
        Get a segment.

        Parameters
        ----------
        subset: Define a subset of the segment being got. The integers in subset
        must appear in the asked for segment.
        """
        seg = None
        match self:
            case SegmentsSuper.PURPOSE:
                seg = Segment(
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
                seg = Segment(
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
                seg = Segment(
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
                seg = Segment(
                    name=self.value,
                    values={1: "Child", 2: "Male", 3: "Female"},
                    exclusions=[
                        Exclusion(
                            seg_name=SegmentsSuper.SOC.value,
                            own_val=1,
                            other_vals={1, 2, 3},
                        )
                    ],
                )
            case SegmentsSuper.SOC:
                seg = Segment(
                    name=self.value,
                    values={
                        1: "High Skilled",
                        2: "High Skilled",
                        3: "High Skilled",
                        4: "Skilled",
                    },
                )
            case SegmentsSuper.CA:
                seg = Segment(name=self.value, values={1: "dummy", 2: "dummy"})
            case SegmentsSuper.NS:
                seg = Segment(
                    name=self.value,
                    values={1: "dummy", 2: "dummy", 3: "dummy", 4: "dummy", 5: "dummy"},
                )

        if subset:
            if seg is not None:
                seg.values = {i: j for i, j in seg.values.items() if i in subset}
        return seg


# # # FUNCTIONS # # #
