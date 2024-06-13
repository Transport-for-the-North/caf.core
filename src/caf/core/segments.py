# -*- coding: utf-8 -*-
"""Module defining Segments class and enumeration."""
import enum
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import pydantic
from pydantic import ConfigDict
from caf.toolkit import BaseConfig


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
    # pylint: disable=not-an-iterable
    # Pylint doesn't seem to understand pydantic.Field
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

    def __len__(self):
        return len(self.values)

    # pylint: enable=not-an-iterable


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
    TFN_TT = "tfn_tt"
    USERCLASS = "uc"
    ACCOMODATION_TYPE_H = "accom_h"
    ACCOMODATION_TYPE_HR = "accom_hr"
    ADULTS = "adults"
    CHILDREN = "children"
    CAR_AVAILABILITY = "car_availability"
    AGE = "age_9"
    AGE_11 = "age_11"
    AGE_AGG = "age_5"
    GENDER_3 = "gender_3"
    ECONOMIC_STATUS = "economic_status"
    POP_EMP = "pop_emp"
    POP_ECON = "pop_econ"
    NS_SEC = "ns_sec"
    AWS = "aws"
    HH_TYPE = "hh_type"

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

            case SegmentsSuper.ACCOMODATION_TYPE_H:
                seg = Segment(
                    name=self.value,
                    values={
                        1: "Whole house or bungalow: Detached",
                        2: "Whole house or bungalow: Semi-detached",
                        3: "Whole house or bungalow: Terraced",
                        4: "Flat, maisonette or apartment",
                        5: "A caravan or other mobile or temporary structure",
                    },
                )

            case SegmentsSuper.ACCOMODATION_TYPE_HR:
                seg = Segment(
                    name=self.value,
                    values={
                        1: "Whole house or bungalow: Detached",
                        2: "Whole house or bungalow: Semi-detached",
                        3: "Whole house or bungalow: Terraced",
                        4: "Flat, maisonette or apartment",
                    },
                )

            case SegmentsSuper.ADULTS:
                seg = Segment(
                    name=self.value,
                    values={
                        1: "No adults or 1 adult in household",
                        2: "2 adults in household",
                        3: "3 or more adults in household",
                    },
                )

            case SegmentsSuper.CHILDREN:
                seg = Segment(
                    name=self.value,
                    values={
                        1: "Household with no children or all children non-dependent",
                        2: "Household with one or more dependent children",
                    },
                    exclusions=[
                        Exclusion(
                            seg_name=SegmentsSuper.AGE_11.value,
                            own_val=1,
                            other_vals={1, 2, 3},
                        )
                    ],
                )

            case SegmentsSuper.CAR_AVAILABILITY:
                seg = Segment(
                    name=self.value,
                    values={
                        1: "No cars or vans in household",
                        2: "1 car or van in household",
                        3: "2 or more cars or vans in household",
                    },
                )

            case SegmentsSuper.AGE:
                seg = Segment(
                    name=self.value,
                    values={
                        1: "0 to 4 years",
                        2: "5 to 9 years",
                        3: "10 to 15 years",
                        4: "16 to 19 years",
                        5: "20 to 34 years",
                        6: "35 to 49 years",
                        7: "50 to 64 years",
                        8: "65 to 74 years",
                        9: "75+ years",
                    },
                )

            case SegmentsSuper.AGE_11:
                seg = Segment(
                    name=self.value,
                    values={
                        1: "0 to 4 years",
                        2: "5 to 9 years",
                        3: "10 to 15 years",
                        4: "16 to 19 years",
                        5: "20 to 24 years",
                        6: "25 to 34 years",
                        7: "35 to 49 years",
                        8: "50 to 64 years",
                        9: "65 to 74 years",
                        10: "75 to 84 years",
                        11: "85 + years",
                    },
                    exclusions=[
                        Exclusion(
                            seg_name=SegmentsSuper.ECONOMIC_STATUS.value,
                            own_val=1,
                            other_vals={1, 2, 3, 4, 5, 6},
                        ),
                        Exclusion(
                            seg_name=SegmentsSuper.ECONOMIC_STATUS.value,
                            own_val=2,
                            other_vals={1, 2, 3, 4, 5, 6},
                        ),
                        Exclusion(
                            seg_name=SegmentsSuper.ECONOMIC_STATUS.value,
                            own_val=3,
                            other_vals={1, 2, 3, 4, 5, 6},
                        ),
                        Exclusion(
                            seg_name=SegmentsSuper.SOC.value,
                            own_val=1,
                            other_vals={1, 2, 3},
                        ),
                        Exclusion(
                            seg_name=SegmentsSuper.SOC.value,
                            own_val=2,
                            other_vals={1, 2, 3},
                        ),
                        Exclusion(
                            seg_name=SegmentsSuper.SOC.value,
                            own_val=3,
                            other_vals={1, 2, 3},
                        ),
                    ],
                )

            case SegmentsSuper.AGE_AGG:
                seg = Segment(
                    name=self.value,
                    values={
                        1: "aged 15 years and under",
                        2: "aged 16 to 24 years",
                        3: "aged 25 to 34 years",
                        4: "aged 35 to 49 years",
                        5: "aged 50 years and over",
                    },
                )

            case SegmentsSuper.GENDER:
                seg = Segment(
                    name=self.value,
                    values={1: "male", 2: "female"},
                )

            case SegmentsSuper.NS_SEC:
                seg = Segment(
                    name=self.value,
                    values={
                        1: "HRP managerial / professional",
                        2: "HRP intermediate / technical",
                        3: "HRP semi-routine / routine",
                        4: "HRP never worked / long-term unemployed",
                        5: "HRP full-time student",
                    },
                )

            case SegmentsSuper.SOC:
                seg = Segment(
                    name=self.value,
                    values={1: "SOC1", 2: "SOC2", 3: "SOC3", 4: "SOC4"},
                    exclusions=[
                        Exclusion(
                            seg_name=SegmentsSuper.ECONOMIC_STATUS.value,
                            own_val=1,
                            other_vals={2, 4, 5, 6},
                        ),
                        Exclusion(
                            seg_name=SegmentsSuper.ECONOMIC_STATUS.value,
                            own_val=2,
                            other_vals={2, 4, 5, 6},
                        ),
                        Exclusion(
                            seg_name=SegmentsSuper.ECONOMIC_STATUS.value,
                            own_val=3,
                            other_vals={2, 4, 5, 6},
                        ),
                        Exclusion(
                            seg_name=SegmentsSuper.ECONOMIC_STATUS.value,
                            own_val=4,
                            other_vals={1, 3},
                        ),
                    ],
                )

            case SegmentsSuper.POP_EMP:
                seg = Segment(
                    name=self.value,
                    values={
                        1: "full_time",
                        2: "part_time",
                        3: "unemployed",
                        4: "students",
                        5: "non-working_age",
                    },
                )

            case SegmentsSuper.POP_ECON:
                seg = Segment(
                    name=self.value,
                    values={
                        1: "Economically active employees",
                        2: "Economically active unemployed",
                        3: "Economically inactive",
                        4: "Students",
                    },
                )

            case SegmentsSuper.ECONOMIC_STATUS:
                seg = Segment(
                    name=self.value,
                    values={
                        1: "Economically active employment",
                        2: "Economically active unemployed",
                        3: "Economically active student employment",
                        4: "Economically active student unemployed",
                        5: "Economically inactive student",
                        6: "Economically inactive",
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

            case SegmentsSuper.GENDER_3:
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

            case SegmentsSuper.CA:
                seg = Segment(name=self.value, values={1: "dummy", 2: "dummy"})

            case SegmentsSuper.AWS:
                seg = Segment(
                    name=self.value,
                    values={1: "Child", 2: "FTE", 3: "PTE", 4: "Student", 5: "NEET", 6: "75+"},
                )

            case SegmentsSuper.HH_TYPE:
                seg = Segment(
                    name=self.value,
                    values={
                        1: "1 adult with 0 car",
                        2: "1 adult with 1+ cars",
                        3: "2 adults with 0 car",
                        4: "2 adults with 1 car",
                        5: "2 adults with 2+ cars",
                        6: "3+ adults with 0 car",
                        7: "3+ adults with 1 car",
                        8: "3+ adults with 2+ cars",
                    },
                )

        if subset:
            if seg is not None:
                seg.values = {i: j for i, j in seg.values.items() if i in subset}
        return seg


class SegConverter(enum.Enum):

    AG_G = "ag_g"
    APOPEMP_AWS = "apopemp_aws"
    CARADULT_HHTYPE = "caradult_hhtype"

    def get_conversion(self):
        con = None
        match self:
            case SegConverter.AG_G:
                from_ind = pd.MultiIndex.from_tuples(
                    [
                        (1, 1),
                        (2, 1),
                        (3, 1),
                        (1, 2),
                        (2, 2),
                        (3, 2),
                        (4, 1),
                        (5, 1),
                        (6, 1),
                        (7, 1),
                        (8, 1),
                        (9, 1),
                        (4, 2),
                        (5, 2),
                        (6, 2),
                        (7, 2),
                        (8, 2),
                        (9, 2),
                    ],
                    names=["age_9", "g"],
                )
                to_vals = [
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                ]
                return pd.DataFrame(index=from_ind, data={"gender_3": to_vals})
        match self:
            case SegConverter.APOPEMP_AWS:
                from_ind = pd.MultiIndex.from_tuples(
                    [
                        (9, 1),
                        (9, 2),
                        (9, 3),
                        (9, 4),
                        (9, 5),
                        (1, 1),
                        (1, 2),
                        (1, 3),
                        (1, 4),
                        (1, 5),
                        (2, 1),
                        (2, 2),
                        (2, 3),
                        (2, 4),
                        (2, 5),
                        (3, 1),
                        (3, 2),
                        (3, 3),
                        (3, 4),
                        (3, 5),
                        (4, 1),
                        (5, 1),
                        (6, 1),
                        (7, 1),
                        (8, 1),
                        (4, 2),
                        (5, 2),
                        (6, 2),
                        (7, 2),
                        (8, 2),
                        (4, 3),
                        (5, 3),
                        (6, 3),
                        (7, 3),
                        (8, 3),
                        (4, 4),
                        (5, 4),
                        (6, 4),
                        (7, 4),
                        (8, 4),
                    ],
                    names=["age_9", "pop_emp"],
                )

                to_vals = [
                    6,
                    6,
                    6,
                    6,
                    6,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    2,
                    2,
                    2,
                    2,
                    2,
                    3,
                    3,
                    3,
                    3,
                    3,
                    5,
                    5,
                    5,
                    5,
                    5,
                    4,
                    4,
                    4,
                    4,
                    4,
                ]

                return pd.DataFrame(index=from_ind, data={"aws": to_vals})

        match self:
            case SegConverter.CARADULT_HHTYPE:
                from_ind = pd.MultiIndex.from_tuples(
                    [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)],
                    names=["adults", "car_availability"],
                )
                to_vals = [1, 2, 2, 3, 4, 5, 6, 7, 8]

                return pd.DataFrame(index=from_ind, data={"hh_type": to_vals})


# # # FUNCTIONS # # #
