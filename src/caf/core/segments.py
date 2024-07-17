# -*- coding: utf-8 -*-
"""Module defining Segments class and enumeration."""
import enum
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import os

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
    other_name: str
        Name of the other segment this exclusion applies to
    exclusions: int
        The value for self segmentation which has exclusions in other
    """
    other_name: str
    exclusions: dict[int, set[int]]

    def build_index(self):
        """Return an index formed of the exclusions."""
        frame = pd.DataFrame.from_dict(self.exclusions, orient='index').stack().reset_index().drop(columns='level_1')
        return pd.MultiIndex.from_frame(frame, names=['dummy', self.other_name])


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
        return [seg.other_name for seg in self.exclusions]

    def _drop_indices(self, other_seg: str):
        if other_seg not in self._exclusion_segs:
            return None
        ind_tuples = []
        for excl in self.exclusions:
            if excl.other_name == other_seg:
                return excl.build_index()

    def __len__(self):
        return len(self.values)

    def translate_segment(self, new_seg):
        if isinstance(new_seg, Segment):
            return new_seg
        if isinstance(new_seg, str):
            return SegmentsSuper(new_seg).get_segment()
        raise TypeError("translate_method expects either an instance of the Segment "
                        "class, or a str contained within the SegmentsSuper enum class. "
                        f"{type(new_seg)} cannot be handled.")

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
    ADULT_NSSEC = "adult_nssec"

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
        segs_dir = Path(__file__).parent / "segments"
        try:
            seg = Segment.load_yaml(segs_dir / f"{self.value}.yml")
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find a segment saved at {segs_dir / self.value}.yml."
                                    f"This means an enum has been defined, but a segment has not, so this "
                                    f"is probably a placeholder.")

        if subset:
            if seg is not None:
                seg.values = {i: j for i, j in seg.values.items() if i in subset}
        return seg


class SegConverter(enum.Enum):

    AG_G = "ag_g"
    APOPEMP_AWS = "apopemp_aws"
    CARADULT_HHTYPE = "caradult_hhtype"

    def get_conversion(self):
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


if __name__ == '__main__':
    import os
    from pathlib import Path
    cwd = Path(os.getcwd())
    for seg in SegmentsSuper:
        try:
            segment = seg.get_segment()
        except AttributeError:
            continue

        if segment is not None:
            segment.save_yaml(cwd / "segments" / f"{seg.value}.yml")
# # # FUNCTIONS # # #
