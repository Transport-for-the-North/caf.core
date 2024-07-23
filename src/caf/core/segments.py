# -*- coding: utf-8 -*-
"""Module defining Segments class and enumeration."""
import enum
from pydantic import dataclasses
from typing import Optional
from pathlib import Path
import os

import pandas as pd
import pydantic
from pydantic import ConfigDict
from caf.toolkit import BaseConfig


# # # CLASSES # # #
@dataclasses.dataclass
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
        frame = pd.DataFrame.from_dict(self.exclusions, orient='index').stack().reset_index()
        frame[0] = frame[0].astype(int)
        frame.drop('level_1', axis=1, inplace=True)
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
        Define incompatibilities between segments. See Correspondence class
    lookups: list[Exclusion]
        Define lookups between segments, essentially the reverse of exclusions.
        More efficient for segments with mappings, e.g. different defintions of age.
    """

    name: str
    values: dict[int, str]
    exclusions: list[Exclusion] = pydantic.Field(default_factory=list)
    lookups: list[Exclusion] = pydantic.Field(default_factory=list)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # pylint: disable=too-few-public-methods
    # pylint: disable=not-an-iterable
    # Pylint doesn't seem to understand pydantic.Field
    @property
    def _exclusion_segs(self):
        return [seg.other_name for seg in self.exclusions]

    @property
    def _lookup_segs(self):
        return [seg.other_name for seg in self.lookups]

    def _drop_indices(self, other_seg: str):
        if other_seg not in self._exclusion_segs:
            return None
        for excl in self.exclusions:
            if excl.other_name == other_seg:
                return excl.build_index()

    def _lookup_indices(self, other_seg: str):
        if other_seg not in self._lookup_segs:
            return None
        for lookup in self.lookups:
            if lookup.other_name == other_seg:
                return lookup.build_index()

    @property
    def int_values(self):
        return list(self.values.keys())

    def __len__(self):
        return len(self.values)

    def translate_segment(self, new_seg, reverse=False, exclude=False):
        lookup_dir = Path(__file__).parent / "seg_translations"
        if not isinstance(new_seg, (str, Segment, SegmentsSuper)):
            raise TypeError("translate_method expects either an instance of the Segment "
                            "class, or a str contained within the SegmentsSuper enum class. "
                            f"{type(new_seg)} cannot be handled.")
        if isinstance(new_seg, str):
            new_seg = SegmentsSuper(new_seg).get_segment()
        if isinstance(new_seg, SegmentsSuper):
            new_seg = new_seg.get_segment()
        new_name = new_seg.name
        name_1 = self.name
        name_2 = new_name
        if reverse:
            name_1, name_2 = name_2, name_1
        lookup = pd.read_csv(lookup_dir / f"{name_1}_to_{name_2}.csv", index_col=0).squeeze()
        if exclude:
            full_product = pd.MultiIndex.from_product([self.values, new_seg.values], names=[self.name, new_seg.name])
            corr = lookup.to_frame().set_index(lookup.name, append=True)
            excl = pd.DataFrame(index=full_product.difference(corr.index)).reset_index(level=self.name).to_dict()
            excl = Exclusion(other_name=self.name, exclusions=excl)
            if new_seg.exclusions is None:
                new_seg.exclusions = [excl]
            else:
                new_seg.exclusions.append(excl)
        return new_seg, lookup

    def translate_exclusion(self, new_seg):
        segs_dir = Path(__file__).parent / "segments"
        new_seg, lookup = self.translate_segment(new_seg)
        update_seg = new_seg.copy()
        exclusions = []
        for exc in self.exclusions:
            from_exc = pd.DataFrame(index=exc.build_index()).reset_index().rename(columns={'dummy':self.name}).set_index(self.name)
            joined = from_exc.join(lookup).groupby([new_seg.name, exc.other_name]).sum().reset_index(level=exc.other_name)
            new_exc = {}
            for ind in joined.index.unique():
                new_exc[ind] = joined.loc[ind].squeeze().to_list()
            exclusions.append(Exclusion(other_name=exc.other_name, exclusions=new_exc))
        if len(update_seg.exclusions) == 0:
            update_seg.exclusions = exclusions
        else:
            update_seg.exclusions += exclusions
        update_seg.save_yaml(segs_dir / f"{new_seg.name}.yml")




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
    AGE_NTEM = "age_ntem"
    AGE_EDGE = "age_edge"
    GENDER_3 = "gender_3"
    ECONOMIC_STATUS = "economic_status"
    POP_EMP = "pop_emp"
    POP_ECON = "pop_econ"
    NS_SEC = "ns_sec"
    AWS = "aws"
    HH_TYPE = "hh_type"
    ADULT_NSSEC = "adult_nssec"
    SIC_1 = "sic_1_digit"
    SIC_2 = "sic_2_digit"
    SIC_4 = "sic_4_digit"

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
    NSSEC_ADULT = "nssec_adult"

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

        match self:
            case SegConverter.NSSEC_ADULT:
                from_ind = pd.MultiIndex.from_product([range(1,6), range(1,4)], names=['ns_sec', 'adults'])
                to_vals = range(1, 16)
                return pd.DataFrame(index=from_ind, data={"adult_nssec": to_vals})



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
