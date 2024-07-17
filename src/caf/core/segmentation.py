# -*- coding: utf-8 -*-
"""
Module for handling segmentation objects.

This imports the Segment class from caf.core.segments, and the SegmentsSuper
enumeration from caf.core.segments. Both are used for building segmentations.
"""
from __future__ import annotations

# Built-Ins
import warnings
from typing import Union, Literal, Optional
from os import PathLike
from pathlib import Path
from copy import deepcopy

# Third Party
import pandas as pd
import pydantic
import h5py
from caf.toolkit import BaseConfig


# Local Imports
# pylint: disable=import-error,wrong-import-position
# Local imports here
from caf.core.segments import Segment, SegmentsSuper

# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #


# # # CLASSES # # #
class SegmentationWarning(Warning):
    """Warn about segmentation objects."""


class SegmentationError(Exception):
    """Error for segmentation objects."""

    pass


class SegmentationInput(BaseConfig):
    """
    Input class for segmentation objects.

    Parameters
    ----------
    enum_segments: list[SegmentsSuper]
        Provide as strings matching enumerations in SegmentsSuper. In 99% of
        cases segments should be provided in this form. If a segment doesn't
        exist in the enumeration it should usually be added.
    subsets: dict[str, list[int]]
        Define any segments you want subsets of. Keys should match values in
        'enum_segments', values should be lists defining values from the
        segment which should be included.
    custom_segments: list[Segment]
        User defined segments which don't appear in SegmentsSuper. As stated,
        in almost all cases a missing segment should be added to that class,
        and this option is a contingency.
    naming_order: list[str]
        The naming order of the segments. This primarily affects the index
        order of the multi-index formed from the segmentation.
    """

    enum_segments: list[SegmentsSuper]
    naming_order: list[str]
    custom_segments: list[Segment] = pydantic.Field(default_factory=list)
    subsets: dict[str, list[int]] = pydantic.Field(default_factory=dict)

    # @pydantic.field_validator("enum_segments")
    # @classmethod
    # def make_enum(cls, v):
    #     out = []
    #     for val in v:
    #         if isinstance(val, SegmentsSuper):
    #             out.append(val)
    #         else:
    #             out.append(SegmentsSuper(val))
    #     return out

    @pydantic.model_validator(mode="before")
    @classmethod
    def no_copied_names(cls, values):
        """Validate the custom_segments do not clash with existing segments."""
        if "custom_segments" not in values:
            return values
        v = values["custom_segments"]
        for seg in v:
            if seg.name in SegmentsSuper.values():
                raise ValueError(
                    "There is already a segment defined with name "
                    f"{seg.name}. Segment names must be unique "
                    "even if the existing segment isn't in this "
                    "segmentation. This error is raised on the "
                    "first occurrence so it is possible there is "
                    "more than one clash. 'caf.core.SegmentsSuper.values' "
                    "will list all existing segment names."
                )
        return values

    @pydantic.model_validator(mode="after")
    @classmethod
    def names_match_segments(cls, values):
        """Validate that naming order names match segment names."""
        v = values.naming_order
        seg_names = [i.value for i in values.enum_segments]
        if len(values.custom_segments) > 0:
            seg_names += [i.name for i in values.custom_segments]

        if set(seg_names) != set(v):
            raise ValueError("Names provided for naming_order do not match names in segments")

        return values

    @pydantic.model_validator(mode="after")
    @classmethod
    def enums(cls, values):
        """Validate the subsets match segments."""
        if len(values.subsets) == 0:
            return values
        for seg in values.subsets.keys():
            if seg not in [i.value for i in values.enum_segments]:
                raise ValueError(
                    f"{seg} is not a valid segment  " ", and so can't be a subset value."
                )
        return values


class Segmentation:
    """
    Segmentation class for handling segmentation objects.

    Parameters
    ----------
        config: SegmentationInput
        Instance of SegmentationInput. See that class for details.
    """

    # This currently isn't used and doesn't mean anything. In few places code
    # relating to time periods or time formats is included from normits_core but
    # never used.
    _time_period_segment_name = "tp3"

    def __init__(self, config: SegmentationInput):
        self.input = config
        # unpack enum segments, applying subsets if necessary
        if config.subsets is None:
            enum_segments = [
                SegmentsSuper(string).get_segment() for string in config.enum_segments
            ]

        else:
            enum_segments = []
            for seg in config.enum_segments:
                segment = SegmentsSuper(seg).get_segment(subset=config.subsets.get(seg.value))
                enum_segments.append(segment)

        self.segments = config.custom_segments + enum_segments
        self.naming_order = config.naming_order

    @property
    def seg_dict(self):
        """Access segments in dict form."""
        return {seg.name: seg for seg in self.segments}

    def __iter__(self):
        return self.seg_dict.__iter__()

    @property
    def names(self):
        """Return the names of all segments."""
        return [seg.name for seg in self.segments]

    @property
    def seg_descriptions(self):
        """Return a list of segment descriptions."""
        return [seg.values.values() for seg in self.segments]

    @property
    def seg_vals(self):
        """Return all segmentation values."""
        return [seg.values.keys() for seg in self.segments]

    def ind(self):
        """
        Return a pandas MultiIndex of the segmentation.

        This is by default just a product of all segments given, taking
        exclusions into account if any exist between segments.
        """
        index = pd.MultiIndex.from_product(self.seg_vals, names=self.names)
        df = pd.DataFrame(index=index).reorder_levels(self.naming_order).reset_index()
        drop_iterator = self.naming_order.copy()
        # TODO hard coded here for now as a quick fix, needs thought
        # hh_dropper = pd.MultiIndex.from_tuples([(2, 1, 1, 4),
        #     (2, 1, 2, 4),
        #     (2, 1, 3, 1),
        #     (2, 1, 3, 4),
        #     (2, 2, 1, 4),
        #     (2, 2, 2, 4),
        #     (2, 2, 3, 1),
        #     (2, 2, 3, 4),
        #     (3, 1, 1, 4),
        #     (3, 1, 2, 4),
        #     (3, 1, 3, 1),
        #     (3, 1, 3, 4),
        #     (3, 2, 1, 4),
        #     (3, 2, 2, 4),
        #     (3, 2, 3, 1),
        #     (3, 2, 3, 4),
        #     (4, 1, 4, 1),
        #     (4, 1, 4, 2),
        #     (4, 1, 4, 3),
        #     (4, 1, 4, 4),
        #     (4, 2, 4, 1),
        #     (4, 2, 4, 2),
        #     (4, 2, 4, 3),
        #     (4, 2, 4, 4)],
        #    names=['aws', 'hh_type', 'soc', 'ns_sec'])
        for own_seg in self.segments:
            for other_seg in drop_iterator:
                if other_seg == own_seg.name:
                    continue
                # pylint: disable=protected-access
                if other_seg in own_seg._exclusion_segs:
                    dropper = own_seg._drop_indices(other_seg)
                    df = df.set_index([own_seg.name, other_seg])
                    mask = ~df.index.isin(dropper)
                    df = df[mask].reset_index()
                # pylint: enable=protected-access
        # if set(hh_dropper.names).issubset(self.names):
        #     df = df.set_index(hh_dropper.names).drop(hh_dropper).reset_index()
        return df.set_index(self.naming_order).sort_index().index

    def has_time_period_segments(self) -> bool:
        """Check whether this segmentation has time period segmentation.

        Returns
        -------
        has_time_period_segments:
            True if there is a time_period segment in this segmentation,
            False otherwise
        """
        return self._time_period_segment_name in self.naming_order

    # pylint: disable=too-many-branches
    @classmethod
    def validate_segmentation(
        cls,
        source: Union[Path, pd.DataFrame],
        segmentation: Segmentation,
        escalate_warning: bool = False,
        cut_read: bool = False
    ) -> Segmentation:
        """
        Validate a segmentation from either a path to a csv, or a dataframe.

        This could either be purely a segmentation, or data with a segmentation
        index.

        Parameters
        ----------
        source : Path | pd.DataFrame
            Either a path to a csv containing a segmentation or a dataframe
            containing a segmentation. If source is a dataframe the
            segmentation should not form the index.
        segmentation : Segmentation
            The segmentation you expect 'source' to match.

        Returns
        -------
        Segmentation class
        """
        if escalate_warning:
            warnings.filterwarnings("error", category=SegmentationWarning)
        if isinstance(source, Path):
            df = pd.read_csv(source).sort_index()
        else:
            df = source.sort_index()

        naming_order = segmentation.naming_order
        conf = segmentation.input.copy()
        if df.index.names == naming_order:
            read_index = df.index
        else:
            # Try to build index from df columns
            try:
                df.set_index(naming_order, inplace=True)
            # Assume the index is already correct but reorder to naming_order
            except KeyError:
                df = df.reorder_levels(naming_order)
            read_index = df.index
        # Index to validate against
        built_index = segmentation.ind()
        # I think an error would already be raised at this point
        if built_index.names != read_index.names:
            raise ValueError(
                "The read in segmentation does not match the given parameters. "
                "The segment names are not correct."
            )

        # Perfect match, return segmentation with no more checks
        if read_index.equals(built_index):
            return segmentation, False
        if not cut_read:
            if len(read_index) > len(built_index):
                raise IndexError(
                    "The segmentation of the read in dvector data "
                    "does not match the expected segmentation. This "
                    "is likely due to unconsidered exclusions."
                    f"{read_index.difference(built_index)} in read in index but not "
                    f"in expected index."
                )
        for name in built_index.names:
            built_level = set(built_index.get_level_values(name))
            read_level = set(read_index.get_level_values(name))
            # This level matches, check the next one
            if read_level == built_level:
                continue
            # The input segmentation should have had subsets defined. warn user but allow
            if read_level.issubset(built_level):
                warnings.warn(
                    f"Read in level {name} is a subset of the segment. If this was not"
                    f" expected check the input segmentation.",
                    SegmentationWarning,
                )
                # Define the read subset in the generated config
                if conf.subsets is not None:
                    conf.subsets.update({name: list(read_level)})
                else:
                    conf.subsets = {name: list(read_level)}
            # Not a subset so doesn't match completely
            else:
                raise ValueError(
                    f"The segment for {name} does not match the inbuilt definition."
                    f"Check for mistakes in the read in segmentation, or redefine the"
                    f"segment with a different name."
                )

        built_segmentation = cls(conf)
        built_index = built_segmentation.ind()

        if read_index.equals(built_index):
            return built_segmentation, False
        # Still doesn't match, this is probably an exclusion error. User should check that
        # proper exclusions are defined in SegmentsSuper.
        if built_index.equals(built_index.intersection(read_index)):
            if cut_read:
                return built_segmentation, False
            raise SegmentationError("Read data contains rows not in the generated segmentation. "
                                    "If you want this data to simply be cut to match, set 'cut_read=True'")
        if read_index.equals(built_index.intersection(read_index)):
            warnings.warn("Combinations missing from the read in data. This may mean an exclusion should "
                          "be defined but isn't. The data will be expanded to the expected segmenation, and "
                          "infilled with zeroes.")
            return built_segmentation, True
        raise ValueError(
            "The read in segmentation does not match the given parameters. The segment names"
            " are correct, but segment values don't match. This could be due to an incompatibility"
            " between segments which isn't reflected in the loaded in the segmentation, or it could be"
            " an out of date in built segmentation in the caf.core package. The first place to "
            "look is the SegmentsSuper class."
        )
    # pylint: enable=too-many-branches

    def translate_segment(self, from_seg: Segment, to_seg):
        to_seg = from_seg.translate_segment(to_seg)
        new_conf = self.input.copy()
        if SegmentsSuper(from_seg.name) in new_conf.enum_segments:
            new_conf.enum_segments.remove(SegmentsSuper(from_seg.name))
        else:
            new_conf.custom_segments.remove(from_seg)
        new_conf.naming_order[new_conf.naming_order.index(from_seg.name)] = to_seg.name
        try:
            new_conf.enum_segments.append(SegmentsSuper(to_seg.name))
        except ValueError:
            new_conf.custom_segments.append(to_seg)
        return Segmentation(new_conf)




    def reinit(self):
        """Regenerate Segmentation from its input."""
        return Segmentation(self.input)

    def save(self, out_path: PathLike, mode: Literal["hdf", "yaml"] = "hdf"):
        """
        Save a segmentation to either a yaml file or an hdf file if part of a DVector.

        Parameters
        ----------
        out_path: PathLike
            Path to where the data should be saved. The file extension must
            match 'mode'
        mode: Literal["hdf", "yaml"]
            Currently only can be 'hdf' or 'yaml'. How to save the file.
        """
        if mode == "hdf":
            with h5py.File(out_path, "a") as h_file:
                h_file.create_dataset(
                    "segmentation", data=self.input.to_yaml().encode("utf-8")
                )

        elif mode == "yaml":
            self.input.save_yaml(Path(out_path))

        else:
            raise ValueError(f"Mode must be either 'hdf' or 'yaml', not {mode}")

    @classmethod
    def load(cls, in_path: PathLike, mode: Literal["hdf", "yaml"] = "hdf") -> Segmentation:
        """
        Load the segmentation from a file, either an hdf or csv file.

        Parameters
        ----------
        in_path: PathLike
            Path to the file. File extension must match 'mode'
        mode: Literal["hdf", "yaml"], default "hdf"
            Mode to load in, either 'hdf' or 'yaml'

        Returns
        -------
        Segmentation class
        """
        # pylint: disable=no-member
        if mode == "hdf":
            with h5py.File(in_path, "r") as h_file:
                yam_load = h_file["segmentation"][()].decode("utf-8")
                config = SegmentationInput.from_yaml(yam_load)
        # pylint: enable=no-member

        elif mode == "yaml":
            config = SegmentationInput.load_yaml(Path(in_path))

        else:
            raise ValueError(f"Mode must be either 'hdf' or 'yaml', not {mode}")

        return cls(config)

    def __copy__(self):
        """Return a copy of this class."""
        return self.copy()

    def __eq__(self, other) -> bool:
        """Override the default implementation."""
        if not isinstance(other, Segmentation):
            return False

        if self.naming_order != other.naming_order:
            return False

        if set(self.names) != set(other.names):
            return False

        return True

    @staticmethod
    def ordered_set(list_1: list, list_2: list) -> list:
        """Take in two lists and combine them, removing duplicates but preserving order."""
        combined_list = list_1 + list_2
        unique_list = []
        for item in combined_list:
            if item not in unique_list:
                unique_list.append(item)
        return unique_list

    def __len__(self):
        return len(self.ind())

    def __add__(self, other):
        """
        Combine two segmentations without duplicates.

        Order of naming_order in resulting segmentation will have self before
        other. This name may be misleading as this is the method used for most
        of the dunder methods in DVector for combining resulting segmentations.
        """
        enum_in = set(self.input.enum_segments + other.input.enum_segments)
        cust_in = self.input.custom_segments
        for seg in other.input.custom_segments:
            if seg.name not in [i.name for i in cust_in]:
                cust_in.append(seg)
        subsets = self.input.subsets.copy()
        subsets.update(other.input.subsets)
        naming_order = self.ordered_set(self.naming_order, other.naming_order)
        config = SegmentationInput(
            enum_segments=enum_in,
            subsets=subsets,
            custom_segments=cust_in,
            naming_order=naming_order,
        )
        return Segmentation(config)

    def overlap(self, other: Segmentation | list[str]):
        """Check the overlap in segments between two segmentations."""
        if isinstance(other, Segmentation):
            return set(self.names).intersection(other.names)
        return set(self.names).intersection(other)

    def is_subset(self, other: Segmentation):
        if self.overlap(other) == set(self.names):
            return True
        else:
            return False

    def __sub__(self, other):
        return [i for i in self.naming_order if i not in other.naming_order]

    def __ne__(self, other) -> bool:
        """Override the default implementation."""
        return not self.__eq__(other)

    def copy(self):
        """Copy an instance of this class."""
        return Segmentation(config=deepcopy(self.input))

    def aggregate(self, new_segs: list[str]):
        """
        Aggregate segmentation to a subset of the segmentation.

        This method isn't exactly an aggregation, it just removes segments.
        It is called aggregate as currently it is the segmentation component
        of the aggregate method in DVector.

        Parameters
        ----------
        new_segs: The new segmentation. All must be in the current segmentation.
        """
        custom = None
        subsets = None

        for i in new_segs:
            if i not in self.names:
                raise SegmentationError(
                    f"{i} is not in the current segmentation, "
                    "so cannot be aggregated to. This is the "
                    "first segment raising an error, and there "
                    "may be more."
                )

        if self.input.custom_segments is not None:
            custom = self.input.custom_segments.copy()
            for seg in self.input.custom_segments:
                if seg.name not in new_segs:
                    custom.remove(seg)

        enum_segs = self.input.enum_segments.copy()
        for enum_seg in self.input.enum_segments:
            if enum_seg.value not in new_segs:
                enum_segs.remove(enum_seg)

        if self.input.subsets is not None:
            subsets = dict()
            for key, val in self.input.subsets.items():
                if key in new_segs:
                    subsets.update({key: val})

        new_order = [i for i in self.naming_order if i in new_segs]

        conf = SegmentationInput(
            enum_segments=enum_segs,
            subsets=subsets,
            custom_segments=custom,
            naming_order=new_order,
        )
        return Segmentation(conf)

    def add_segment(
        self,
        new_seg: Segment | SegmentsSuper,
        subset: Optional[dict[str, list[int]]] = None,
        new_naming_order: Optional[list[str]] = None,
    ):
        """
        Add a new segment to a segmentation.

        Parameters
        ----------
        new_seg: Segment
            The new segment to be added. This will be checked and added as an
            enum_segment if it exists as such, and as a custom segment if not.
            This must be provided as a Segment type, and can't be a string to pass
            to the SegmentSuper enum class

        subset: Optional[dict[str, list[int]]] = None
            A subset definition if the new segmentation is a subset of an existing
            segmentation. This need only be provided for an enum_segment.

        new_naming_order: Optional[list[str]] = None
            The naming order of the resultant segmentation. If not provided,
            the new segment will be appended to the end.
        Returns
        -------
        Segmentation
        """
        out_segmentation = self.copy()
        custom = True
        if isinstance(new_seg, str):
            new_seg = SegmentsSuper(new_seg).get_segment()
        new_name = new_seg.name
        if new_name in SegmentsSuper.values():
            custom = False

        if new_name in self.names:
            raise ValueError(f"{new_name} already contained in segmentation.")
        if custom:
            out_segmentation.input.custom_segments.append(new_seg)
        else:
            out_segmentation.input.enum_segments.append(SegmentsSuper(new_name))
        if new_naming_order is not None:
            out_segmentation.input.naming_order = new_naming_order
        else:
            out_segmentation.input.naming_order.append(new_name)
        if subset is not None:
            out_segmentation.input.subsets.update(subset)
        return out_segmentation.reinit()

    def remove_segment(self, segment_name: str | Segment, inplace: bool = False):
        """
        Remove a segment from a segmentation.

        Parameters
        ----------
        segment_name: str
            The name of the segment to remove
        inplace: bool = False
            Whether to apply in place
        """
        if isinstance(segment_name, Segment):
            segment_name = segment_name.name
        if inplace:
            self.input.naming_order.remove(segment_name)
            if SegmentsSuper(segment_name) in self.input.enum_segments:
                self.input.enum_segments.remove(SegmentsSuper(segment_name))
            else:
                self.input.custom_segments.remove(segment_name)
            if segment_name in self.input.subsets.keys():
                del self.input.subsets[segment_name]
            self.reinit()
            return
        out_seg = deepcopy(self.input)
        out_seg.naming_order.remove(segment_name)
        if segment_name in SegmentsSuper.values():
            out_seg.enum_segments.remove(SegmentsSuper(segment_name))
        else:
            out_seg.custom_segments.remove(segment_name)
        if segment_name in out_seg.subsets.keys():
            del out_seg.subsets[segment_name]
        return Segmentation(out_seg)

    def update_subsets(self, extension: dict[str, int | list[int]], remove=False):
        """
        Add to subsets dict.

        Parameters
        ----------
        extension: dict[str, int]
            Values to add to the subsets dict. This will still work if subsets
            is currently empty.
        """
        out_seg = self.input.copy()
        for key, val in extension.items():

            if key not in self.names:
                raise ValueError(
                    f"{key} not in current segmentation, so can't " "be added to subsets"
                )
            if isinstance(val, int):
                val = [val]
            if key in out_seg.subsets:
                if remove:
                    out_seg.subsets[key] = list(set(out_seg.subsets[key]) - set(val))
                else:
                    out_seg.subsets[key] = list(set(out_seg.subsets[key] + val))
            else:
                if remove:
                    out_seg.subsets[key] = list(set(self.seg_dict[key]) - set(val))

                out_seg.subsets.update({key: val})
        return Segmentation(out_seg)


# # # FUNCTIONS # # #
