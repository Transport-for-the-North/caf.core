# -*- coding: utf-8 -*-
"""
Module containing the data structures used in the CAF package.

Currently this is only the DVector class, but this may be expanded in the future.
"""
from __future__ import annotations

import enum
import itertools
import logging
import math
import operator
import tempfile
import warnings
from collections.abc import Collection
from dataclasses import dataclass
from numbers import Number
from os import PathLike, listdir
from pathlib import Path
from typing import Callable, Literal, Optional, Union

import caf.toolkit as ctk
import numpy as np
import pandas as pd

# pylint: disable=no-name-in-module,import-error
from caf.base.segmentation import Segmentation, SegmentationError, SegmentationWarning
from caf.base.segments import SegConverter, Segment, SegmentsSuper
from caf.base.zoning import (
    BalancingZones,
    TranslationError,
    TranslationWeighting,
    ZoningSystem,
    normalise_column_name,
)

# pylint: enable=no-name-in-module,import-error

# # # CONSTANTS # # #
LOG = logging.getLogger(__name__)


# # # CLASSES # # #
# pylint: disable-all
@enum.unique
class TimeFormat(enum.Enum):
    """Class for time formats."""

    AVG_WEEK = "avg_week"
    AVG_DAY = "avg_day"
    AVG_HOUR = "avg_hour"

    @staticmethod
    def _valid_time_formats() -> list[str]:
        """Return a list of valid strings to pass for time_format."""
        return [x.value for x in TimeFormat]

    @staticmethod
    def get_time_periods() -> list[int]:
        """Get time periods."""
        return [1, 2, 3, 4, 5, 6]

    @staticmethod
    def conversion_order() -> list[TimeFormat]:
        """Return a conversion order."""
        return [TimeFormat.AVG_WEEK, TimeFormat.AVG_DAY, TimeFormat.AVG_HOUR]

    @staticmethod
    def _week_to_hour_factors() -> dict[int, float]:
        """Compound week to day and day to hour factors."""
        return ctk.toolbox.combine_dict_list(
            dict_list=[TimeFormat._week_to_day_factors(), TimeFormat._day_to_hour_factors()],
            operation=operator.mul,
        )

    @staticmethod
    def _hour_to_week_factors() -> dict[int, float]:
        """Compound hour to day and day to week factors."""
        return ctk.toolbox.combine_dict_list(
            dict_list=[TimeFormat._hour_to_day_factors(), TimeFormat._day_to_week_factors()],
            operation=operator.mul,
        )

    @staticmethod
    def _hour_to_day_factors() -> dict[int, float]:
        """Inverse of day to hour factors."""
        return {k: 1 / v for k, v in TimeFormat._day_to_hour_factors().items()}

    @staticmethod
    def _day_to_week_factors() -> dict[int, float]:
        """Inverse of week to day factors."""
        return {k: 1 / v for k, v in TimeFormat._week_to_day_factors().items()}

    @staticmethod
    def _week_to_day_factors() -> dict[int, float]:
        return {
            1: 0.2,
            2: 0.2,
            3: 0.2,
            4: 0.2,
            5: 1,
            6: 1,
        }

    @staticmethod
    def _day_to_hour_factors() -> dict[int, float]:
        return {
            1: 1 / 3,
            2: 1 / 6,
            3: 1 / 3,
            4: 1 / 12,
            5: 1 / 24,
            6: 1 / 24,
        }

    @staticmethod
    def avg_hour_to_total_hour_factors() -> dict[int, float]:
        """Get a dictionary of conversion factors."""
        return TimeFormat._hour_to_day_factors()

    @staticmethod
    def total_hour_to_avg_hour_factors() -> dict[int, float]:
        """Get a dictionary of conversion factors."""
        return TimeFormat._day_to_hour_factors()

    @staticmethod
    def get(value: str) -> TimeFormat:
        """Get an instance of this with value.

        Parameters
        ----------
        value:
            The value of the enum to get the entire class for

        Returns
        -------
        time_format:
            The gotten time format

        Raises
        ------
        ValueError:
            If the given value cannot be found in the class enums.
        """
        # Check we've got a valid value
        value = value.strip().lower()
        if value not in TimeFormat._valid_time_formats():
            raise ValueError(
                "The given time_format is not valid.\n"
                "\tGot: %s\n"
                f"\tExpected one of: {(value, TimeFormat._valid_time_formats())}"
            )

        # Convert into a TimeFormat constant
        return_val = None
        for name, time_format_obj in TimeFormat.__members__.items():
            if name.lower() == value:
                return_val = time_format_obj
                break

        if return_val is None:
            raise ValueError(
                "We checked that the given time_format was valid, but it "
                "wasn't set when we tried to set it. This shouldn't be "
                "possible!"
            )
        return return_val

    def get_conversion_factors(
        self,
        to_time_format: TimeFormat,
    ) -> dict[int, float]:
        """
        Get the conversion factors for each time period.

        Get a dictionary of the values to multiply each time period by
        in order to convert between time formats

        Parameters
        ----------
        to_time_format:
            The time format you want to convert this time format to.
            Cannot be the same TimeFormat as this.

        Returns
        -------
        conversion_factors:
            A dictionary of conversion factors for each time period.
            Keys will the the time period, and values are the conversion
            factors.

        Raises
        ------
        ValueError:
            If any of the given values are invalid, or to_time_format
            is the same TimeFormat as self.
        """
        # Validate inputs
        if not isinstance(to_time_format, TimeFormat):
            raise ValueError(
                "Expected to_time_format to be a TimeFormat object. "
                f"Got: {type(to_time_format)}"
            )

        if to_time_format == self:
            raise ValueError("Cannot get the conversion factors when converting to self.")

        # Figure out which function to call
        if self == TimeFormat.AVG_WEEK and to_time_format == TimeFormat.AVG_DAY:
            factors_fn = self._week_to_day_factors
        elif self == TimeFormat.AVG_WEEK and to_time_format == TimeFormat.AVG_HOUR:
            factors_fn = self._week_to_hour_factors
        elif self == TimeFormat.AVG_DAY and to_time_format == TimeFormat.AVG_WEEK:
            factors_fn = self._day_to_week_factors
        elif self == TimeFormat.AVG_DAY and to_time_format == TimeFormat.AVG_HOUR:
            factors_fn = self._day_to_hour_factors
        elif self == TimeFormat.AVG_HOUR and to_time_format == TimeFormat.AVG_WEEK:
            factors_fn = self._hour_to_week_factors
        elif self == TimeFormat.AVG_HOUR and to_time_format == TimeFormat.AVG_DAY:
            factors_fn = self._hour_to_day_factors
        else:
            raise TypeError(
                "Cannot figure out the conversion factors to get from "
                f"time_format {self.value} to {to_time_format.value}"
            )

        return factors_fn()


# pylint enable-all
class DVector:
    """
    Class to store and manipulate data with segmentation and optionally zoning.

    The segmentation is stored as an attribute as well as forming the index of
    the data. Zoning, if present, is stored as an attribute as well as forming
    the columns of the data. Data is in the form of a dataframe and reads/writes
    to h5 along with all metadata.
    """

    def __init__(
        self,
        segmentation: Segmentation,
        import_data: pd.DataFrame,
        zoning_system: Optional[ZoningSystem] = None,
        time_format: Optional[Union[str, TimeFormat]] = None,
        val_col: Optional[str] = "val",
        low_memory: bool = False,
        cut_read: bool = False,
        _bypass_validation: bool = False,
    ) -> None:
        """
        Init method.

        Parameters
        ----------
        segmentation: Segmentation
            An instance of the segmentation class. This should usually be built
            from enumerated options in the SegmentsSuper class, but custom
            segments can be user defined if necesssary.
        import_data: pd.Dataframe
            The DVector data. This should usually be a dataframe or path to a
            dataframe, but there is also an option to read in and convert
            DVectors in the old format from NorMITs-demand.
        zoning_system: Optional[ZoningSystem] = None
            Instance of ZoningSystem. This must match import data. If this is
            given, import data must contain zone info in the column names, if
            this is not given import data must contain only 1 column.
        low_memory: bool = False
            Set to True for low_memory dunder_methods.
        _bypass_validation: bool = False
            Can be used to bypass validation and save some time.
            THIS IS ADDED AS AN OPTION TO USE IN IPF ONCE IT IS CERTAIN THE RETURN
            IS CORRECT. DO NOT MANUALLY SET TO TRUE.
        """
        if zoning_system is not None:
            if not isinstance(zoning_system, ZoningSystem):
                raise ValueError(
                    "Given zoning_system is not a caf.base.ZoningSystem object."
                    f"Got a {type(zoning_system)} object instead."
                )

        if not isinstance(segmentation, Segmentation):
            raise ValueError(
                "Given segmentation is not a caf.base.SegmentationLevel object."
                f"Got a {type(segmentation)} object instead."
            )

        self.low_memory = low_memory
        self._zoning_system = zoning_system
        self._segmentation = segmentation
        self._time_format = None
        if time_format is not None:
            self._time_format = self._validate_time_format(time_format)

        # Set defaults if args not set
        self._val_col = val_col
        self._cut_read = cut_read

        # Try to convert the given data into DVector format
        if _bypass_validation:
            self._data = import_data
        elif isinstance(import_data, (pd.DataFrame, pd.Series)):
            self._data, self._segmentation = self._dataframe_to_dvec(import_data)
        else:
            raise NotImplementedError(
                "Don't know how to deal with anything other than: pandas DF, or dict"
            )

    # SETTERS AND GETTERS
    @property
    def val_col(self):
        """Name of column containing DVector values, not relevant if DVector has a zoning system."""
        return self._val_col

    @property
    def zoning_system(self) -> ZoningSystem:
        """Get _zoning_system."""
        return self._zoning_system

    @property
    def segmentation(self) -> Segmentation:
        """Get _segmentation."""
        return self._segmentation

    @property
    def data(self) -> pd.DataFrame | pd.Series:
        """
        Get _data.

        data is a pandas DataFrame or pandas Series. It will have a multiindex
        comprising the segmentation, and columns comprising the zones, if there
        are zones.
        """
        return self._data

    @data.setter
    def data(self, value: pd.Series | pd.DataFrame):
        """Set _data."""
        if not isinstance(value, (pd.DataFrame, pd.Series)):
            raise TypeError(
                "data must be a pandas DataFrame or Series. Input " f"value is {value.type}."
            )
        if isinstance(value, pd.Series):
            value = value.to_frame()
        self._data, _ = self._dataframe_to_dvec(value)

    @property
    def time_format(self):
        """Get _time_format."""
        if self._time_format is None:
            return None
        return self._time_format.name

    @property
    def total(self):
        """Return the total of a DVector."""
        return self.data.values.sum()

    @staticmethod
    def _valid_time_formats() -> list[str]:
        """Return a list of valid strings to pass for time_format."""
        return [x.value for x in TimeFormat]

    def _validate_time_format(
        self,
        time_format: Union[str, TimeFormat],
    ) -> TimeFormat:
        """Validate the time format is a valid value.

        Parameters
        ----------
        time_format:
            The name of the time format name to validate

        Returns
        -------
        time_format:
            Returns a tidied up version of the passed in time_format.

        Raises
        ------
        ValueError:
            If the given time_format is not on of self._valid_time_formats
        """
        # Time period format only matters if it's in the segmentation
        if self.segmentation.has_time_period_segments() and time_format is None:
            raise ValueError(
                "The given segmentation level has time periods in its "
                "segmentation, but the format of this time period has "
                "not been defined.\n"
                f"\tTime periods segment name: {self.segmentation._time_period_segment_name}\n"
                f"\tValid time_format values: {self._valid_time_formats()}"
            )

        # If None or TimeFormat, that's fine
        if time_format is None or isinstance(time_format, TimeFormat):
            return time_format

        # Check we've got a valid value
        time_format = time_format.strip().lower()
        try:
            return TimeFormat(time_format)
        except ValueError as exc:
            raise ValueError(
                "The given time_format is not valid.\n"
                f"\tGot: {time_format}\n"
                f"\tExpected one of: {self._valid_time_formats()}"
            ) from exc

    def _dataframe_to_dvec(self, import_data: pd.DataFrame):
        """
        Take a dataframe and ensure it is in DVec data format.

        This requires the dataframe to be in wide format.
        """
        seg, expand_to_read = Segmentation.validate_segmentation(
            source=import_data, segmentation=self.segmentation, cut_read=self._cut_read
        )

        if len(seg.naming_order) > 1:
            sorted_data = import_data.reorder_levels(seg.naming_order).sort_index()
        else:
            sorted_data = import_data.sort_index()

        if expand_to_read:
            expander = pd.DataFrame(index=seg.ind(), data={"dummy": 1})
            if isinstance(sorted_data, pd.Series):
                sorted_data = sorted_data.to_frame()
            sorted_data = (
                sorted_data.join(expander, how="outer").fillna(0).drop("dummy", axis=1)
            )

        if self._cut_read:
            full_sum = sorted_data.values.sum()
            sorted_data = sorted_data.reindex(
                seg.ind(), axis="index", method=None
            ).sort_index()
            cut_sum = sorted_data.values.sum()
            warnings.warn(f"{full_sum - cut_sum} dropped on seg validation.")

        if self.zoning_system is None:
            if isinstance(sorted_data, pd.DataFrame):
                sorted_data = sorted_data.squeeze()
            sorted_data.name = self.val_col
            return sorted_data, seg

        # TODO: consider replacing with alternative checks that allow string IDs
        ### This chunk of code requires the zone names to be integers
        ### This has been commented out to allow LSOA (or other) zone codes to be used
        ### directly instead to avoid the added step of providing zone lookups with
        ### integer zone numbers for all zone systems
        # # Check columns are labelled with zone IDs
        # try:
        #     import_data.columns = import_data.columns.astype(int)
        # except ValueError as exc:
        #     raise TypeError(
        #         "DataFrame columns should be integers corresponding "
        #         f"to zone IDs not {import_data.columns.dtype}"
        #     ) from exc

        if set(sorted_data.columns) != set(self.zoning_system.zone_ids):
            missing = self.zoning_system.zone_ids[
                ~np.isin(self.zoning_system.zone_ids, sorted_data.columns)
            ]
            extra = sorted_data.columns.values[
                ~np.isin(sorted_data.columns.values, self.zoning_system.zone_ids)
            ]
            if len(extra) > 0:
                raise ValueError(
                    f"{len(missing)} zone IDs from zoning system {self.zoning_system.name}"
                    f" aren't found in the DVector data and {len(extra)} column names are"
                    " found which don't correspond to zone IDs.\nDVector DataFrame column"
                    " names should be the zone IDs (integers) for the given zone system."
                )
            if len(missing) > 0:
                warnings.warn(
                    f"{len(missing)} zone IDs from zoning system {self.zoning_system.name}"
                    f" aren't found in the DVector data. This may be by design"
                    f" e.g. you are using a subset of a zoning system."
                )

        return sorted_data, seg

    def save(self, out_path: PathLike):
        """
        Save the DVector.

        DVector will be saved to a hdf file containing the DVector. The preferred
        extension for DVectors is .dvec for clarity, but this isn't enforced
        anywhere

        Parameters
        ----------
        out_path: PathLike
            Path to the DVector, which should be an HDF file.

        Returns
        -------
        None
        """
        out_path = Path(out_path)
        if "." not in out_path.name:
            out_path.name = out_path.name

        self._data.to_hdf(out_path, key="data", mode="w", complevel=1)
        if self.zoning_system is not None:
            self.zoning_system.save(out_path, "hdf")
        self.segmentation.save(out_path, "hdf")

    @classmethod
    def load(cls, in_path: PathLike, cut_read: bool = False):
        """
        Load the DVector.

        Parameters
        ----------
        in_path: PathLike
            Path to where the DVector is saved. This should be a single hdf file.

        cut_read: bool = False
            If True, the read in data will be cut to match the expected segmentation.
        """
        in_path = Path(in_path)
        zoning = ZoningSystem.load(in_path, "hdf")
        segmentation = Segmentation.load(in_path, "hdf")
        data = pd.read_hdf(in_path, key="data", mode="r")

        return cls(
            segmentation=segmentation,
            import_data=data,
            zoning_system=zoning,
            cut_read=cut_read,
        )

    def translate_zoning(
        self,
        new_zoning: ZoningSystem,
        cache_path: Optional[PathLike] = None,
        trans_vector: pd.DataFrame = None,
        weighting: str | TranslationWeighting = TranslationWeighting.SPATIAL,
        check_totals: bool = True,
        no_factors: bool = False,
        one_to_one: bool = False,
        _bypass_validation: bool = False,
    ) -> DVector:
        """
        Translate this DVector into another zoning system and returns a new DVector.

        Parameters
        ----------
        new_zoning: ZoningSystem
            The zoning system to translate into.

        cache_path: Optional[PathLike]
            Path to a cache containing zoning translations.

        trans_vector: pd.DataFrame = None
            If provided this is the translation vector which will be used. If not
            provided this method will attempt to find one from cache_path.

        weighting : str | TranslationWeighting = TranslationWeighting.SPATIAL
            The weighting to use when building the zone_translation. Must be
            one of TranslationWeighting.

        check_totals: bool = True
            Whether to raise a warning if the translated total doesn't match the
            input total. Should be set to False for one-to-one translations.

        no_factors: bool = False
            Whether to run as a one-to-one zone_translation, e.g. all data will be
            multiplied by one, and zone numbers will change. This should only be
            used for perfectly nesting zone systems when disaggregating, e.g.
            msoa to lsoa.

        one_to_one: bool = False
            If True will force a one-to-one translation, where the largest factor is
            chosen for each zone combination. This will turn on no_factors as well.

        Returns
        -------
        translated_dvector:
            This DVector translated into new_new_zoning zoning system

        Warns
        -----
        TranslationWarning
            If there are zone IDs missing from the zone_translation or the
            zone_translation factors don't sum to 1.
        """
        # Validate inputs
        if not isinstance(new_zoning, ZoningSystem):
            raise ValueError(
                "new_zoning is not the correct type. "
                f"Expected ZoningSystem, got {type(new_zoning)}"
            )

        if self.zoning_system is None:
            raise ValueError(
                "Cannot translate the zoning system of a DVector that does "
                "not have a zoning system to begin with."
            )

        # If we're translating to the same thing, return a copy
        if self.zoning_system == new_zoning:
            return self.copy()

        # Translation validation is handled by ZoningSystem with TranslationWarning
        if trans_vector is None:
            if cache_path is None:
                trans_vector = self.zoning_system.translate(new_zoning, weighting=weighting)
            else:
                trans_vector = self.zoning_system.translate(
                    new_zoning, weighting=weighting, cache_path=cache_path
                )
        factor_col = self.zoning_system.translation_column_name(new_zoning)
        # factors equal one to propagate perfectly
        # This only works for perfect nesting
        if one_to_one:
            idx = trans_vector.groupby(f"{normalise_column_name(self.zoning_system.name)}_id")[
                factor_col
            ].idxmax()
            trans_vector = trans_vector.loc[idx]
            no_factors = True
        if no_factors:
            trans_vector[factor_col] = 1
        # Use a simple replace and group for nested zoning
        if trans_vector[
            f"{normalise_column_name(self.zoning_system.name)}_id"
        ].nunique() == len(trans_vector):
            if set(trans_vector[self.zoning_system.column_name]).intersection(
                self.zoning_system.zone_ids
            ) != set(self.zoning_system.zone_ids):
                warnings.warn(
                    "Not all zones in the DVector or defined in the zone_translation."
                )
            trans_vector = trans_vector.set_index(self.zoning_system.column_name)[
                new_zoning.column_name
            ].to_dict()
            translated = self.data.rename(columns=trans_vector).T.groupby(level=0).sum().T
            new_zones = set(trans_vector.values())
            untranslated = [i for i in translated.columns if i not in new_zones]
            if len(untranslated) > 0:
                warnings.warn(f"{untranslated} zones not translated. These are being dropped.")
            translated.drop(untranslated, axis=1, inplace=True)
            return DVector(
                zoning_system=new_zoning,
                segmentation=self.segmentation,
                time_format=self.time_format,
                import_data=translated,
                low_memory=self.low_memory,
                _bypass_validation=_bypass_validation,
                cut_read=self._cut_read,
            )

        transposed = self.data.transpose()
        transposed.index.names = [self.zoning_system.column_name]
        translated = ctk.translation.pandas_vector_zone_translation(
            transposed,
            trans_vector,
            translation_from_col=self.zoning_system.column_name,
            translation_to_col=new_zoning.column_name,
            translation_factors_col=factor_col,
            check_totals=check_totals,
        )
        translated = translated.transpose()

        return DVector(
            zoning_system=new_zoning,
            segmentation=self.segmentation,
            time_format=self.time_format,
            import_data=translated,
            low_memory=self.low_memory,
            _bypass_validation=_bypass_validation,
            cut_read=self._cut_read,
        )

    def split_by_agg_zoning(
        self, agg_zoning: ZoningSystem, trans: pd.DataFrame | None = None
    ) -> dict[int, DVector]:
        """
        Split a DVector into a different new DVector for each aggregate zone.

        Returns a dictionary with keys of agg_zone id and values of DVectors. The
        returned dvectors are still in original zoning, but each will be a subset of
        the full zone system.

        Parameters
        ----------
        agg_zoning: ZoningSystem
            The zoning system to split self by. Must be a strict aggregation of self.zoning_system
        trans: pd.DataFrame | None = None
            The translation to use to split the DVector. If left as None this will be
            found using the DVector method.

        Returns
        -------
        dict[int, DVector]
            A dict of self split into DVectors for each zone in agg_zoning
        """
        if trans is None:
            trans = self.zoning_system.translate(agg_zoning)
        else:
            trans = self.zoning_system.validate_translation_data(agg_zoning, trans)
        # not nested
        if trans[f"{self.zoning_system}_id"].nunique() < len(trans):
            raise TranslationError(
                "split_by_agg_zoning only works when the current zone system "
                "nests perfectly within the agg zone system, i.e. each zone in "
                "the current zone system corresponds to only 1 zone in the agg "
                "zone system."
            )
        out_dvecs = {}
        for zone in trans[agg_zoning.column_name].unique():
            zones = trans[trans[agg_zoning.column_name] == zone][
                self.zoning_system.column_name
            ]
            zones = self.data.columns.intersection(zones)
            new_data = self.data[zones]
            out_dvecs[zone] = DVector(
                import_data=new_data,
                segmentation=self.segmentation,
                zoning_system=self.zoning_system,
                cut_read=self._cut_read,
            )
        return out_dvecs

    def copy(self, _bypass_validation: bool = True):
        """Class copy method."""
        if self._zoning_system is not None:
            out_zoning = self._zoning_system.copy()
        else:
            out_zoning = None
        return DVector(
            segmentation=self._segmentation.copy(),
            zoning_system=out_zoning,
            import_data=self._data.copy(),
            time_format=self.time_format,
            val_col=self.val_col,
            _bypass_validation=_bypass_validation,
            cut_read=self._cut_read,
        )

    def overlap(self, other):
        """Call segmentation overlap method to check two DVectors contain overlapping segments."""
        overlap = self.segmentation.overlap(other.segmentation)
        if not overlap:
            raise NotImplementedError(
                "There are no common segments between the "
                "two DVectors so this operation is not "
                "possible."
            )

    def _generic_dunder(
        self,
        other,
        df_method,
        series_method,
        how: Literal["inner", "outer"] = "inner",
        escalate_warnings: bool = False,
        _bypass_validation: bool = False,
    ):
        """
        Stop telling me to use the imperative mood pydocstyle.

        A generic dunder method which is called by each of the dunder methods.

        Parameters
        ----------
        df_method:
            A pd.DataFrame method to be called on self.data with other.data.
        series_method:
            The equivalent series method to df_method.
        how: Literal["inner", "outer"] = "inner"
            Whether the method should implicitly join inner or outer. If outer
            the non-matching indices will be infilled with zeros
        escalate_warnings:
            Whether to escalate warnings to errors.
        """
        drop_na = False
        if isinstance(other, Number):
            if isinstance(self.data, pd.DataFrame):
                prod = df_method(self.data, other)
            else:
                prod = series_method(self.data, other)
            return DVector(
                import_data=prod,
                segmentation=self.segmentation,
                zoning_system=self.zoning_system,
                time_format=self.time_format,
                _bypass_validation=True,
                cut_read=self._cut_read,
            )
        if escalate_warnings:
            warnings.filterwarnings("error", category=SegmentationWarning)
        # Make sure the two DVectors have overlapping indices
        self.overlap(other)
        subset_diff = self.segmentation.subset_difference(other.segmentation)
        if subset_diff is not None:
            missing_other = subset_diff[1]
            if how == "inner":
                if len(missing_other) > 0:
                    warnings.warn(
                        "There are subsets in other's segmentation not in self. "
                        "This will lead to rows present in self being cut in the product. "
                        f"Missing values are {missing_other}",
                        SegmentationWarning,
                    )
                drop_na = True
            else:
                if self.segmentation.overlap(other.segmentation) != set(
                    other.segmentation.naming_order
                ):
                    excess = [
                        i for i in other.segmentation.names if i not in self.segmentation.names
                    ]
                    raise SegmentationError(
                        "An outer dunder method cannot be performed where other "
                        f"contains segments not present in self. {excess} in "
                        f"other but not in self."
                    )

        out = self.copy()
        # Takes exclusions into account before operating
        if len(self.segmentation) < len(other.segmentation):
            out = self.expand_to_other(other)

        # Alternatively could just try the normal method and use the low memory if an exception is raised
        if self.low_memory:
            raise NotImplementedError("The low memory method is not currently implemented.")
        #     # Assume that low memory means there are zoning systems
        #     if self.zoning_system != other.zoning_system:
        #         raise ValueError("Zonings don't match.")
        #     zoning = self.zoning_system
        #     common_segs = self.segmentation.overlap(other.segmentation)
        #     max_len = 0
        #     storage_seg = None
        #     for seg in common_segs:
        #         seg_len = len(self.segmentation.seg_dict[seg])
        #         # Indexing the temp storage by the longest segment
        #         if seg_len > max_len:
        #             max_len = seg_len
        #             storage_seg = seg
        #     # Temp dir to save inputs in
        #     with tempfile.TemporaryDirectory() as temp_dir:
        #         temp_self = Path(temp_dir) / "temp_self.hdf"
        #         temp_other = Path(temp_dir) / "temp_other.hdf"
        #         for ind in self.segmentation.seg_dict[storage_seg].values.keys():
        #             self.data.xs(ind, level=storage_seg).to_hdf(
        #                 temp_self, mode="a", key=f"node_{ind}"
        #             )
        #             other.data.xs(ind, level=storage_seg).to_hdf(
        #                 temp_other, mode="a", key=f"node_{ind}"
        #             )
        #         # TODO potentially delete one or both of the input DVectors
        #         out_data = {}
        #         for ind in self.segmentation.seg_dict[storage_seg].values.keys():
        #             self_section = pd.read_hdf(temp_self, key=f"node_{ind}")
        #             other_section = pd.read_hdf(temp_other, key=f"node_{ind}")
        #             out_data[ind] = df_method(self_section, other_section)
        #         prod = pd.concat(out_data, names=[seg] + out_data[ind].index.names)

        else:
            # for the same zoning a simple * gives the desired result
            # This drops any nan values (intersecting index level but missing val)
            if self.zoning_system == other.zoning_system:
                if isinstance(self.data, pd.Series):
                    prod = series_method(out.data, other.data)
                else:
                    return_zones = out.data.columns.intersection(other.data.columns)
                    prod = df_method(out.data[return_zones], other.data[return_zones])
                # Either None if both are None, or the right zone system
                zoning = self.zoning_system

            # For a dataframe by a series the mul is broadcast across
            # for this to work axis needs to be set to 'index'
            elif self.zoning_system is None:
                # Allowed but warned
                logging.warning(
                    "For this method to work between a DVector with "
                    "a zoning system and a DVector without one, the "
                    "DVector with a zoning system must come first. "
                    "This is being changed internally but if this was "
                    "not expected, check your inputs"
                )
                prod = df_method(other.data, out.data.squeeze(), axis="index")
                zoning = other.zoning_system
            elif other.zoning_system is None:
                prod = df_method(out.data, other.data.squeeze(), axis="index")
                zoning = self.zoning_system
            # Different zonings raise an error rather than trying to translate
            else:
                raise NotImplementedError(
                    "The two DVectors have different zonings. "
                    "To multiply them, one must be translated "
                    "to match the other."
                )
        # Index unchanged, aside from possible order. Segmentation remained the same
        if drop_na:
            prod.dropna(inplace=True)
        else:
            prod.fillna(self.data, inplace=True)
        prod.sort_index(inplace=True)
        if prod.index.equals(self._data.index):
            return DVector(
                segmentation=self.segmentation,
                import_data=prod,
                zoning_system=zoning,
                _bypass_validation=_bypass_validation,
                cut_read=self._cut_read,
            )
        # Index changed so the segmentation has changed. Segmentation should equal
        # the addition of the two segmentations (see __add__ method in segmentation)
        new_seg = self.segmentation + other.segmentation
        warnings.warn(
            f"This operation has changed the segmentation of the DVector "
            f"from {self.segmentation.names} to {new_seg.names}. This can happen"
            " but it can also be a sign of an error. Check the output DVector.",
            SegmentationWarning,
        )

        prod = prod.reorder_levels(new_seg.naming_order)
        if not prod.index.equals(new_seg.ind()):
            warnings.warn(
                "This operation has dropped some rows due to exclusions "
                f"in the resulting segmentation. {prod.index.difference(new_seg.ind())} "
                f"rows have been dropped from the pure product."
            )
            prod = prod.loc[new_seg.ind()]

        return DVector(
            segmentation=new_seg,
            import_data=prod,
            zoning_system=zoning,
            _bypass_validation=_bypass_validation,
            cut_read=self._cut_read,
        )

    def __len__(self):
        """Return the length of a DVector, defined as number of cells."""
        return len(self.segmentation) * len(self.zoning_system)

    def __pow__(self, exponent: int | float, _bypass_validation: bool = True):
        """Return the exponent of a DVector, essentially a wrapper around DataFrame's __pow__ method."""
        out_data = self.data**exponent
        return DVector(
            import_data=out_data,
            segmentation=self.segmentation,
            zoning_system=self.zoning_system,
            time_format=self.time_format,
            low_memory=self.low_memory,
            _bypass_validation=_bypass_validation,
            cut_read=self._cut_read,
        )

    def __mul__(
        self, other, _bypass_validation: bool = False, how: Literal["inner", "outer"] = "inner"
    ):
        """Multiply dunder method for DVector."""
        return self._generic_dunder(
            other,
            pd.DataFrame.mul,
            pd.Series.mul,
            _bypass_validation=_bypass_validation,
            how=how,
        )

    def __add__(
        self, other, _bypass_validation: bool = False, how: Literal["inner", "outer"] = "inner"
    ):
        """Add dunder method for DVector."""
        return self._generic_dunder(
            other,
            pd.DataFrame.add,
            pd.Series.add,
            _bypass_validation=_bypass_validation,
            how=how,
        )

    def __sub__(
        self, other, _bypass_validation: bool = False, how: Literal["inner", "outer"] = "inner"
    ):
        """Subtract dunder method for DVector."""
        return self._generic_dunder(
            other,
            pd.DataFrame.sub,
            pd.Series.sub,
            _bypass_validation=_bypass_validation,
            how=how,
        )

    def __truediv__(
        self, other, _bypass_validation: bool = False, how: Literal["inner", "outer"] = "inner"
    ):
        """Division dunder method for DVector."""
        return self._generic_dunder(
            other,
            pd.DataFrame.div,
            pd.Series.div,
            _bypass_validation=_bypass_validation,
            how=how,
        )

    def __eq__(self, other):
        """Equals dunder for DVector."""
        if self.zoning_system != other.zoning_system:
            return False
        if self.segmentation != other.segmentation:
            return False
        if not self.data.equals(other.data):
            return False
        return True

    def __ne__(self, other):
        """Note equals dunder for DVector."""
        return not self.__eq__(other)

    def aggregate(self, segs: list[str] | Segmentation, _bypass_validation: bool = False):
        """
        Aggregate DVector to new segmentation.

        New Segmentation must be a subset of the current segmentation. Currently
        this method is essentially a pandas 'groupby.sum()', but other methods
        could be called if needed (e.g. mean())

        This method ignores 'subsets' in segs, if segs is provided as a Segmentation
        rather than a list of strings. Subsets must be accounted for separately.

        Parameters
        ----------
        segs: Segments to aggregate to. Must be a subset of self.segmentation.naming_order,
        naming order will be preserved.
        """
        if isinstance(segs, Segmentation):
            segs = segs.naming_order
        if not isinstance(segs, list):
            raise TypeError(
                "Aggregate expects a list of strings. Even if you "
                "are aggregating to a single level, this should be a "
                "list of length 1."
            )
        segmentation = self.segmentation.aggregate(segs)
        data = self.data.groupby(level=segs).sum()
        return DVector(
            segmentation=segmentation,
            import_data=data,
            zoning_system=self.zoning_system,
            time_format=self.time_format,
            val_col=self.val_col,
            _bypass_validation=_bypass_validation,
            cut_read=self._cut_read,
        )

    def split_by_other(self, other: DVector, agg_zone: ZoningSystem | None = None):
        """
        Split a DVector adding new segments.

        Uses other as weighting, such that the returned DVector sums to the same
        as the input.

        Parameters
        ----------
        other: DVector
            The DVector to use for splitting. Returned DVector will have the
            segmentation of this DVector, with splitting weighted by this DVector.

        agg_zone: ZoningSystem
            The zoning level the splits will be calculated at. This should be more aggregate
            the more confident you are in your attractions distribution spatially, on a scale from
            None if you are very confident in attractions, to model zoning for no confidence
            (choosing model zoning means attractions will essentially mirror productions exactly).
        """
        if agg_zone is None:
            raise ValueError(
                "agg_zone must be provided. To run this process with no agg_zone "
                "please use the expand_to_other method."
            )
        if other.zoning_system != self.zoning_system:
            raise ValueError(
                "The 'other' DVector used for splitting must be "
                "of the same zoning as 'self'. Self has zoning system "
                f"{self.zoning_system.name}, other has {other.zoning_system.name}."
            )

        common = self.segmentation.overlap(other.segmentation)
        other_grouped_data = other.data.groupby(level=common).sum()
        translation = self.zoning_system.translate(agg_zone)
        if not (translation[self.zoning_system.translation_column_name(agg_zone)] == 1).all():
            raise TranslationError(
                "Current zoning must nest perfectly within agg_zone, "
                "i.e. all factors should be 1. The retrieved zone_translation "
                "has non-one factors. If this should not be the case "
                "double check the zone_translation."
            )
        translation_dict = translation.set_index(self.zoning_system.column_name)[
            agg_zone.column_name
        ].to_dict()
        translated_grouped = (
            other_grouped_data.rename(columns=translation_dict).groupby(level=0, axis=1).sum()
        )
        translated_ungrouped = (
            other.data.rename(columns=translation_dict).groupby(level=0, axis=1).sum()
        )
        # factors at common segmentation and agg zoning
        translated = translated_ungrouped / translated_grouped
        # Translate zoning back to DVec zoning to apply to DVector
        splitting_data = ctk.translation.pandas_vector_zone_translation(
            vector=translated.T,
            translation=translation,
            translation_from_col=agg_zone.column_name,
            translation_to_col=self.zoning_system.column_name,
            translation_factors_col=self.zoning_system.translation_column_name(agg_zone),
        ).T
        # Put splitting factors into DVector to apply
        splitting_dvec = DVector(
            import_data=splitting_data,
            segmentation=other.segmentation,
            zoning_system=other.zoning_system,
            time_format=other.time_format,
            val_col=other.val_col,
            low_memory=other.low_memory,
            cut_read=self._cut_read,
        )
        return self * splitting_dvec

    def add_segments(
        self,
        new_segs: list[Segment],
        new_naming_order: Optional[list[str]] = None,
        split_method: Literal["split", "duplicate"] = "duplicate",
        splitter: pd.Series | None = None,
    ):
        """
        Add a segment to a DVector.

        The new segment will multiply the length of the DVector, usually by the
        length of the new segment (but less if an exclusion is introduced between
        the new segment and the current segmentation).

        Parameters
        ----------
        new_segs: Segment
            The new segment to be added. This will be checked and added as an
            enum_segment if it exists as such, and as a custom segment if not.
            This must be provided as a Segment type, and can't be a string to pass
            to the SegmentSuper enum class

        new_naming_order: Optional[list[str]] = None
            The naming order of the resultant segmentation. If not provided,
            the new segment will be appended to the end.

        split_method: Literal["split", "duplicate"] = "duplicate"
            How to deal with the values in the current DVector. "split" will
            split values into the new segment, conserving the sum of the current
            DVector. Duplicate will keep all values the same and duplicate them
            into the new DVector.

        splitter: pd.Series | None = None
            A series with the expanded index. Self.data will be multiplied by this
            to expand its segmentation. If this isn't provided it will be all ones.

        Returns
        -------
        DVector
        """
        new_segmentation = self.segmentation.copy()
        if len(new_segs) == 0:
            raise ValueError("no new segments provided")

        for seg in new_segs[:-1]:
            new_segmentation = new_segmentation.add_segment(seg)
        new_segmentation = new_segmentation.add_segment(
            new_segs[-1], new_naming_order=new_naming_order
        )

        if splitter is None:
            splitter = pd.Series(index=new_segmentation.ind(), data=1)
        if split_method == "split":
            # This method should split evenly, even in the case of exclusions
            factor = splitter.groupby(level=self.segmentation.naming_order).sum()
            splitter /= factor
        new_data = self._data.mul(splitter, axis=0)
        new_data = new_data.reorder_levels(new_segmentation.naming_order)
        if new_data.index.equals(new_segmentation.ind()):
            return DVector(
                segmentation=new_segmentation,
                zoning_system=self.zoning_system,
                import_data=new_data,
                cut_read=self._cut_read,
            )
        if new_data.drop(new_segmentation.ind()).sum().sum() == 0:
            SegmentationWarning(
                f"{new_data.drop(new_segmentation.ind()).index} being dropped."
                f"These rows contain no data."
            )
            return DVector(
                segmentation=new_segmentation,
                zoning_system=self.zoning_system,
                import_data=new_data.loc[new_segmentation.ind()],
                cut_read=self._cut_read,
            )
        raise ValueError("Generated index doesn't match the index of the new " "data.")

    def expand_to_other(self, other: DVector, match_props: bool = False) -> DVector:
        """
        Expand self segmentation to other.

        This adds in all segments in other which are not in self, regardless of
        whether all segments in self are also in other.

        Parameters
        ----------
        other: DVector
            The DVector to expand to
        match_props: bool = False
            Whether to use other to determine proportions for splitting. If True, other
            summed over zones will be used as weighting for the split.

        Returns
        -------
        self expanded to other as required.
        """
        expansion_segs = other.segmentation - self.segmentation
        if match_props:
            splitter = other.data.sum(axis=1)
            return self.add_segments(expansion_segs, split_method="split", splitter=splitter)

        return self.add_segments(expansion_segs)

    @classmethod
    def combine_from_dic(
        cls,
        in_dic: dict[int, DVector],
        new_seg: Segment,
        in_segmentation: Segmentation,
        zoning_system: ZoningSystem,
    ):
        """
        Combine DVectors saved in a dictionary.

        The dictionary keys form values of the added segment.

        Parameters
        ----------
        in_dic: dic[int, DVector]
            Dictionary containing keys of segment values, and DVectors to be combined.
        new_seg: Segment
            The Segment to be added in. Values must match the keys of in_dic
        in_segmentation: Segmentation
            The Segmentation of the DVectors in in_dic. Segmentation must be the same for
            all of them
        zoning_system: ZoningSystem
            The zoning_system of the DVectors in in_dic. Must be the same for all.

        Returns
        -------
        DVector combination of DVector in in_dic
        """
        comb = {val: dvec.data for val, dvec in in_dic.items()}
        new_data = pd.concat(comb)
        new_segmentation = in_segmentation.add_segment(new_seg)
        new_data = new_data.reorder_levels(new_segmentation.naming_order).sort_index()
        return cls(
            segmentation=new_segmentation, import_data=new_data, zoning_system=zoning_system
        )

    def select_zone(self, zone_id) -> DVector:
        """
        Return a DVector for a single zone in a DVector.

        Parameters
        ----------
        zone_id: int
            The zone to select.

        Returns
        -------
        DVector:
            A DVector for a single zone, data will be a series.
        """
        out_data = self.data[zone_id]
        return DVector(
            import_data=out_data,
            segmentation=self.segmentation,
            zoning_system=None,
            time_format=self.time_format,
            cut_read=self._cut_read,
        )

    def filter_segment_value(self, segment_name: str, segment_values: int | list[int]):
        """
        Filter a DVector on a given segment.

        Equivalent to .loc/.xs in pandas.

        Parameters
        ----------
        segment_name: str
            The name of the segment to filter by.
        segment_values: int | list[int]
            The segment values to filter by. If an int is given, the segment is
            dropped from the returned DVector, otherwise the output DVector will
            contain a subset of the segment.
        """
        new_data = self.data.copy()
        new_seg = self.segmentation.copy()
        if isinstance(self.segmentation.ind(), pd.MultiIndex):
            if isinstance(segment_values, list):
                new_data = new_data[
                    new_data.index.get_level_values(level=segment_name).isin(segment_values)
                ]
            else:
                new_data = new_data.xs(segment_values, level=segment_name)
        else:
            new_data = new_data.loc[segment_values]

        if isinstance(segment_values, int):
            new_seg = new_seg.remove_segment(segment_name)
        else:
            new_seg = new_seg.update_subsets({segment_name: segment_values})

        return DVector(
            import_data=new_data,
            segmentation=new_seg,
            zoning_system=self.zoning_system,
            time_format=self.time_format,
            val_col=self.val_col,
            low_memory=self.low_memory,
            cut_read=self._cut_read,
        )

    def drop_by_segment_values(self, segment_name: str, segment_values: list[int]):
        """Remove rows of DVector based on segment values.

        Parameters
        ----------
        segment_name: str
            The name of the segment to select values from.
        segment_values: list[int]
            The values to drop.
        """
        new_data = self.data.copy()
        if isinstance(self.segmentation.ind, pd.MultiIndex):
            new_data = self.data.drop(segment_values, level=segment_name)
        else:
            new_data = new_data.drop[segment_values]

        new_seg = self.segmentation.update_subsets({segment_name, segment_values}, remove=True)

        return DVector(
            import_data=new_data,
            segmentation=new_seg,
            zoning_system=self.zoning_system,
            time_format=self.time_format,
            val_col=self.val_col,
            low_memory=self.low_memory,
            cut_read=self._cut_read,
        )

    def fillna(self, infill_value: float | int):
        """Wrap fillna dataframe method."""
        self.data = self.data.fillna(infill_value)

    def fill(self, find_value: float | int, infill_value: float | int):
        self.data = self.data.replace(to_replace={find_value: infill_value})

    def translate_segment(
        self, from_seg, to_seg, reverse=False, drop_from=True, _bypass_validation: bool = False
    ):
        """
        Translate a segment in the DVector.

        Parameters
        ----------
        from_seg:
            Segment in DVector to translate.
        to_seg:
            Segment to translate from_seg to. A lookup for these must be defined
            in the seg_translations folder.
        reverse: bool = False
            Whether to do this translation in reverse; by default translations are
            defined as aggregations, running in reverse will not conserve totals.
        drop_from: bool = True
            Whether to remove from_seg from the resulting segmentation.
        """
        new_segmentation, lookup = self.segmentation.translate_segment(
            from_seg, to_seg, reverse=reverse, drop_from=drop_from
        )
        if reverse:
            lookup = lookup.to_frame().set_index(from_seg, append=True).index
            new_data = self.data.reindex(lookup, level=from_seg)
            if drop_from:
                new_data = new_data.droplevel(from_seg)
        else:
            try:
                new_data = self.data.join(lookup).reset_index()
            except AttributeError:
                new_data = self.data.to_frame().join(lookup).reset_index()
            if drop_from:
                new_data.drop(from_seg, axis=1, inplace=True)
            new_data = new_data.groupby(new_segmentation.naming_order).sum()

        return DVector(
            import_data=new_data,
            segmentation=new_segmentation,
            zoning_system=self.zoning_system,
            time_format=self.time_format,
            val_col=self.val_col,
            low_memory=self.low_memory,
            _bypass_validation=_bypass_validation,
            cut_read=self._cut_read,
        )

    def trans_seg_from_lookup(self, lookup: SegConverter, drop_old: bool = False):
        """
        Translate segment(s) in self to new segment(s) from a lookup in SegConverter.

        Parameters
        ----------
        lookup: SegConverter
            The name of a lookup in SegConverter
        drop_old: bool = False
            Whether to drop old segments from resulting DVector
        """
        lookup = SegConverter(lookup).get_conversion()
        drop_names = lookup.index.names
        new_names = lookup.columns
        new_seg = self.segmentation.copy()
        for name in drop_names:
            if name not in self.segmentation.names:
                raise ValueError(
                    f"{name} not in current segmentation so can't" f"be used to convert."
                )
            if drop_old:
                new_seg.remove_segment(name, inplace=True)
        for name in new_names:
            new_seg = new_seg.add_segment(SegmentsSuper(name).get_segment())

        new_data = self.data.join(lookup, how="left").reset_index()

        if drop_old:
            new_data.drop(columns=drop_names, inplace=True)
        new_data = new_data.groupby(new_seg.naming_order).sum()

        return DVector(
            import_data=new_data,
            segmentation=new_seg,
            zoning_system=self.zoning_system,
            time_format=self.time_format,
            low_memory=self.low_memory,
            val_col=self.val_col,
            cut_read=self._cut_read,
        )

    def calc_rmse(self, targets: list[IpfTarget]) -> float:
        """
        Calculate the rmse relative to a set of targets.

        Parameters
        ----------
        targets: list[IpfTarget]
            The targets to calc rmse relative to.
        """
        mse = 0
        for target in targets:
            check = self.copy()
            if target.zone_translation is not None:
                check = self.translate_zoning(
                    target.data.zoning_system,
                    trans_vector=target.zone_translation,
                    _bypass_validation=True,
                )
            if target.segment_translations is not None:
                for seg in target.data.segmentation - self.segmentation:
                    seg = seg.name
                    if target.segment_translations[seg] in self.segmentation.names:
                        lower_seg = self.segmentation.get_segment(
                            target.segment_translations[seg]
                        )
                        check = check.translate_segment(
                            from_seg=lower_seg.name, to_seg=seg, _bypass_validation=True
                        )
                    else:
                        raise ValueError("No translation defined for this segment.")
            diff = (
                check.aggregate(target.data.segmentation, _bypass_validation=True).__sub__(
                    target.data, _bypass_validation=True
                )
            ) ** 2
            mse += diff.sum() / len(target.data)
        return mse**0.5

    def validate_ipf_targets(
        self,
        targets: Collection[IpfTarget],
        rel_tol: float = 1e-5,
        cache_path: None | PathLike = None,
    ):
        """
        Check targets for ipf will work, raises errors if not.

        Parameters
        ----------
        targets: list[IpfTarget]
            List of IPF targets to validate.

        rel_tol: float = 1e-5
            The tolerance for relative difference in sums of targets.

        cache_path: None | PathLike = None
            Translations cache path, if None reverts to default.

        Returns
        -------
        Input targets, with zone translations added in if relevant.
        """
        target_sum = 0.0
        for position, target in enumerate(targets):
            # Check targets sum to the same, or they can't converge. Potentially could allow
            # IPF for non-agreeing targets to get as close as possible.
            if target_sum == 0:
                target_sum = target.data.sum()
            else:
                # TODO don't hard code this
                if not math.isclose(target_sum, target.data.sum(), rel_tol=1e5):
                    raise ValueError(
                        "Input target DVectors do not have consistent "
                        f"sums, so ipf will fail target at position {position} doesn't match "
                        "the first target. It is possible later targets also don't match."
                    )
            # Check for zeros
            zeros = target.data.data.to_numpy() == 0
            if np.sum(zeros) > 0:
                warnings.warn(
                    f"There are {np.sum(zeros)} zeros in the target data, making "
                    f"up {np.sum(zeros) / zeros.size:.0%} of "
                    f"the target data. The more zeros, the worse the IPF process will work."
                )
            # Check segmentations are compatible.
            if not target.data.segmentation.is_subset(self.segmentation):
                if target.segment_translations is None:
                    raise ValueError(
                        f"The {position + 1}th target segmentation is not a subset of the seed "
                        "segmentation, but no correspondences are defined."
                    )
                non_matching = target.data.segmentation - self.segmentation
                for seg in non_matching:
                    seg = seg.name
                    if seg in target.segment_translations.keys():
                        if target.segment_translations[seg] in self.segmentation.names:
                            lower_seg = self.segmentation.get_segment(
                                target.segment_translations[seg]
                            )
                            try:
                                # Don't need this for now, checking it exists.
                                lower_seg.translate_segment(seg)
                            except FileNotFoundError:
                                raise FileNotFoundError(
                                    f"No segment translation found for {lower_seg} to {seg}.from."
                                )

            # Check zoning systems are compatible.
            if self.zoning_system != target.data.zoning_system:
                target.zoning_diff = True
                if target.zone_translation is None:
                    try:
                        if cache_path is not None:
                            target.zone_translation = self.zoning_system.translate(
                                target.data.zoning_system, cache_path=cache_path
                            )
                        else:
                            target.zone_translation = self.zoning_system.translate(
                                target.data.zoning_system
                            )
                    except TranslationError:
                        raise TranslationError(
                            "No zone_translation was found for "
                            f"{self.zoning_system} to {target.data.zoning_system}."
                        )
                nested = (
                    target.zone_translation[
                        self.zoning_system.translation_column_name(target.data.zoning_system)
                    ]
                    == 1
                ).all()
                if not nested:
                    raise TranslationError(
                        "For IPF any targets must either be at the same zoning "
                        "system as the seed DVector, or be at a zoning system "
                        "which the seed nests perfectly within. The zone_translation "
                        "found contains non-one factors, which implies the "
                        "zoning system doesn't nest, so IPF can't be performed."
                    )
        return targets

    def ipf(
        self,
        targets: Collection[IpfTarget],
        tol: float = 1e-5,
        max_iters: int = 100,
        zone_trans_cache: Path | None = None,
    ) -> tuple[DVector, float]:
        """
        Implement iterative proportional fitting for DVectors.

        See: https://en.wikipedia.org/wiki/Iterative_proportional_fitting

        Parameters
        ----------
        targets: list[IpfTarget]
            A list of targets to try and match. Due to the nature of the process,
            the final target in this list will be matched most closely to. This
            doesn't matter much when a good convergence is achieved, but can have
            a large impact when not.
        tol: float = 1e-5
            The RMSE between self and targets the process will attempt to reach
            before stopping. Can also stop if consecutive iterations score the same.
        max_iters: int = 100
            The max number of iterations which can run before the process will stop.
        zone_trans_cache: Path | None = None
            The cache to look for translations in

        Returns
        -------
        DVector matched to targets, rmse achieved.
        """
        # check DVectors compatible
        targets = self.validate_ipf_targets(targets, zone_trans_cache)
        new_dvec = self.copy()
        prev_rmse = np.inf
        rmse = np.inf
        bypass = False
        for i in range(max_iters):
            for target in targets:
                if i > 0:
                    bypass = True
                inner = new_dvec.copy()
                if target.segment_translations is not None:
                    for targ_seg, seed_seg in target.segment_translations.items():
                        inner = inner.translate_segment(
                            from_seg=seed_seg, to_seg=targ_seg, _bypass_validation=bypass
                        )
                agg = inner.aggregate(target.data.segmentation, _bypass_validation=bypass)
                if target.zone_translation is not None:
                    agg = agg.translate_zoning(
                        target.data.zoning_system,
                        trans_vector=target.zone_translation,
                        _bypass_validation=bypass,
                    )
                factor = target.data.__truediv__(agg, _bypass_validation=bypass)
                factor.fillna(0)
                if (factor.data.values == np.inf).any():
                    factor.fill(np.inf, 0)

                if target.zoning_diff:
                    factor = factor.translate_zoning(
                        self.zoning_system,
                        trans_vector=target.zone_translation,
                        check_totals=False,
                        no_factors=True,
                    )
                if target.segment_translations is not None:
                    for targ_seg, seed_seg in target.segment_translations.items():
                        factor = factor.translate_segment(
                            from_seg=targ_seg,
                            to_seg=seed_seg,
                            reverse=True,
                            _bypass_validation=bypass,
                        )
                new_dvec = new_dvec.__mul__(factor, _bypass_validation=bypass, how="outer")

            rmse = new_dvec.calc_rmse(targets)
            LOG.info(f"RMSE = {rmse} after {i + 1} iterations.")
            if rmse < tol:
                print("Convergence met, returning DVector.")
                return new_dvec, rmse
            if abs(rmse - prev_rmse) < tol:
                LOG.info(f"RMSE has stopped improving at {rmse}.")
                return new_dvec, rmse
            prev_rmse = rmse
        warnings.warn(
            "Convergence has not been met, and RMSE has not stopped improving. "
            f"Last cycle saw a {rmse - prev_rmse} improvement in RMSE. "
            "May converge with more iterations."
        )
        return new_dvec, rmse

    @staticmethod
    def old_to_new_dvec(import_data: dict):
        """
        Convert the old format of DVector into the new.

        This only applies to the new dataframe.
        """
        zoning = import_data["zoning_system"]["unique_zones"]
        data = import_data["data"].values()
        segmentation = import_data["data"].keys()
        naming_order = import_data["segmentation"]["naming_order"]
        # Convert list of segmentations into multiindex
        dict_list = []
        for string in segmentation:
            int_list = [int(x) for x in string.split("_")]
            row_dict = {naming_order[i]: value for i, value in enumerate(int_list)}
            dict_list.append(row_dict)
        ind = pd.MultiIndex.from_frame(pd.DataFrame(dict_list))
        return pd.DataFrame(data=data, index=ind, columns=zoning)

    def remove_zoning(self, fn: Callable = pd.DataFrame.sum) -> DVector:
        """
        Aggregate all the zone values in DVector into a single value using fn.

        Returns a copy of Dvector.

        Parameters
        ----------
        fn:
            The function to use when aggregating all zone values. fn must
            be able to take a np.array of values and return a single value
            in order for this to work.

        Returns
        -------
        summed_dvector:
            A copy of DVector, without any zoning.
        """
        # Validate fn
        if not callable(fn):
            raise ValueError(
                "fn is not callable. fn must be a function that "
                "takes an np.array of values and return a single value."
            )

        if self.zoning_system is None:
            raise ValueError("There is no zoning to remove.")

        # Aggregate all the data
        summed = fn(self.data, axis=1)

        return DVector(
            zoning_system=None,
            segmentation=self.segmentation,
            time_format=self.time_format,
            import_data=summed,
            cut_read=self._cut_read,
        )

    @classmethod
    def concat_from_dir(
        cls,
        folder: PathLike,
        zoning: ZoningSystem | None = None,
        segmentation: Segmentation | None = None,
    ):
        """
        Load all DVectors in a directory and concatenate them into a single DVector.

        Parameters
        ----------
        folder: PathLike
            The dir containing DVectors to be concatenated.
        zoning: ZoningSystem | None = None
            The zoning of the resulting DVector. If not provided this will be inferred
            from the first DVector read in, and subsequent DVectors will be translated
            to this zoning if they don't match.
        segmentation: Segmentation | None = None
            The Segmentation of the resulting DVector. If not provided this will be inferred
            from the first DVector read in, and subsequent DVectors will be aggregated
            to this segmentation if necessary and possible.

        Returns
        -------
        DVector
        """
        folder = Path(folder)
        dvecs = []
        for file in listdir(folder):
            if file.endswith("hdf"):
                try:
                    dvec = cls.load(folder / file)
                except:
                    continue
                if zoning is None:
                    zoning = dvec.zoning_system
                else:
                    if dvec.zoning_system != zoning:
                        dvec = dvec.translate_zoning(zoning)
                if segmentation is None:
                    segmentation = dvec.segmentation
                else:
                    if dvec.segmentation != segmentation:
                        if segmentation.is_subset(dvec.segmentation):
                            dvec = dvec.aggregate(segmentation)
                        else:
                            raise SegmentationError(
                                "Dvec cannot be aggregated to a segmentation which is "
                                "not a subset of the current segmentation."
                            )
                dvecs.append(dvec)
        new_data = pd.concat([dvec.data for dvec in dvecs], axis=1)
        return cls(
            import_data=new_data,
            segmentation=segmentation,
            zoning_system=zoning,
        )

    def sum_zoning(self):
        """Sum over zones."""
        return self.remove_zoning()

    def to_ie(self):
        """Convert zoning to internal/external."""
        new_data = self.data.rename(columns=self.zoning_system.id_to_external)
        new_data = new_data.T.groupby(level=0).sum().T
        new_data.columns = [int(i) + 1 for i in new_data.columns]
        new_zoning = ZoningSystem.get_zoning("ie_sector")

        return DVector(
            import_data=new_data,
            zoning_system=new_zoning,
            segmentation=self.segmentation,
            time_format=self.time_format,
            cut_read=self._cut_read,
        )

    def write_sector_reports(
        self,
        segment_totals_path: PathLike,
        ca_sector_path: PathLike,
        ie_sector_path: PathLike,
        lad_report_path: PathLike | None = None,
        lad_report_seg: Segmentation | None = None,
    ) -> None:
        """
        Write segment, CA sector, and IE sector reports to disk.

        Parameters
        ----------
        segment_totals_path:
            Path to write the segment totals report to

        ca_sector_path:
            Path to write the CA sector report to

        ie_sector_path:
            Path to write the IE sector report to

        lad_report_path:
            Path to write the LAD report to

        lad_report_seg:
            The segmentation to output the LAD report at

        Returns
        -------
        None
        """
        # Check that not just one argument has been set
        if bool(lad_report_path) != bool(lad_report_seg):
            raise ValueError(
                "Only one of lad_report_path and lad_report_seg has been set. "
                "Either both values need to be set, or neither."
            )

        # Segment totals report
        df = self.sum_zoning().data
        df.to_csv(segment_totals_path)

        # Segment by CA Sector total reports - 1 to 1, No weighting
        # try:
        tfn_ca_sectors = ZoningSystem.get_zoning("ca_sector_2020")
        dvec = self.translate_zoning(tfn_ca_sectors)
        dvec.data.to_csv(ca_sector_path)
        # except Exception as err:
        #     LOG.error("Error creating CA sector report: %s", err)

        # Segment by IE Sector total reports - 1 to 1, No weighting
        try:
            dvec = self.to_ie()
            dvec.data.to_csv(ie_sector_path)
        except Exception as err:
            LOG.error("Error creating IE sector report: %s", err)

        if lad_report_seg is None:
            return

        # Segment by LAD segment total reports - 1 to 1, No weighting
        try:
            lad = ZoningSystem.get_zoning("lad_2020")
            dvec = self.aggregate(lad_report_seg)
            dvec = dvec.translate_zoning(lad)
            dvec.data.to_csv(lad_report_path)
        except Exception as err:
            LOG.error("Error creating LAD report: %s", err)

    def sum(self) -> float:
        """Sum DVector."""
        if isinstance(self.data, pd.DataFrame):
            return self.data.values.sum()
        if isinstance(self.data, pd.Series):
            return self.data.sum()
        raise ValueError("This error can't be raised but mypy is complaining.")

    def sum_is_close(self, other: DVector | float, rel_tol: float, abs_tol: float):
        """
        Check if self sums close to other.

        Calls math.isclose on respective sums.

        Parameters
        ----------
        other: DVector
            The other DVector to compare self to
        rel_tol: float
            See math.isclose
        abs_tol: float
            see math.isclose
        """
        if isinstance(other, DVector):
            other_sum = other.sum()
        else:
            other_sum = other
        return math.isclose(self.sum(), other_sum, rel_tol=rel_tol, abs_tol=abs_tol)

    def concat(self, other: DVector):
        """
        Analogous to pandas dataframe concat method.

        The DVector being concatenated must have the same zoning system, and the same segmentation
        levels. The segmentations of each must not overlap, meaning that in the case of enum
        segments used, both must contain subsets.
        """
        new_seg = self.segmentation.copy()
        intersection = self.data.index.intersection(other.data.index)
        # if data.segmentation != self.segmentation:
        #     raise ValueError("Additional data has incorrect segmentation.")
        if other.zoning_system != self.zoning_system:
            raise ValueError("Zoning systems don't match.")
        if len(intersection) > 0:
            raise ValueError("There is an overlap in indices.")
        try:
            new_data = other.data.reorder_levels(self.segmentation.naming_order)
        except TypeError:
            new_data = other.data
        for name in self.segmentation.naming_order:
            own_vals = self.segmentation.seg_dict[name].int_values
            new_vals = other.segmentation.seg_dict[name].int_values
            additional = [i for i in new_vals if i not in own_vals]
            if len(additional) > 0:
                try:
                    new_seg.input.subsets[name] += additional
                    if new_seg.input.subsets[name] == list(
                        SegmentsSuper(name).get_segment().values.keys()
                    ):
                        del new_seg.input.subsets[name]
                except KeyError:
                    continue
        new_data = pd.concat([self.data, new_data])
        return DVector(
            import_data=new_data,
            segmentation=new_seg.reinit(),
            zoning_system=self.zoning_system,
            time_format=self.time_format,
        )

    def add_value_to_subset(self, segment: str, value: int, data: pd.DataFrame):
        """
        Add a value to a subset segment in a DVector

        Parameters
        ----------
        segment: str
            The target segment in self.Segmentation
        value: int
            The value to add to the target subset
        data:
            The data associated with the new value to be added. This data should
            contain the same segment names as self.

        Returns
        -------
        DVector
        """
        warnings.warn(
            "This method is being deprecated and won't be available in future."
            "Please use the concat method instead."
        )
        new_seg = self.segmentation.copy()
        new_seg.input.subsets[segment].append(value)
        new_seg = new_seg.reinit()
        new_data = self.data
        if data.index.names != new_data.index.names:
            data[segment] = value
            data.set_index(segment, append=True, inplace=True)
            data.index = data.index.reorder_levels(self.segmentation.naming_order)
        new_data = pd.concat([new_data, data])
        return DVector(
            import_data=new_data,
            segmentation=new_seg,
            zoning_system=self.zoning_system,
            time_format=self.time_format,
            cut_read=self._cut_read,
        )

    @staticmethod
    def _balance_zones_internal(
        self_data: pd.DataFrame,
        self_zoning: ZoningSystem,
        other_data: pd.DataFrame,
        other_zoning: ZoningSystem,
        balancing_zones: ZoningSystem,
    ):
        self_trans = self_zoning.translate(balancing_zones)
        self_trans_dic = ZoningSystem.trans_df_to_dict(
            self_trans,
            self_zoning.column_name,
            balancing_zones.column_name,
            self_zoning.translation_column_name(balancing_zones),
        )
        if self_zoning == other_zoning:
            other_trans_dic = self_trans_dic
        else:
            other_trans = other_zoning.translate(balancing_zones)
            other_trans_dic = ZoningSystem.trans_df_to_dict(
                other_trans,
                other_zoning.column_name,
                balancing_zones.column_name,
                other_zoning.translation_column_name(balancing_zones),
            )
        self_agg = self_data.rename(columns=self_trans_dic).groupby(level=0, axis=1).sum()
        other_agg = other_data.rename(columns=other_trans_dic).groupby(level=0, axis=1).sum()
        agg_factors = other_agg / self_agg
        factors = ctk.translation.pandas_vector_zone_translation(
            agg_factors,
            self_trans,
            balancing_zones.column_name,
            self_zoning.column_name,
            self_zoning.translation_column_name(balancing_zones),
            check_totals=False,
        )
        return factors

    def balance_by_segments(
        self,
        other: DVector,
        balancing_zones: ZoningSystem | BalancingZones | None = None,
    ):
        """
        Balance one DVector to another.

        This means that in the end the DVectors will match at some level of detail.

        Parameters
        ----------
        other: DVector
            The DVector to balance self to. Must have the same segmentation.

        balancing_zones: ZoningSystem | BalancingZones = None
            The zoning to perform balancing at. If None, rows will be balanced as
            a whole, conserving the spatial distribution of self, and only scaling up
            or down rows to match other. The more detailed the zoning system provided
            is, the closer self's spatial distribution will be matched to other's.
        """
        if balancing_zones is None:
            # Zone agnostic, just making sure DVectors matched along common segments
            factor = other.data.sum(axis=1) / self.data.sum(axis=1)
            balanced = self.data * factor
        elif isinstance(balancing_zones, ZoningSystem):
            factors = self._balance_zones_internal(
                self.data, self.zoning_system, other.data, other.zoning_system, balancing_zones
            )
            balanced = self.data * factors
        elif isinstance(balancing_zones, BalancingZones):
            if balancing_zones._segment_values is not None:
                if len(balancing_zones._segment_values.keys()) > 1:
                    # TODO implement this
                    raise ValueError(
                        "This method is not currently implemented for "
                        "balancing zones with individual values defined "
                        "for multiple segments."
                    )
                seg = list(balancing_zones._segment_values.keys())[0]
                vals = balancing_zones._segment_values[seg]
                zone = balancing_zones._segment_zoning[seg]
                self_slice = self.filter_segment_value(seg, vals)
                self_remaining = self.drop_by_segment_values(seg, vals)
                other_slice = other.filter_segment_value(seg, vals)
                other_remaining = other.drop_by_segment_values(seg, vals)
                slice_factors = self._balance_zones_internal(
                    self_slice, self.zoning_system, other_slice, other.zoning_system, zone
                )
                remaining_factors = self._balance_zones_internal(
                    self_remaining,
                    self.zoning_system,
                    other_remaining,
                    other.zoning_system,
                    balancing_zones._default_zoning,
                )
                balanced = pd.concat(
                    [self_slice * slice_factors, self_remaining * remaining_factors]
                )
            else:
                balanced = self.data.copy()
                for zon, segs in balancing_zones.zoning_groups():
                    grouped_self = self.data.groupby(level=segs).sum()
                    grouped_other = other.data.groupby(level=segs).sum()
                    factors = self._balance_zones_internal(
                        grouped_self,
                        self.zoning_system,
                        grouped_other,
                        other.zoning_system,
                        zon,
                    )
                    balanced *= factors
                remaining_segs = set(self.segmentation.names) - set(
                    balancing_zones._segment_zoning.keys()
                )
                grouped_self = self.data.groupby(level=list(remaining_segs)).sum()
                grouped_other = other.data.groupby(level=list(remaining_segs)).sum()
                factors = self._balance_zones_internal(
                    grouped_self,
                    self.zoning_system,
                    grouped_other,
                    other.zoning_system,
                    balancing_zones._default_zoning,
                )
                balanced *= factors
                # Factors have been applied multiple times at different segmentation levels, so need to balance once more over whole rows to make totals match
                factor = other.data.sum(axis=1) / balanced.sum(axis=1)
                balanced *= factor
        else:
            raise ValueError(
                "balancing_zones must be either BalancingZones, ZoningSystem, or None"
                f"type provided: {type(balancing_zones)}"
            )
        return DVector(
            import_data=balanced,
            segmentation=self.segmentation,
            zoning_system=self.zoning_system,
            time_format=self.time_format,
            cut_read=self._cut_read,
        )


@dataclass
class IpfTarget:
    """
    Dataclass to store targets to pass to IPF method of DVector.

    Parameters
    ----------
    data: DVector
        DVector containing the target data.
    zoning_diff: bool | None = None
        Whether target zoning is different to seed. This needn't be set by the user
        as it is determined internally.
    zone_translation: pd.DataFrame | None = None
        A translation between the seed and target zoning if necessary. This must
        be a strict aggregation from seed to target zoning. If left blank the
        IPF method will attempt to find a translation and use that using DVector
        translation methods.
    segment_translations: dict[str, str] | None = None
        A dict defining corresponding segments in seed for segments in target not
        in seed. These will be used to find lookups in the seg_translations folder.
        As with zoning, these must be strict aggregations from seed to target segment.
    """

    data: DVector
    zoning_diff: bool | None = None
    zone_translation: pd.DataFrame | None = None
    segment_translations: dict[str, str] | None = (
        None  # keys are segment in target, values, segment in seed
    )

    @staticmethod
    def check_compatibility(targets: Collection[IpfTarget], adjust: bool = False):
        """
        Check compatibility between ipf targets, and optionally adjust to match.

        This will check rmse between targets and return a rmse value for the set.
        If adjust is set to True then all targets will be set to match the last
        target at whatever level of zoning/segmentation they match. If two targets
        don't match at any level of zoning/segmentation this method will not
        attempt to convert to check, it will just ignore those.

        Parameters
        ----------
        targets: Collection[IpfTarget]
            The targets to check.
        adjust: bool = False
            Whether to change the targets or just report on their compatibility.
        """
        targ_dict = {i: j for i, j in enumerate(targets)}
        rmses = {}
        for pos in list(itertools.combinations(reversed(targ_dict), 2)):
            target_1, target_2 = targ_dict[pos[1]].data, targ_dict[pos[0]].data
            if target_1.zoning_system != target_2.zoning_system:
                continue
            common_segs = target_1.segmentation.overlap(target_2.segmentation)
            if len(common_segs) == 0:
                agg_1 = target_1.data.sum()
                agg_2 = target_2.data.sum()
            else:
                agg_1 = target_1.aggregate(list(common_segs))
                agg_2 = target_2.aggregate(list(common_segs))
            diff = (agg_1 - agg_2) ** 2
            rmse = (diff.sum() / len(diff)) ** 0.5
            rmses[tuple(common_segs)] = rmse
            if adjust:
                adj = agg_2 / agg_1
                adj.fill(np.inf, 0)
                target_1 *= adj
                target_1 *= target_2.sum() / target_1.sum()
                targ_dict[pos[1]].data = target_1
        targets = list(targ_dict.values())
        return pd.DataFrame.from_dict(rmses, orient="index"), targets
