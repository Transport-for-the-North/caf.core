# -*- coding: utf-8 -*-
"""
Module containing the data structures used in the CAF package.

Currently this is only the DVector class, but this may be expanded in the future.
"""
from __future__ import annotations

import enum
import logging
import operator
import warnings
from os import PathLike
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import caf.toolkit as ctk


# pylint: disable=no-name-in-module,import-error
from caf.core.segmentation import Segmentation, SegmentationWarning
from caf.core.zoning import ZoningSystem, TranslationWeighting

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
        """
        if zoning_system is not None:
            if not isinstance(zoning_system, ZoningSystem):
                raise ValueError(
                    "Given zoning_system is not a caf.core.ZoningSystem object."
                    f"Got a {type(zoning_system)} object instead."
                )

        if not isinstance(segmentation, Segmentation):
            raise ValueError(
                "Given segmentation is not a caf.core.SegmentationLevel object."
                f"Got a {type(segmentation)} object instead."
            )

        self._zoning_system = zoning_system
        self._segmentation = segmentation
        self._time_format = None
        if time_format is not None:
            self._time_format = self._validate_time_format(time_format)

        # Set defaults if args not set
        self._val_col = val_col

        # Try to convert the given data into DVector format
        if isinstance(import_data, (pd.DataFrame, pd.Series)):
            self._data = self._dataframe_to_dvec(import_data)
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
    def zoning_system(self):
        """_zoning_system getter."""
        return self._zoning_system

    @property
    def segmentation(self):
        """_segmentation getter."""
        return self._segmentation

    @property
    def data(self):
        """_data getter."""
        return self._data

    @property
    def time_format(self):
        """_time_format getter."""
        if self._time_format is None:
            return None
        return self._time_format.name

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
        Segmentation.validate_segmentation(source=import_data, segmentation=self.segmentation)

        if self.zoning_system is None:
            import_data.columns = [self.val_col]
            return import_data

        # Check columns are labelled with zone IDs
        try:
            import_data.columns = import_data.columns.astype(int)
        except ValueError as exc:
            raise TypeError(
                "DataFrame columns should be integers corresponding "
                f"to zone IDs not {import_data.columns.dtype}"
            ) from exc

        if set(import_data.columns) != set(self.zoning_system.zone_ids):
            missing = self.zoning_system.zone_ids[
                np.isin(self.zoning_system.zone_ids, import_data.columns)
            ]
            extra = import_data.columns.values[
                np.isin(import_data.columns.values, self.zoning_system.zone_ids)
            ]

            raise ValueError(
                f"{len(missing)} zone IDs from zoning system {self.zoning_system.name}"
                f" aren't found in the DVector data and {len(extra)} column names are"
                " found which don't correspond to zone IDs.\nDVector DataFrame column"
                " names should be the zone IDs (integers) for the given zone system."
            )

        return import_data

    def save(self, out_path: PathLike):
        """
        Save the DVector.

        DVector will be saved to a hdf file containing the DVector.

        Parameters
        ----------
        out_path: PathLike
            Path to the DVector, which should be an HDF file.

        Returns
        -------
        None
        """
        out_path = Path(out_path)

        self._data.to_hdf(out_path, key="data", mode="w", complevel=1)
        if self.zoning_system is not None:
            self.zoning_system.save(out_path, "hdf")
        self.segmentation.save(out_path, "hdf")

    @classmethod
    def load(cls, in_path: PathLike):
        """
        Load the DVector.

        Parameters
        ----------
        in_path: PathLike
            Path to where the DVector is saved. This should be a single hdf file.
        """
        in_path = Path(in_path)
        zoning = ZoningSystem.load(in_path, "hdf")
        segmentation = Segmentation.load(in_path, "hdf")
        data = pd.read_hdf(in_path, key="data", mode="r")

        return cls(segmentation=segmentation, import_data=data, zoning_system=zoning)

    def translate_zoning(
        self,
        new_zoning: ZoningSystem,
        cache_path: Optional[PathLike],
        weighting: str | TranslationWeighting = TranslationWeighting.SPATIAL,
    ) -> DVector:
        """
        Translate this DVector into another zoning system and returns a new DVector.

        Parameters
        ----------
        new_zoning: ZoningSystem
            The zoning system to translate into.

        cache_path: Optional[PathLike]
            Path to a cache containing zoning translations.

        weighting : str | TranslationWeighting = TranslationWeighting.SPATIAL
            The weighting to use when building the translation. Must be
            one of TranslationWeighting.

        Returns
        -------
        translated_dvector:
            This DVector translated into new_new_zoning zoning system

        Warns
        -----
        TranslationWarning
            If there are zone IDs missing from the translation or the
            translation factors don't sum to 1.
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
        translation = self.zoning_system.translate(
            new_zoning, weighting=weighting, cache_path=cache_path
        )

        transposed = self.data.transpose()
        transposed.index.names = [self.zoning_system.column_name]
        translated = ctk.translation.pandas_vector_zone_translation(
            transposed,
            translation,
            translation_from_col=self.zoning_system.column_name,
            translation_to_col=new_zoning.column_name,
            translation_factors_col=self.zoning_system.translation_column_name(new_zoning),
        )

        return DVector(
            zoning_system=new_zoning,
            segmentation=self.segmentation,
            time_format=self.time_format,
            import_data=translated.transpose(),
        )

    def copy(self):
        """Class copy method."""
        return DVector(
            segmentation=self._segmentation.copy(),
            zoning_system=self._zoning_system.copy(),
            import_data=self._data.copy(),
            time_format=self.time_format,
            val_col=self.val_col,
        )

    def overlap(self, other):
        """Call segmentation overlap method to check two DVectors contain overlapping segments."""
        overlap = self.segmentation.overlap(other.segmentation)
        if overlap == []:
            raise NotImplementedError(
                "There are no common segments between the "
                "two DVectors so this operation is not "
                "possible."
            )

    def _generic_dunder(
        self, other, df_method, series_method, escalate_warnings: bool = False
    ):
        """
        Stop telling me to use the imperative mood pydocstyle.

        A generic dunder method which is called by each of the dunder methods.
        """
        if escalate_warnings:
            warnings.filterwarnings("error", category=SegmentationWarning)
        # Make sure the two DVectors have overlapping indices
        self.overlap(other)
        # for the same zoning a simple * gives the desired result
        # This drops any nan values (intersecting index level but missing val)
        if self.zoning_system == other.zoning_system:
            if isinstance(self.data, pd.Series):
                prod = series_method(self.data, other.data)
            else:
                prod = df_method(self.data, other.data)
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
            prod = df_method(other.data, self.data.squeeze(), axis="index")
            zoning = other.zoning_system
        elif other.zoning_system is None:
            prod = df_method(self.data, other.data.squeeze(), axis="index")
            zoning = self.zoning_system
        # Different zonings raise an error rather than trying to translate
        else:
            raise NotImplementedError(
                "The two DVectors have different zonings. "
                "To multiply them, one must be translated "
                "to match the other."
            )
        # Index unchanged, aside from possible order. Segmentation remained the same
        if prod.index.equal_levels(self._data.index):
            return DVector(
                segmentation=self.segmentation, import_data=prod, zoning_system=zoning
            )
        # Index changed so the segmentation has changed. Segmentation should equal
        # the addition of the two segmentations (see __add__ method in segmentation)
        new_seg = self.segmentation + other.segmentation
        warnings.warn(
            f"This operation has changed the segmentation of the DVector"
            f"from {self.segmentation} to {new_seg}. This can happen"
            "but it can also be a sign of an error. Check the output DVector.",
            SegmentationWarning,
        )
        prod = prod.reorder_levels(new_seg.naming_order)
        return DVector(segmentation=new_seg, import_data=prod, zoning_system=zoning)

    def __mul__(self, other):
        """Multiply dunder method for DVector."""
        return self._generic_dunder(other, pd.DataFrame.mul, pd.Series.mul)

    def __add__(self, other):
        """Add dunder method for DVector."""
        return self._generic_dunder(other, pd.DataFrame.add, pd.Series.add)

    def __sub__(self, other):
        """Subtract dunder method for DVector."""
        return self._generic_dunder(other, pd.DataFrame.sub, pd.Series.sub)

    def __truediv__(self, other):
        """Division dunder method for DVector."""
        return self._generic_dunder(other, pd.DataFrame.div, pd.Series.div)

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

    def aggregate(self, segs: list[str]):
        """
        Aggregate DVector to new segmentation.

        New Segmentation must be a subset of the current segmentation. Currently
        this method is essentially a pandas 'groupby.sum()', but other methods
        could be called if needed (e.g. mean())

        Parameters
        ----------
        segs: Segments to aggregate to. Must be a subset of self.segmentation.naming_order,
        naming order will be preserved.
        """
        segmentation = self.segmentation.aggregate(segs)
        data = self.data.groupby(level=segs).sum()
        return DVector(
            segmentation=segmentation,
            import_data=data,
            zoning_system=self.zoning_system,
            time_format=self.time_format,
            val_col=self.val_col,
        )

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


# # # FUNCTIONS # # #
