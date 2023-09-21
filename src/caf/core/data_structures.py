# -*- coding: utf-8 -*-
"""
Created on: 19/09/2023
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:

"""
from __future__ import annotations
# Built-Ins
import enum
import operator
from typing import Union, Optional, Any
import os
from os import PathLike
# Third Party
import pandas as pd
import numpy as np
import caf.toolkit as ctk
# Local Imports
from segmentation import Segmentation
from zoning import ZoningSystem
# pylint: disable=import-error,wrong-import-position
# Local imports here

# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #

# # # CLASSES # # #
@enum.unique
class TimeFormat(enum.Enum):
    AVG_WEEK = 'avg_week'
    AVG_DAY = 'avg_day'
    AVG_HOUR = 'avg_hour'

    @staticmethod
    def _valid_time_formats() -> list[str]:
        """
        Returns a list of valid strings to pass for time_format
        """
        return [x.value for x in TimeFormat]

    @staticmethod
    def get_time_periods() -> list[int]:
        return [1, 2, 3, 4, 5, 6]

    @staticmethod
    def conversion_order() -> list[TimeFormat]:
        return [TimeFormat.AVG_WEEK, TimeFormat.AVG_DAY, TimeFormat.AVG_HOUR]

    @staticmethod
    def _week_to_hour_factors() -> dict[int, float]:
        """Compound week to day and day to hour factors"""
        return ctk.toolbox.combine_dict_list(
            dict_list=[TimeFormat._week_to_day_factors(), TimeFormat._day_to_hour_factors()],
            operation=operator.mul,
        )

    @staticmethod
    def _hour_to_week_factors() -> dict[int, float]:
        """Compound hour to day and day to week factors"""
        return ctk.toolbox.combine_dict_list(
            dict_list=[TimeFormat._hour_to_day_factors(), TimeFormat._day_to_week_factors()],
            operation=operator.mul,
        )

    @staticmethod
    def _hour_to_day_factors() -> dict[int, float]:
        """Inverse of day to hour factors"""
        return {k: 1 / v for k, v in TimeFormat._day_to_hour_factors().items()}

    @staticmethod
    def _day_to_week_factors() -> dict[int, float]:
        """Inverse of week to day factors"""
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
            1: 1/3,
            2: 1/6,
            3: 1/3,
            4: 1/12,
            5: 1/24,
            6: 1/24,
        }

    @staticmethod
    def avg_hour_to_total_hour_factors() -> dict[int, float]:
        """Get a dictionary of conversion factors"""
        return TimeFormat._hour_to_day_factors()

    @staticmethod
    def total_hour_to_avg_hour_factors() -> dict[int, float]:
        """Get a dictionary of conversion factors"""
        return TimeFormat._day_to_hour_factors()

    @staticmethod
    def get(value: str) -> TimeFormat:
        """Get an instance of this with value

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
                "\tExpected one of: %s"
                % (value, TimeFormat._valid_time_formats())
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

    def get_conversion_factors(self,
                               to_time_format: TimeFormat,
                               ) -> dict[int, float]:
        """Get the conversion factors for each time period

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
            raise ValueError(
                "Cannot get the conversion factors when converting to self."
            )

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

class DVector:
    _chunk_size = 1000
    _val_col = 'val'
    def __init__(self,
                 segmentation: Segmentation,
                 import_data: Union[pd.DataFrame, PathLike],
                 zoning_system: Optional[ZoningSystem] = None,
                 time_format: Optional[Union[str, TimeFormat]] = None,
                 zone_col: Optional[str] = None,
                 val_col: Optional[str] = None,
                 df_naming_conversion: Optional[str] = None,
                 df_chunk_size: Optional[int] = None,
                 infill: Optional[Any] = 0,
                 process_count: Optional[int] = -2,
                 ) -> None:

        if zoning_system is not None:
            if not isinstance(zoning_system, ZoningSystem):
                raise ValueError(
                    "Given zoning_system is not a nd.core.ZoningSystem object."
                    "Got a %s object instead."
                    % type(zoning_system)
                )

        if not isinstance(segmentation, Segmentation):
            raise ValueError(
                "Given segmentation is not a nd.core.SegmentationLevel object."
                "Got a %s object instead."
                % type(segmentation)
            )

        self._zoning_system = zoning_system
        self._segmentation = segmentation
        self._time_format = self._validate_time_format(time_format)
        self._df_chunk_size = self._chunk_size if df_chunk_size is None else df_chunk_size

        # Define multiprocessing arguments
        self.process_count = process_count

        if self.process_count == 0:
            self._chunk_divider = 1
        else:
            self._chunk_divider = self.process_count * 3

        # Set defaults if args not set
        val_col = self._val_col if val_col is None else val_col
        if zone_col is None and zoning_system is not None:
            zone_col = zoning_system.col_name
        self.zone_col = zone_col

        # Try to convert the given data into DVector format
        if isinstance(import_data, pd.DataFrame):
            # validate segmentation of incoming data
            loaded_seg = Segmentation.load_segmentation(source=import_data, segs=segmentation.names, naming_order=segmentation.naming_order)

            self._data = pd.DataFrame(data=import_data.data, index=loaded_seg.ind, )
        elif isinstance(import_data, dict):
            self._data = self._old_to_new_dvec(
                import_data=import_data,
                infill=infill,
            )
        else:
            raise NotImplementedError(
                "Don't know how to deal with anything other than: "
                "pandas DF, or dict"
            )

    # SETTERS AND GETTERS
    @property
    def val_col(self):
        return self._val_col

    @property
    def zoning_system(self):
        return self._zoning_system

    @property
    def segmentation(self):
        return self._segmentation

    @property
    def process_count(self):
        return self._process_count

    @process_count.setter
    def process_count(self, a):
        if a < 0:
            self._process_count = os.cpu_count() + a
        else:
            self._process_count = a

    @property
    def time_format(self):
        if self._time_format is None:
            return None
        return self._time_format.name

    @staticmethod
    def _valid_time_formats() -> list[str]:
        """
        Returns a list of valid strings to pass for time_format
        """
        return [x.value for x in TimeFormat]
    def _validate_time_format(self,
                              time_format: Union[str, TimeFormat],
                              ) -> TimeFormat:
        """Validate the time format is a valid value

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
        if self.segmentation.has_time_period_segments():
            if time_format is None:
                raise ValueError(
                    "The given segmentation level has time periods in its "
                    "segmentation, but the format of this time period has "
                    "not been defined.\n"
                    "\tTime periods segment name: %s\n"
                    "\tValid time_format values: %s"
                    % (self.segmentation._time_period_segment_name,
                       self._valid_time_formats(),
                       )
                )

        # If None or TimeFormat, that's fine
        if time_format is None or isinstance(time_format, TimeFormat):
            return time_format

        # Check we've got a valid value
        time_format = time_format.strip().lower()
        if time_format not in self._valid_time_formats():
            raise ValueError(
                "The given time_format is not valid.\n"
                "\tGot: %s\n"
                "\tExpected one of: %s"
                % (time_format, self._valid_time_formats())
            )

        # Convert into a TimeFormat constant
        return_val = None
        for name, time_format_obj in TimeFormat.__members__.items():
            if name.lower() == time_format:
                return_val = time_format_obj
                break

        if return_val is None:
            raise ValueError(
                "We checked that the given time_format was valid, but it "
                "wasn't set when we tried to set it. This shouldn't be "
                "possible!"
            )

        return return_val
    def _dataframe_to_dvec(self,
                           df: pd.DataFrame,
                           zone_col: str,
                           val_col: str,
                           segment_naming_conversion: str,
                           infill: Any,
                           ):
        """
        Converts a pandas dataframe into dvec.data internal structure

        While converting, will:
        - Make sure that any missing segment/zone combinations are infilled
          with infill
        - Make sure only one value exist for each segment/zone combination
        """
        # Init columns depending on if we have zones
        required_cols = self.segmentation.naming_order + [self._val_col]
        sort_cols = [self._segment_col]

        # Add zoning if we need it
        if self.zoning_system is not None:
            required_cols += [self._zone_col]
            sort_cols += [self._zone_col]

        # ## VALIDATE AND CONVERT THE GIVEN DATAFRAME ## #
        # Rename import_data columns to internal names
        rename_dict = {zone_col: self._zone_col, val_col: self._val_col}
        df = df.rename(columns=rename_dict)

        # Rename the segment columns if needed
        if segment_naming_conversion is not None:
            df = self.segmentation.rename_segment_cols(df, segment_naming_conversion)
            # Set to None so the columns aren't renamed again in `create_segement_col`
            segment_naming_conversion = None

        # Make sure we don't have any extra columns
        extra_cols = set(list(df)) - set(required_cols)
        if len(extra_cols) > 0:
            raise ValueError(
                "Found extra columns in the given DataFrame than needed. The "
                "given DataFrame should only contain val_col, "
                "segmentation_cols, and the zone_col (where applicable).\n"
                "Expected: %s\n"
                "Found the following extra columns: %s"
                % (required_cols, extra_cols)
            )

        # Add the segment column - drop the individual cols
        df[self._segment_col] = self.segmentation.create_segment_col(
            df=df,
            naming_conversion=segment_naming_conversion
        )
        df = df.drop(columns=self.segmentation.naming_order)

        # Sort by the segment columns for MP speed
        df = df.sort_values(by=sort_cols)

        # ## MULTIPROCESSING SETUP ## #
        # If the dataframe is smaller than the chunk size, evenly split across cores
        if len(df) < self._df_chunk_size * self.process_count:
            chunk_size = math.ceil(len(df) / self.process_count)
        else:
            chunk_size = self._df_chunk_size

        # setup a pbar
        pbar_kwargs = {
            'desc': "Converting df to dvec",
            'unit': "segment",
            'disable': (not self._debugging_mp_code),
            'total': math.ceil(len(df) / chunk_size),
        }

        # ## MULTIPROCESS THE DATA CONVERSION ## #
        # Build a list of arguments
        kwarg_list = list()
        for df_chunk in pd_utils.chunk_df(df, chunk_size):
            kwarg_list.append({'df_chunk': df_chunk})

        # Call across multiple threads
        data_chunks = multiprocessing.multiprocess(
            fn=self._dataframe_to_dvec_internal,
            kwargs=kwarg_list,
            process_count=self.process_count,
            pbar_kwargs=pbar_kwargs,
        )
        data = du.sum_dict_list(data_chunks)

        # ## MAKE SURE DATA CONTAINS ALL SEGMENTS ##
        # find the segments which arent in there
        not_in = set(self.segmentation.segment_names) - data.keys()

        # Figure out what the default value should be
        if self.zoning_system is None:
            default_val = infill
        else:
            default_val = np.array([infill] * self.zoning_system.n_zones)

        # Infill the missing segments
        for name in not_in:
            data[name] = copy.copy(default_val)

        return data
    def translate_zoning(self,
                         new_zoning: ZoningSystem,
                         weighting: str = None,
                         ) -> DVector:
        """
        Translates this DVector into another zoning system and returns a new
        DVector.

        Parameters
        ----------
        new_zoning:
            The zoning system to translate into.

        weighting:
            The weighting to use when building the translation. Must be None,
            or one of ZoningSystem.possible_weightings

        Returns
        -------
        translated_dvector:
            This DVector translated into new_new_zoning zoning system

        """
        # Validate inputs
        if not isinstance(new_zoning, ZoningSystem):
            raise ValueError(
                "new_zoning is not the correct type. "
                "Expected ZoningSystem, got %s"
                % type(new_zoning)
            )

        if self.zoning_system.name is None:
            raise nd.NormitsDemandError(
                "Cannot translate the zoning system of a DVector that does "
                "not have a zoning system to begin with."
            )

        # If we're translating to the same thing, return a copy
        if self.zoning_system == new_zoning:
            return self.copy()

        # Get translation
        translation = self.zoning_system.translate(new_zoning, weighting)

        # ## MULTIPROCESS ## #
        # Define the chunk size
        total = len(self._data)
        chunk_size = math.ceil(total / self._chunk_divider)

        # Make sure the chunks aren't too small
        if chunk_size < self._translate_zoning_min_chunk_size:
            chunk_size = self._translate_zoning_min_chunk_size

        # Define the kwargs
        kwarg_list = list()
        for keys_chunk in du.chunk_list(self._data.keys(), chunk_size):
            # Calculate subsets of self.data to avoid locks between processes
            self_data_subset = {k: self._data[k] for k in keys_chunk}

            # Assign to a process
            kwarg_list.append({
                'self_data': self_data_subset,
                'translation': translation.copy(),
            })

        # Define pbar
        pbar_kwargs = {
            'desc': "Translating",
            'disable': not self._debugging_mp_code,
        }

        # Run across processes
        data_chunks = multiprocessing.multiprocess(
            fn=self._translate_zoning_internal,
            kwargs=kwarg_list,
            process_count=self.process_count,
            pbar_kwargs=pbar_kwargs,
        )

        # Combine all computation chunks into one
        dvec_data = dict.fromkeys(self._data.keys())
        for chunk in data_chunks:
            dvec_data.update(chunk)

        return DVector(
            zoning_system=new_zoning,
            segmentation=self.segmentation,
            time_format=self.time_format,
            import_data=dvec_data,
            process_count=self.process_count,
        )
# # # FUNCTIONS # # #
