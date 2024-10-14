# -*- coding: utf-8 -*-
"""Holds the ZoningSystem Class which stores all information on different zoning systems."""
# Allow class self hinting
from __future__ import annotations

import enum
import itertools
import logging
import os
import re
import warnings
from os import PathLike
from pathlib import Path
from typing import Any, Literal, Optional, Union

import caf.toolkit as ctk
import h5py
import numpy as np
import pandas as pd
from typing_extensions import Self

from caf.base.segmentation import Segmentation, SegmentationInput

pd.set_option("future.no_silent_downcasting", True)
LOG = logging.getLogger(__name__)

# This is temporary, and will be an environment variable
ZONE_CACHE_HOME = Path(r"I:\Data\Zoning Systems\core_zoning")
ZONE_TRANSLATION_CACHE = Path(r"I:\Data\Zone Translations\cache")


class TranslationWarning(RuntimeWarning):
    """Warning related to zone zone_translation."""


class TranslationError(Exception):
    """Error related to zone zone_translation."""


# TODO(MB) Can be switched to StrEnum when support from Python 3.10 isn't required
class TranslationWeighting(enum.Enum):
    """Available weightings for zone translations."""

    SPATIAL = "spatial"
    POPULATION = "population"
    EMPLOYMENT = "employment"
    NO_WEIGHT = "no_weight"
    AVERAGE = "average"
    POP = "pop"
    EMP = "emp"

    def get_suffix(self) -> str:
        """Get filename suffix for weighting."""
        lookup = {
            self.SPATIAL: "spatial",
            self.POPULATION: "population_weight",
            self.EMPLOYMENT: "employment_weight",
            self.NO_WEIGHT: "no_weighting",
            self.AVERAGE: "weighted_average",
            self.POP: "pop",
            self.EMP: "emp",
        }

        return lookup[self]  # type: ignore


# pylint: disable=too-many-public-methods
class ZoningSystem:
    """Zoning definitions to provide common interface.

    Attributes
    ----------
    name:
        The name of the zoning system. This will be the same as the name in
        the definitions folder

    col_name:
        The default naming that should be given to this zoning system if
        defined to a pandas.DataFrame

    unique_zones:
        A sorted numpy array of unique zone names for this zoning system.

    n_zones:
        The number of zones in this zoning system
    """

    _id_column = "zone_id"
    _name_column = "zone_name"
    _desc_column = "descriptions"

    def __init__(
        self,
        name: str,
        unique_zones: pd.DataFrame,
        metadata: Union[ZoningSystemMetaData, PathLike],
    ):
        """Build a ZoningSystem.

        This class should almost never be constructed directly. If an
        instance of ZoningSystem is needed, the classmethod
        `get_zoning_system()` should be used instead.

        Parameters
        ----------
        name:
            The name of the zoning system to create.

        unique_zones:
            A dataframe of unique zone IDs and names, descriptions and subset flags
            for this zoning system. Should contain at least one column with unique
            zone ID integers labelled 'zone_id'.
        """
        self.name = name
        self._zones, self._subset_columns = self._validate_unique_zones(unique_zones)
        self.n_zones = len(self._zones)

        if isinstance(metadata, PathLike):
            self.metadata = ZoningSystemMetaData.load_yaml(Path(metadata))
        else:
            self.metadata = metadata

    # pylint: disable=too-many-branches
    def _validate_unique_zones(
        self, zones: pd.DataFrame
    ) -> tuple[pd.DataFrame, tuple[str, ...]]:
        """Normalise column names and set index to ID column.

        Returns
        -------
        pd.DataFrame
            Validated and normalised zones DataFrame.
        list[str]
            Names of subset columns found.

        Raises
        ------
        ValueError
            If zone ID column is missing or the values aren't unique integers.
        ValueError
            If any subset columns found aren't (or can't be converted to)
            boolean values. The boolean conversion process is restricted to
            integers values 0, 1 or string values "TRUE", "FALSE".
        """
        zones = zones.copy()
        zones.columns = [normalise_column_name(i) for i in zones.columns]

        if self._id_column not in zones.columns:
            raise ValueError(
                f"mandatory ID column ({self._id_column}) missing from zones data"
            )
        # TODO: consider replacing with alternative checks that allow string IDs
        ### This chunk of code requires the zone names to be integers
        ### This has been commented out to allow LSOA (or other) zone codes to be used
        ### directly instead to avoid the added step of providing zone lookups with
        ### integer zone numbers for all zone systems
        # try:
        #     zones.loc[:, self._id_column] = zones[self._id_column].astype(int)
        # except ValueError as exc:
        #     raise ValueError(
        #         f"zone IDs should be integers not {zones[self._id_column].dtype}"
        #     ) from exc

        try:
            zones = zones.set_index(self._id_column, verify_integrity=True)
        except ValueError as exc:
            duplicates = zones[self._id_column].drop_duplicates(keep="first")
            raise ValueError(
                f"duplicate zone IDs: {', '.join(duplicates.astype(str))}"
            ) from exc

        # Zone names and description columns are optional but should contain strings
        optional_columns = (self._name_column, self._desc_column)
        for name in optional_columns:
            if name in zones.columns:
                zones.loc[:, name] = zones[name].astype(str)

        # Any other columns are assumed to be subset mask columns so should be boolean
        # Restrictive boolean conversion is used that expects "TRUE", "FALSE" strings
        # or 0, 1 integers
        subset_column = []
        non_bool_columns = []
        for name in zones.columns:
            if name in optional_columns:
                continue

            if zones[name].dtype.kind == "b":
                subset_column.append(name)
                continue

            column = zones[name]

            try:
                column = column.astype(int)
            except ValueError:
                pass  # Attempt to convert column to integer for checking

            if column.dtype.kind in ("i", "u"):
                # Only convert integers 0 and 1 to boolean values
                if column.min() < 0 or column.max() > 1:
                    non_bool_columns.append(name)
                else:
                    zones[name] = column.astype(bool)
                    subset_column.append(name)
                continue

            # Check if column contains strings "TRUE" and "FALSE"
            column = column.astype(str).str.strip().str.upper()
            if np.isin(column.unique(), ("TRUE", "FALSE")).all():
                zones[name] = column.replace({"TRUE": True, "FALSE": False}).astype(bool)
                subset_column.append(name)
                continue

            non_bool_columns.append(name)

        if len(non_bool_columns) > 0:
            raise ValueError(
                f"{len(non_bool_columns)} subset columns found "
                f"which don't contain boolean values: {non_bool_columns}"
            )

        return zones, tuple(subset_column)

    # pylint: enable=too-many-branches

    @property
    def zones_data(self) -> pd.DataFrame:
        """
        Return a copy of the zones DataFrame.

        This contains zone ID as the index and some optional columns for names,
        descriptions and subset flags.
        """
        return self._zones.copy()

    @property
    def zone_ids(self) -> np.ndarray:
        """Return a copy of the zone IDs array."""
        return self._zones.index.values.copy()

    @property
    def subset_columns(self) -> tuple[str, ...]:
        """Names of subset columns available."""
        return self._subset_columns

    @property
    def name_to_id(self) -> dict:
        """Return a lookup dict of zone name to zone id."""
        return self.zone_names().reset_index().set_index("zone_name").to_dict()["zone_id"]

    @property
    def id_to_name(self) -> dict:
        """Return a lookup dict of zone id to zone name."""
        return self.zone_names().to_dict()

    @property
    def id_to_internal(self) -> dict:
        """Produce lookup to convert id to internal."""
        return self.internal().to_dict()

    @property
    def id_to_external(self) -> dict:
        """Produce lookup to convert id to external."""
        return self.external().to_dict()

    @property
    def desc_to_id(self) -> dict:
        """Return a lookup dict of zone description to zone id."""
        return (
            self.zone_descriptions()
            .reset_index()
            .set_index("descriptions")
            .to_dict()["zone_id"]
        )

    @property
    def id_to_desc(self) -> dict:
        """Return a lookup dict of zone id to zone description."""
        return self.zone_descriptions().to_dict()

    def _get_column(self, column: str) -> pd.Series:
        """
        Get `column` from zones data.

         Normalises `column` name.

        Raises
        ------
        KeyError
            If `column` doesn't exist in zones data.
        """
        normal = normalise_column_name(column)
        if normal not in self._zones:
            raise KeyError(f"{column} not found in zones data")
        return self._zones[normal].copy()

    def zone_descriptions(self) -> pd.Series:
        """
        Describe zones, with the index as the zone ID.

        Raises
        ------
        KeyError
            If zone descriptions column doesn't exist.
        """
        return self._get_column(self._desc_column)

    def zone_names(self) -> pd.Series:
        """
        Name of zones, with the index as the zone ID.

        Raises
        ------
        KeyError
            If zone names column doesn't exist.
        """
        return self._get_column(self._name_column)

    def internal(self) -> pd.Series:
        """Getter for internal column."""
        return self._get_column("internal")

    def external(self) -> pd.Series:
        """Getter for external column."""
        return self._get_column("external")

    def _get_mask_column(self, name: str) -> pd.Series:
        """Get subset mask column from zones data, validate it contains bool values."""
        if name in (self._id_column, self._name_column, self._desc_column):
            raise ValueError(f"{name} column is not a subset mask column")

        mask = self._get_column(name)
        if mask.dtype.kind != "b":
            raise TypeError(
                f"found subset column ({name}) but it is the "
                f"wrong type ({mask.dtype}), should be boolean"
            )

        return mask

    def get_subset(self, name: str) -> np.ndarray:
        """
        Get subset of zone IDs based on subset column `name`.

        Raises
        ------
        KeyError
            If `name` column doesn't exist in zones data.
        ValueError
            If `name` isn't a subset column, i.e. is a
            name or description column.
        TypeError
            If a subset column is found but doesn't contain
            boolean values.
        """
        mask = self._get_mask_column(name)
        return mask[mask].index.values.copy()

    def get_inverse_subset(self, name: str) -> np.ndarray:
        """
        Get inverse of the `name` subset.

        See Also
        --------
        get_subset
        """
        mask = self._get_mask_column(name)
        return mask[~mask].index.values.copy()

    @property
    def column_name(self) -> str:
        """Expected name of columns in translations or DVectors."""
        return f"{self.name}_id".lower()

    def __copy__(self):
        """Return a copy of this class."""
        return self.copy()

    # TODO(MB) Define almost equals method which ignores optional columns and
    # just compares zone ID, zone name and zone description e.g. if 2 MSOA
    # zone systems were compared with different subsets of internal zones
    def __eq__(self, other) -> bool:
        """
        Override the default implementation.

        Note: internal zones dataframe must be identical to `other`
        for the zone systems to be considered equal.
        """
        if not isinstance(other, ZoningSystem):
            return False

        # Make sure names, unique zones, and n_zones are all the same
        if self.name != other.name:
            return False

        if self.n_zones != other.n_zones:
            return False
        # this sort_index is incompatible with pandas 2.0. At the moment
        # we need <2.0 as it is required by toolkit, but should be noted.
        sorted_self = self._zones.sort_index(axis=0, inplace=False).sort_index(
            axis=1, inplace=False
        )
        sorted_other = other._zones.sort_index(axis=0, inplace=False).sort_index(
            axis=1, inplace=False
        )
        if not sorted_self.equals(sorted_other):
            return False

        return True

    def __ne__(self, other) -> bool:
        """Override the default implementation."""
        return not self.__eq__(other)

    def __len__(self) -> int:
        """Get the length of the zoning system."""
        return self.n_zones

    def _generate_spatial_translation(
        self, other: ZoningSystem, cache_path: Path = ZONE_CACHE_HOME
    ) -> pd.DataFrame:
        """Generate spatial zone_translation using `caf.space`, if available."""
        try:
            # pylint: disable=import-outside-toplevel
            import caf.space as cs

            # pylint: enable=import-outside-toplevel
        except ModuleNotFoundError as exc:
            raise ImportError(
                "caf.space is not installed in this environment. "
                "A zone_translation cannot be generated."
            ) from exc

        zone_1 = cs.TransZoneSystemInfo(
            name=self.name,
            shapefile=self.metadata.shapefile_path,
            id_col=self.metadata.shapefile_id_col,
        )

        zone_2 = cs.TransZoneSystemInfo(
            name=other.name,
            shapefile=other.metadata.shapefile_path,
            id_col=other.metadata.shapefile_id_col,
        )
        conf = cs.ZoningTranslationInputs(zone_1=zone_1, zone_2=zone_2, cache_path=cache_path)
        trans = cs.ZoneTranslation(conf).spatial_translation()
        # TODO fix return type in caf.space
        trans[trans.columns[:2]] = trans[trans.columns[:2]].astype(str)

        return trans

    def _get_translation_definition(
        self,
        other: ZoningSystem,
        weighting: TranslationWeighting = TranslationWeighting.SPATIAL,
        trans_cache: Path = ZONE_TRANSLATION_CACHE,
    ) -> pd.DataFrame:
        """Return a zone zone_translation between self and other."""
        names = sorted([self.name, other.name])
        folder = f"{names[0]}_{names[1]}"

        if trans_cache is None:
            trans_cache = ZONE_TRANSLATION_CACHE
        else:
            trans_cache = Path(trans_cache)

        file = f"{names[0]}_to_{names[1]}_{weighting.get_suffix()}.csv"

        # Try find a zone_translation
        if (trans_cache / folder).is_dir():
            # The folder exists so there is almost definitely at least one zone_translation
            try:
                trans = pd.read_csv(trans_cache / folder / file)
                LOG.info(
                    "A zone_translation has been found in %s "
                    "and is being used. This has been done based on given zone "
                    "names so it is advised to double check the used zone_translation "
                    "to make sure it matches what you expect.",
                    trans_cache / folder / file,
                )
            except FileNotFoundError as error:
                # As there is probably a zone_translation one isn't generated by default
                raise TranslationError(
                    "A zone_translation for this weighting has not been found, but the folder "
                    "exists so there is probably a zone_translation with a different weighting. "
                    f"Files in folder are : {os.listdir(trans_cache / folder)}. Please choose"
                    f" one of these or generate your own zone_translation using caf.space."
                ) from error

        elif (self.metadata.shapefile_path is not None) & (
            other.metadata.shapefile_path is not None
        ):
            LOG.warning(
                "A zone_translation for these zones does not exist. Trying to generate a "
                "zone_translation using caf.space. This will be spatial regardless of the "
                "input weighting. For a different weighting make your own."
            )
            try:
                trans = self._generate_spatial_translation(other, cache_path=trans_cache)
            except ImportError as exc:
                raise TranslationError(
                    f"A zone_translation from {self.name} to {other.name}"
                    " cannot be found or generated."
                ) from exc

        else:
            raise TranslationError(
                f"A zone_translation between {self.name} and {other.name} "
                "does not exist and cannot be generated. To perform this "
                "zone_translation you must generate a zone_translation using "
                "caf.space."
            )

        trans = self.validate_translation_data(other, trans)
        return trans

    def translation_column_name(self, other: ZoningSystem) -> str:
        """
        Return expected name for zone_translation factors column in zone_translation data.

        Expected to be lowercase in the format "{self.name}_to_{other.name}".
        """
        return f"{self.name}_to_{other.name}".lower()

    def _replace_id(
        self, missing_rep, missing_id, translation, zone_system, translation_name, replacer
    ):
        if np.sum(missing_rep) > 0:
            if np.sum(missing_rep) >= np.sum(missing_id):
                warnings.warn(
                    f"{np.sum(missing_id)} {zone_system.name} zones "
                    f"missing from zone_translation {translation_name}",
                    TranslationWarning,
                )
            else:
                warnings.warn(
                    f"For {zone_system.name} zone name matches the zone_translation better than id, "
                    f"so that will be used. {np.sum(missing_rep)} missing for name, and "
                    f"{np.sum(missing_id)} missing for id."
                )
                translation[zone_system.column_name].replace(to_replace=replacer, inplace=True)
        else:
            translation[zone_system.column_name].replace(to_replace=replacer, inplace=True)

        return translation

    def validate_translation_data(
        self,
        other: ZoningSystem,
        translation: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Validate zone_translation data, checking for missing zones and factors sum to 1.

        Normalises column names (`normalise_column_name`) before checking
        if columns are present.

        Returns
        -------
        pd.DataFrame
            `zone_translation` data after column name normlisation.

        Raises
        ------
        TranslationError
            If a zone_translation definition between self and other cannot be
            found or generated, or there is an error in the zone_translation file.

        Warns
        -----
        TranslationWarning
            If the zone_translation doesn't contain all zones from either zone
            system.
        """
        translation_name = f"{self.name} to {other.name}"
        translation_column = self.translation_column_name(other)

        translation = translation.copy()
        translation.columns = [normalise_column_name(i) for i in translation.columns]

        # Check required columns are present
        missing = []
        for column in (self.column_name, other.column_name, translation_column):
            if column not in translation.columns:
                missing.append(column)

        if len(missing) > 0:
            raise TranslationError(
                f"required columns missing from zone zone_translation: {missing}"
            )

        # Warn if any zone IDs are missing
        for zone_system in (self, other):
            missing_internal_id: np.ndarray = ~np.isin(
                zone_system.zone_ids, translation[zone_system.column_name].values
            )

            if np.sum(missing_internal_id) > 0:
                try:
                    missing_internal_name: np.ndarray | float = ~np.isin(
                        zone_system.zone_names(), translation[zone_system.column_name].values
                    )
                except KeyError:
                    missing_internal_name = np.inf
                try:
                    missing_internal_desc: np.ndarray | float = ~np.isin(
                        zone_system.zone_descriptions(),
                        translation[zone_system.column_name].values,
                    )
                except KeyError:
                    missing_internal_desc = np.inf
                if np.sum(missing_internal_name) <= np.sum(missing_internal_desc):
                    translation = self._replace_id(
                        missing_internal_name,
                        missing_internal_id,
                        translation,
                        zone_system,
                        translation_name,
                        zone_system.name_to_id,
                    )
                else:
                    translation = self._replace_id(
                        missing_internal_desc,
                        missing_internal_id,
                        translation,
                        zone_system,
                        translation_name,
                        zone_system.desc_to_id,
                    )
                translation = translation[
                    translation[zone_system.column_name].isin(zone_system.zone_ids)
                ]

        # Warn if zone_translation factors don't sum to 1 from each from zone
        from_sum = translation.groupby(self.column_name)[translation_column].sum()
        if (from_sum != 1).any():
            max_ = np.max(np.abs(from_sum - 1))
            warnings.warn(
                f"{(from_sum != 1).sum()} {self.name} zones have splitting factors "
                f"which don't sum to 1 (value totals may change during zone_translation), "
                f"the maximum difference is {max_:.1e}",
                TranslationWarning,
            )

        return translation

    def copy(self):
        """Return a copy of this class."""
        return ZoningSystem(
            name=self.name,
            unique_zones=self._zones.copy().reset_index(),
            metadata=self.metadata.copy(),
        )

    def translate(
        self,
        other: ZoningSystem,
        cache_path: PathLike = ZONE_TRANSLATION_CACHE,
        weighting: TranslationWeighting | str = TranslationWeighting.SPATIAL,
    ) -> pd.DataFrame:
        """
        Find, or generates, the zone_translation data from `self` to `other`.

        Parameters
        ----------
        other : ZoningSystem
            The zoning system to translate this zoning system into

        weighting : TranslationWeighting | str, default TranslationWeighting.SPATIAL
            The weighting to use when building the zone_translation. Must be
            one of TranslationWeighting.

        Returns
        -------
        pd.DataFrame
            A numpy array defining the weights to use for the zone_translation.
            The rows correspond to self.unique_zones
            The columns correspond to other.unique_zones

        Raises
        ------
        TranslationError
            If a zone_translation definition between self and other cannot be
            found or generated, or there is an error in the zone_translation file.

        Warns
        -----
        TranslationWarning
            If the zone_translation doesn't contain all zones from either zone
            system.
        """
        if not isinstance(other, ZoningSystem):
            raise ValueError(
                f"other is not the correct type. Expected ZoningSystem, got " f"{type(other)}"
            )

        if isinstance(weighting, str):
            weighting = TranslationWeighting(weighting)

        translation_df = self._get_translation_definition(
            other, weighting, trans_cache=Path(cache_path)
        )

        return translation_df

    @staticmethod
    def trans_df_to_dict(trans_df, from_col, to_col, factor_col):
        """Convert a translation dataframe to a dict."""
        if not (trans_df[factor_col] == 1).all():
            raise TranslationError("This method only works for nested zoning systems.")
        return trans_df.set_index(from_col)[to_col].to_dict()

    def save(self, path: PathLike, mode: Literal["csv", "hdf"] = "csv"):
        """
        Save zoning data as a dataframe and a yml file.

        The dataframe will be saved to either a csv or a DVector Hdf file. If
        hdf, the key is 'zoning']. The dataframe will contain a minimum of
        'zone_id' and 'zone_name', with optional extra columns depending on
        whether they exist in the saved object. The yml will contain the zoning
        metadata, which at a minimum contains the zone name.
        """
        out_path = Path(path)
        save_df = self._zones.reset_index()
        if mode.lower() == "hdf":
            save_df.to_hdf(out_path, key="zoning", mode="a")
            with h5py.File(out_path, "a") as h_file:
                h_file.create_dataset(
                    "zoning_meta", data=self.metadata.to_yaml().encode("utf-8")
                )
        elif mode.lower() == "csv":
            out_path = out_path / self.name
            out_path.mkdir(exist_ok=True, parents=False)
            save_df.to_csv(out_path / "zoning.csv", index=False)
            self.metadata.save_yaml(out_path / "zoning_meta.yml")
        else:
            raise ValueError("Mode can only be 'hdf' or 'csv', not " f"{mode}.")

    @classmethod
    def load(cls, in_path: PathLike, mode: str):
        """
        Create a ZoningSystem instance from path_or_instance_dict.

        If path_or_instance_dict is a path, the file is loaded in and
        the instance_dict extracted.
        The instance_dict is then used to recreate the saved instance, using
        the class constructor.
        Use `save()` to save the data in the correct format.

        Parameters
        ----------
        path_or_instance_dict:
            Path to read the data in from.
        """

        # make sure in_path is a Path
        in_path = Path(in_path)
        # If this file exists the zoning should be in the hdf and vice versa
        if mode.lower() == "hdf":
            zoning = pd.read_hdf(in_path, key="zoning", mode="r")
            with h5py.File(in_path, "r") as h_file:
                # pylint: disable=no-member
                yam_load = h_file["zoning_meta"][()].decode("utf-8")
                zoning_meta = ZoningSystemMetaData.from_yaml(yam_load)
        elif mode.lower() == "csv":
            zoning = pd.read_csv(in_path / "zoning.csv")
            zoning_meta = ZoningSystemMetaData.load_yaml(in_path / "zoning_meta.yml")
        else:
            raise ValueError("Mode can only be 'hdf' or 'csv', not " f"{mode}.")

        return cls(name=zoning_meta.name, unique_zones=zoning, metadata=zoning_meta)

    @classmethod
    def old_to_new_zoning(
        cls,
        old_dir: PathLike,
        new_dir: PathLike = ZONE_CACHE_HOME,
        mode: Literal["csv", "hdf"] = "csv",
    ) -> ZoningSystem:
        """
        Convert zoning info stored in the old format to the new format.

        Optionally returns the zoning as well, but this is primarily designed
        for read in -> write out.

        Parameters
        ----------
        old_dir: Directory containing the zoning data in the old format (i.e.
        in normits_demand/base/definitions/zoning_systems
        new_dir: Directory for the reformatted zoning to be saved in. It will
        be saved in a sub-directory named for the zoning system.
        mode: Whether to save as a csv or HDF. Passed directly to save method
        """
        old_dir = Path(old_dir)
        name = old_dir.name
        # read zones, expect at least zone_id and zone_name, possibly zone_desc too
        zones = pd.read_csv(old_dir / "zones.csv.bz2")
        zones.columns = [normalise_column_name(i) for i in zones.columns]

        zones = zones.rename(columns={"zone_desc": cls._desc_column})

        if cls._id_column not in zones.columns and cls._name_column in zones.columns:
            zones.loc[:, cls._id_column] = zones[cls._name_column].astype(int)

        # It might be more appropriate to check if files exist explicitly
        try:
            metadata = ZoningSystemMetaData.load_yaml(old_dir / "metadata.yml")
            metadata.name = name
        except FileNotFoundError:
            metadata = ZoningSystemMetaData(name=name)

        for file in old_dir.glob("*_zones.csv*"):
            subset = pd.read_csv(file)[cls._id_column].astype(int)

            match = re.match(r"(.*)_zones", file.stem, re.I)
            assert match is not None, "impossible for match to be None"

            column = normalise_column_name(match.group(1))
            zones.loc[:, column] = zones[cls._id_column].isin(subset)

        zoning = ZoningSystem(name=name, unique_zones=zones, metadata=metadata)

        zoning.save(new_dir, mode=mode)

        return zoning

    @classmethod
    def get_zoning(cls, name: str, search_dir: PathLike = ZONE_CACHE_HOME):
        """Call load method to return zoning info based on a name."""
        zone_dir = Path(search_dir) / name
        if zone_dir.is_dir():
            try:
                return cls.load(zone_dir, "csv")
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    "There is a directory for this zone_name, but "
                    "the required files are not there. There "
                    "should be two files called 'zoning.csv' and "
                    f"'metadata.yml' in the folder {zone_dir}."
                ) from exc
        raise FileNotFoundError(f"{zone_dir} does not exist. Please recheck inputs.")

    @classmethod
    def zoning_from_shapefile(
        cls,
        name: str,
        shapefile: PathLike,
        name_col: str,
        tfn_bound: PathLike = Path(
            r"Y:\Data Strategy\GIS Shapefiles\TfN Boundary\Transport_for_the_north_boundary_2020_generalised.shp"
        ),
    ) -> Self:
        """
        Produce a ZoningSystem from a shapefile.

        Parameters
        ----------
        name: str
            The name of the zone system.
        shapefile: PathLike
            A path to the shapefile to generate the zoning from.
        name_col:
            The column in the shapefile containing the zone 'names'.
        tfn_bound_path:
            A path to a polygon shapefile used to determine internal/external in
            resulting ZoneSystem

        Returns
        -------
        ZoneSystem
        """
        # pylint: disable=import-outside-toplevel
        try:
            import geopandas as gpd
        except ImportError as exc:
            raise ImportError("Geopandas must be installed to use this method.") from exc
        # pylint: enable=import-outside-toplevel
        gdf = gpd.read_file(shapefile)[[name_col, "geometry"]]
        tfn_bound = gpd.read_file(tfn_bound)
        inner = gdf.sjoin(tfn_bound, predicate="within")
        gdf["internal"] = False
        gdf.loc[inner.index, "internal"] = True
        gdf["external"] = ~gdf["internal"]
        gdf.index += 1
        gdf.index.name = "zone_id"
        gdf.rename(columns={name_col: "zone_name"}, inplace=True)
        gdf.reset_index(inplace=True)
        zoning = gdf[["zone_id", "zone_name", "internal", "external"]]
        meta = ZoningSystemMetaData(name=name, shapefile_path=Path(shapefile))
        return cls(name=name, unique_zones=zoning, metadata=meta)


# pylint: enable=too-many-public-methods


class ZoningSystemMetaData(ctk.BaseConfig):
    """Class to store metadata relating to zoning systems in normits_demand."""

    name: Optional[str]
    shapefile_id_col: Optional[str] = None
    shapefile_path: Optional[Path] = None
    extra_columns: Optional[list[str]] = None


class BalancingZones:
    """
    Stores the zoning systems for the attraction model balancing.

    Allows a different zone system to be defined for each segment
    and a default zone system. An instance of this class can be
    iterated through to give the groups of segments defined for
    each unique zone system.

    Parameters
    ----------
    segmentation : Segmentation
        Segmentation level of the attractions being balanced.
    default_zoning : ZoningSystem
        Default zoning system to use for any segments which aren't
        given in `segment_zoning`.
    segment_zoning : Dict[str, ZoningSystem]
        Dictionary containing the name of the segment (key) and
        the zoning system for that segment (value).
    """

    def __init__(
        self,
        segmentation: Segmentation,
        default_zoning: ZoningSystem,
        segment_zoning: dict[str, ZoningSystem],
        segment_values: dict[str, list[int]] | None = None,
    ):

        # Validate inputs
        if not isinstance(segmentation, Segmentation):
            raise ValueError(f"segmentation should be Segmentation not {type(segmentation)}")

        if not isinstance(default_zoning, ZoningSystem):
            raise ValueError(
                f"default_zoning should be ZoningSystem not {type(default_zoning)}"
            )

        # Assign attributes
        self._segmentation = segmentation
        self._default_zoning = default_zoning
        self._segment_zoning = segment_zoning
        self._segment_values = segment_values
        self._unique_zoning: dict[Any, Any] | None = None

    def get_zoning(self, segment_name: str) -> ZoningSystem:
        """Return `ZoningSystem` for given `segment_name`.

        Parameters
        ----------
        segment_name : str
            Name of the segment to return, if a zone system isn't
            defined for this name then the default is used.

        Returns
        -------
        ZoningSystem
            Zone system for given segment, or default.
        """
        if segment_name not in self._segment_zoning:
            return self._default_zoning
        return self._segment_zoning[segment_name]

    @property
    def unique_zoning(self) -> dict[str, ZoningSystem]:
        """Dictionary containing a lookup of segments to zoning systems."""
        if self._unique_zoning is None:
            self._unique_zoning = dict()
            for zoning in self._segment_zoning.values():
                if zoning.name not in self._unique_zoning:
                    self._unique_zoning[zoning.name] = zoning
            self._unique_zoning[self._default_zoning.name] = self._default_zoning
        return self._unique_zoning

    def zoning_groups(self):
        """Iterate through the unique zoning systems and provides list of segments.

        Yields
        ------
        ZoningSystem
            Zone system for this group of segments.
        List[str]
            List of segment names which use this zone system.
        """

        def zone_name(s):
            return self.get_zoning(s).name

        zone_ls = sorted(self._segmentation.names, key=zone_name)
        for zone_name, segments in itertools.groupby(zone_ls, key=zone_name):
            zoning = self.unique_zoning[zone_name]
            yield zoning, list(segments)

    def __iter__(self) -> tuple[ZoningSystem, list[str]]:
        """See `BalancingZones.zoning_groups`."""
        return self.zoning_groups()

    class BalancingConfClass(ctk.BaseConfig):
        """Cong class for balancing."""

        seg_conf: SegmentationInput
        zon_conf: ZoningSystemMetaData
        seg_zon: dict[str, ZoningSystemMetaData]

    def save(self, path: Path) -> None:
        """Save balancing zones to output file.

        Output file is saved to a yaml file

        Parameters
        ----------
        path : Path
            Path to output file to save.
        """
        seg_zon_dict = dict()
        for name, zoning in self._segment_zoning.items():
            seg_zon_dict[name] = zoning.metadata

        out_conf = self.BalancingConfClass(
            seg_conf=self._segmentation.input,
            zon_conf=self._default_zoning.metadata,
            seg_zon=seg_zon_dict,
        )
        out_conf.save_yaml(path)

    @classmethod
    def load(cls, path: Path) -> BalancingZones:
        """Load balancing zones from config file.

        Parameters
        ----------
        path : Path
            Path to config file, should be the format defined
            by `configparser` with section names defined in
            `BalancingZones.OUTPUT_FILE_SECTIONS`.

        Returns
        -------
        BalancingZones
            Balancing zones with loaded parameters.
        """
        conf = cls.BalancingConfClass.load_yaml(path)
        segmentation = Segmentation(conf.seg_conf)
        default_zoning = ZoningSystem.get_zoning(conf.zon_conf.name)
        segment_zoning = {}
        for name, _ in conf.seg_zon:
            segment_zoning[name] = ZoningSystem.get_zoning(name)
        return cls(segmentation, default_zoning, segment_zoning)

    # @staticmethod
    # def build_single_segment_group(
    #     segmentation: Segmentation,
    #     default_zoning: ZoningSystem,
    #     segment_column: str,
    #     segment_zones: dict[Any, ZoningSystem],
    # ) -> BalancingZones:
    #     """Build `BalancingZones` for a single segment group.
    #
    #     Defines different zone systems for all unique values
    #     in a single segment column.
    #
    #     Parameters
    #     ----------
    #     segmentation : nd.SegmentationLevel
    #         Segmentation to use for the balancing.
    #     default_zoning : ZoningSystem
    #         Default zone system for any undefined segments.
    #     segment_column : str
    #         Name of the segment column which will have
    #         different zone system for each unique value.
    #     segment_zones : Dict[Any, ZoningSystem]
    #         The unique segment values for `segment_column` and
    #         their corresponding zone system. Any values not
    #         include will use `default_zoning`.
    #
    #     Returns
    #     -------
    #     BalancingZones
    #         Instance of class with different zone systems for
    #         each segment corresponding to the `segment_zones`
    #         given.
    #
    #     Raises
    #     ------
    #     ValueError
    #         - If `segmentation` is not an instance of `SegmentationLevel`.
    #         - If `group_name` is not the name of a `segmentation` column.
    #         - If any keys in `segment_zones` aren't found in the `group_name`
    #           segmentation column.
    #
    #     Examples
    #     --------
    #     The example below will create an instance for `hb_p_m` attraction balancing with
    #     the zone system `lad_2020` for all segments with mode 1 and `msoa` for all with mode 2.
    #     >>> hb_p_m_balancing = BalancingZones.build_single_segment_group(
    #     >>>     Segmentation(SegmentationInput(enum_segments=['p','m'],
    #     >>>                                    naming_order=['p','m'])),
    #     >>>     ZoningSystem.get_zoning("gor"),
    #     >>>     "m",
    #     >>>     {1: ZoningSystem.get_zoning("lad_2020"), 2: ZoningSystem.get_zoning("msoa")},
    #     >>> )
    #     """
    #     if not isinstance(segmentation, Segmentation):
    #         raise ValueError(
    #             f"segmentation should be SegmentationLevel not {type(segmentation)}"
    #         )
    #     if segment_column not in segmentation.naming_order:
    #         raise ValueError(
    #             f"group_name should be one of {segmentation.naming_order}"
    #             f" for, not {segment_column}"
    #         )
    #     # Check all segment values refer to a possible value for that column
    #     unique_params = set(segmentation.seg_dict[segment_column])
    #     missing = [i for i in segment_zones if i not in unique_params]
    #     if missing:
    #         raise ValueError(
    #             "segment values not present in segment " f"column {segment_column}: {missing}"
    #         )
    #     segment_zoning = {}
    #     for value in segmentation.seg_dict[segment_column]:
    #         if value in segment_zones.keys():
    #             name = f"{segment_column}_{value}"
    #             segment_zoning[name] = segment_zones[value]
    #     return BalancingZones(segmentation, default_zoning, segment_zoning)


def normalise_column_name(column: str) -> str:
    """Convert column to lowercase and replace spaces with underscores."""
    column = column.lower().strip()
    return re.sub(r"\s+", "_", column)
