# -*- coding: utf-8 -*-
"""Holds the ZoningSystem Class which stores all information on different zoning systems."""
# Allow class self hinting
from __future__ import annotations

import enum
import logging
import os
import re
from os import PathLike
from pathlib import Path
from typing import Literal, Optional, Union
import warnings

import h5py
import numpy as np
import pandas as pd
import caf.toolkit as ctk

LOG = logging.getLogger(__name__)

# This is temporary, and will be an environment variable
ZONE_CACHE_HOME = Path(r"I:\Data\Zoning Systems\core_zoning")
ZONE_TRANSLATION_CACHE = Path(r"I:\Data\Zone Translations\cache")


class TranslationWarning(RuntimeWarning):
    """Warning related to zone translation."""


class TranslationError(Exception):
    """Error related to zone translation."""


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
                zones[name] = column.replace({"TRUE": True, "FALSE": False})
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

    def get_column(self, column: str) -> pd.Series:
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
        return self.get_column(self._desc_column)

    def zone_names(self) -> pd.Series:
        """
        Name of zones, with the index as the zone ID.

        Raises
        ------
        KeyError
            If zone names column doesn't exist.
        """
        return self.get_column(self._name_column)

    def _get_mask_column(self, name: str) -> pd.Series:
        """Get subset mask column from zones data, validate it contains bool values."""
        if name in (self._id_column, self._name_column, self._desc_column):
            raise ValueError(f"{name} column is not a subset mask column")

        mask = self.get_column(name)
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
        self, other: ZoningSystem, cache_path: Path = ZONE_TRANSLATION_CACHE
    ) -> pd.DataFrame:
        """Generate spatial translation using `caf.space`, if available."""
        try:
            # pylint: disable=import-outside-toplevel
            import caf.space as cs

            # pylint: enable=import-outside-toplevel
        except ModuleNotFoundError as exc:
            raise ImportError(
                "caf.space is not installed in this environment. "
                "A translation cannot be generated."
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

        return cs.ZoneTranslation(conf).spatial_translation()

    def _get_translation_definition(
        self,
        other: ZoningSystem,
        weighting: TranslationWeighting = TranslationWeighting.SPATIAL,
        trans_cache: Path = ZONE_TRANSLATION_CACHE,
    ) -> pd.DataFrame:
        """Return a zone translation between self and other."""
        names = sorted([self.name, other.name])
        folder = f"{names[0]}_{names[1]}"

        if trans_cache is None:
            trans_cache = ZONE_TRANSLATION_CACHE
        else:
            trans_cache = Path(trans_cache)

        file = f"{names[0]}_to_{names[1]}_{weighting.get_suffix()}.csv"

        # Try find a translation
        if (trans_cache / folder).is_dir():
            # The folder exists so there is almost definitely at least one translation
            try:
                trans = pd.read_csv(trans_cache / folder / file)
                LOG.info(
                    "A translation has been found in %s "
                    "and is being used. This has been done based on given zone "
                    "names so it is advised to double check the used translation "
                    "to make sure it matches what you expect.",
                    trans_cache / folder / file,
                )
            except FileNotFoundError as error:
                # As there is probably a translation one isn't generated by default
                raise TranslationError(
                    "A translation for this weighting has not been found, but the folder "
                    "exists so there is probably a translation with a different weighting. "
                    f"Files in folder are : {os.listdir(trans_cache / folder)}. Please choose"
                    f" one of these or generate your own translation using caf.space."
                ) from error

        elif (self.metadata.shapefile_path is not None) & (
            other.metadata.shapefile_path is not None
        ):
            LOG.warning(
                "A translation for these zones does not exist. Trying to generate a "
                "translation using caf.space. This will be spatial regardless of the "
                "input weighting. For a different weighting make your own."
            )
            try:
                trans = self._generate_spatial_translation(other, trans_cache)
            except ImportError as exc:
                raise TranslationError(
                    f"A translation from {self.name} to {other.name}"
                    " cannot be found or generated."
                ) from exc

        else:
            raise TranslationError(
                f"A translation between {self.name} and {other.name} "
                "does not exist and cannot be generated. To perform this "
                "translation you must generate a translation using "
                "caf.space."
            )

        trans = self.validate_translation_data(other, trans)
        return trans

    def translation_column_name(self, other: ZoningSystem) -> str:
        """
        Return expected name for translation factors column in translation data.

        Expected to be lowercase in the format "{self.name}_to_{other.name}".
        """
        return f"{self.name}_to_{other.name}".lower()

    def validate_translation_data(
        self,
        other: ZoningSystem,
        translation: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Validate translation data, checking for missing zones and factors sum to 1.

        Normalises column names (`normalise_column_name`) before checking
        if columns are present.

        Returns
        -------
        pd.DataFrame
            `translation` data after column name normlisation.

        Raises
        ------
        TranslationError
            If a translation definition between self and other cannot be
            found or generated, or there is an error in the translation file.

        Warns
        -----
        TranslationWarning
            If the translation doesn't contain all zones from either zone
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
                f"required columns missing from zone translation: {missing}"
            )

        # Warn if any zone IDs are missing
        for zone_system in (self, other):
            missing_internal: np.ndarray = ~np.isin(
                zone_system.zone_ids, translation[zone_system.column_name].values
            )

            if np.sum(missing_internal) > 0:
                warnings.warn(
                    f"{np.sum(missing_internal)} {zone_system.name} zones "
                    f"missing from translation {translation_name}",
                    TranslationWarning,
                )

        # Warn if translation factors don't sum to 1 from each from zone
        from_sum = translation.groupby(self.column_name)[translation_column].sum()
        if (from_sum != 1).any():
            max_ = np.max(np.abs(from_sum - 1))
            warnings.warn(
                f"{(from_sum != 1).sum()} {self.name} zones have splitting factors "
                f"which don't sum to 1 (value totals may change during translation), "
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
        Find, or generates, the translation data from `self` to `other`.

        Parameters
        ----------
        other : ZoningSystem
            The zoning system to translate this zoning system into

        weighting : TranslationWeighting | str, default TranslationWeighting.SPATIAL
            The weighting to use when building the translation. Must be
            one of TranslationWeighting.

        Returns
        -------
        pd.DataFrame
            A numpy array defining the weights to use for the translation.
            The rows correspond to self.unique_zones
            The columns correspond to other.unique_zones

        Raises
        ------
        TranslationError
            If a translation definition between self and other cannot be
            found or generated, or there is an error in the translation file.

        Warns
        -----
        TranslationWarning
            If the translation doesn't contain all zones from either zone
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
        in normits_demand/core/definitions/zoning_systems
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


class ZoningSystemMetaData(ctk.BaseConfig):
    """Class to store metadata relating to zoning systems in normits_demand."""

    name: Optional[str]
    shapefile_id_col: Optional[str] = None
    shapefile_path: Optional[Path] = None
    extra_columns: Optional[list[str]] = None


def normalise_column_name(column: str) -> str:
    """Convert column to lowercase and replace spaces with underscores."""
    column = column.lower().strip()
    return re.sub(r"\s+", "_", column)
