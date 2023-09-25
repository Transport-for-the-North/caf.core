# -*- coding: utf-8 -*-
"""
Created on: Tues May 25th 15:04:32 2021
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:
Holds the ZoningSystem Class which stores all information on different zoning
systems
"""
# Allow class self hinting
from __future__ import annotations

# Builtins
import os
import logging
import warnings
import itertools
import configparser

from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Optional

# Third Party
import numpy as np
import pandas as pd
import caf.toolkit as ctk

# Local Imports


LOG = logging.getlogger(__name__)


class ZoningSystem:
    """Zoning definitions to provide common interface

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

    _zoning_system_import_fname = "zoning_systems"
    _base_col_name = "%s_zone_id"

    # File names
    _valid_ftypes = [".csv", ".pbz2", ".csv.bz2", ".bz2"]
    _zones_csv_fname = "zones.csv"
    _internal_zones_fname = "internal_zones.csv"
    _external_zones_fname = "external_zones.csv"

    # Df col names
    _df_name_col = "zone_name"
    _df_desc_col = "zone_desc"

    _zoning_definitions_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "definitions",
        "zoning_systems",
    )

    _translation_dir = os.path.join(_zoning_definitions_path, "_translations")

    _translate_infill = 0
    _translate_base_zone_col = "%s_zone_id"
    _translate_base_trans_col = "%s_to_%s"

    _default_weighting_suffix = "correspondence"
    _weighting_suffix = {
        "population": "population_weight",
        "employment": "employment_weight",
        "no_weight": "no_weighting",
        "average": "weighted_average",
    }

    possible_weightings = list(_weighting_suffix.keys()) + [None]

    def __init__(
        self,
        name: str,
        unique_zones: np.ndarray,
        metadata: Union[ZoningSystemMetaData, PathLike],
        zone_descriptions: Optional[np.ndarray] = None,
        internal_zones: Optional[np.ndarray] = None,
        external_zones: Optional[np.ndarray] = None,
    ):
        """Builds a ZoningSystem

        This class should almost never be constructed directly. If an
        instance of ZoningSystem is needed, the helper function
        `get_zoning_system()` should be used instead.

        Parameters
        ----------
        name:
            The name of the zoning system to create.

        unique_zones:
            A numpy array of unique zone names for this zoning system.

        internal_zones:
            A numpy array of unique zone names that make up the "internal"
            area of this zoning system. Every value in this array must also
            be contained in unique_zones.

        external_zones:
            A numpy array of unique zone names that make up the "external"
            area of this zoning system. Every value in this array must also
            be contained in unique_zones.
        """
        # Init
        self._name = name
        self._col_name = self._base_col_name % name
        self._unique_zones = unique_zones
        self._n_zones = len(self.unique_zones)
        if isinstance(metadata, PathLike):
            self._metadata = ZoningSystemMetaData.load_yaml(metadata)
        else:
            self._metadata = metadata

        # Validate and assign the optional arguments
        self._internal_zones = internal_zones
        self._external_zones = external_zones
        self._zone_descriptions = zone_descriptions

        if zone_descriptions is not None:
            if zone_descriptions.shape != unique_zones.shape:
                raise ValueError(
                    "zone_names is not the same shape as unique_zones. "
                    f"Expected shape of {unique_zones.shape}, got shape of "
                    f"{zone_descriptions.shape}"
                )

            # Order the zone names the same as the unique zones
            name_dict = dict(zip(unique_zones, zone_descriptions))
            self._zone_descriptions = np.array([name_dict[x] for x in self._unique_zones])

    @property
    def name(self) -> str:
        """The name of the zoning system"""
        return self._name

    @property
    def col_name(self) -> str:
        """The default name to give a column containing the zone data"""
        return self._col_name

    @property
    def unique_zones(self) -> pd.DataFrame:
        """A numpy array of the unique zones in order"""
        return self._unique_zones

    @property
    def zone_descriptions(self) -> np.ndarray:
        """A numpy array of the unique zone names in order"""
        if self._zone_descriptions is None:
            raise ValueError(
                f"No definition for zone descriptions has been set for this "
                f"zoning system. Name: {self.name}"
            )
        return self._zone_descriptions

    @property
    def zone_to_description_dict(self) -> Dict[Any, Any]:
        """A Dictionary of zones to their names"""
        return dict(zip(self._unique_zones, self.zone_descriptions))

    @property
    def n_zones(self) -> int:
        """The number of zones in this zoning system"""
        return self._n_zones

    @property
    def internal_zones(self) -> np.ndarray:
        """A numpy array of the internal zones in order"""
        if self._internal_zones is None:
            raise ValueError(
                f"No definition for internal zones has been set for this "
                f"zoning system. Name: {self.name}"
            )
        return self._internal_zones

    @property
    def external_zones(self) -> np.ndarray:
        """A numpy array of the external zones in order"""
        if self._external_zones is None:
            raise ValueError(
                f"No definition for external zones has been set for this "
                f"zoning system. Name: {self.name}"
            )
        return self._external_zones

    def __copy__(self):
        """Returns a copy of this class"""
        return self.copy()

    def __eq__(self, other) -> bool:
        """Overrides the default implementation"""
        # May need to update in future, but assume they are equal if names match
        if not isinstance(other, ZoningSystem):
            return False

        # Make sure names, unique zones, and n_zones are all the same
        if self.name != other.name:
            return False

        if set(self.unique_zones) != set(other.unique_zones):
            return False

        if self.n_zones != other.n_zones:
            return False

        return True

    def __ne__(self, other) -> bool:
        """Overrides the default implementation"""
        return not self.__eq__(other)

    def __len__(self) -> int:
        """Get the length of the zoning system"""
        return self._n_zones

    def _get_weighting_suffix(self, weighting: str = None) -> str:
        """
        Takes a weighting name and converts it into a file suffix
        """
        if weighting is None:
            return self._default_weighting_suffix
        return self._weighting_suffix[weighting]

    def _get_translation_definition(
        self,
        other: ZoningSystem,
        weighting: str = "spatial",
        trans_cache: Path = Path(r"I:\Data\Zone Translations\cache"),
    ) -> pd.DataFrame:
        """
        Returns a space generate zone translation between self and other.
        """
        # Init
        home_dir = trans_cache
        names = sorted([self.name, other.name])
        folder = f"{names[0]}_{names[1]}"
        file = f"{names[0]}_to_{names[1]}_{weighting}"

        # Try find a translation
        if (home_dir / folder).is_dir():
            try:
                trans = pd.read_csv(home_dir / folder / file, index_col=[0, 1])
            except FileNotFoundError as error:
                raise FileNotFoundError(
                    "A translation for this weighting has not been found, but the folder "
                    "exists so there is probably a translation with a different weighting. "
                    f"Files in folder are : {os.listdir(home_dir / folder)}. Please choose"
                    f" one of these or generate your own translation using caf.space."
                ) from error
        elif (self._metadata is not None) | (other._metadata is not None):
            LOG.warning(
                "A translation for these zones does not exist. Trying to generate a "
                "translation using caf.space. This will be spatial regardless of the "
                "input weighting. For a different weighting make your own."
            )
            try:
                import caf.space as cs
            except ModuleNotFoundError:
                raise ImportError(
                    "caf.space is not installed in this environment. A translation"
                    " cannot be found or generated."
                )
            zone_1 = cs.TransZoneSystemInfo(
                name=self.name,
                shapefile=self._metadata.shapefile_path,
                id_col=self._metadata.shapefile_id_col,
            )
            zone_2 = cs.TransZoneSystemInfo(
                name=other.name,
                shapefile=other._metadata.shapefile_path,
                id_col=other._metadata.shapefile_id_col,
            )
            conf = cs.ZoningTranslationInputs(zone_1=zone_1, zone_2=zone_2)
            trans = cs.ZoneTranslation(conf).spatial_translation()
        else:
            raise NotImplementedError("")
        return trans

    def _check_translation_zones(
        self,
        other: ZoningSystem,
        translation: pd.DataFrame,
        self_col: str,
        other_col: str,
    ) -> None:
        """Check if any zones are missing from the translation DataFrame."""
        translation_name = f"{self.name} to {other.name}"
        for zone_system, column in ((self, self_col), (other, other_col)):
            missing = ~np.isin(zone_system.unique_zones, translation[column])
            if np.sum(missing) > 0:
                LOG.warning(
                    "%s %s zones missing from translation %s",
                    np.sum(missing),
                    zone_system.name,
                    translation_name,
                )

    def copy(self):
        """Returns a copy of this class"""
        return ZoningSystem(
            name=self.name,
            unique_zones=self._unique_zones.copy(),
            internal_zones=self._internal_zones.copy(),
            external_zones=self._external_zones.copy(),
            zone_descriptions=self._zone_descriptions.copy(),
            metadata=self._metadata.copy(),
        )

    def translate(
        self,
        other: ZoningSystem,
        weighting: str = None,
    ) -> np.ndarray:
        """
        Returns a numpy array defining the translation of self to other

        Parameters
        ----------
        other:
            The zoning system to translate this zoning system into

        weighting:
            The weighting to use when building the translation. Must be None,
            or one of ZoningSystem.possible_weightings

        Returns
        -------
        translations_array:
            A numpy array defining the weights to use for the translation.
            The rows correspond to self.unique_zones
            The columns correspond to other.unique_zones

        Raises
        ------
        ZoningError:
            If a translation definition between self and other cannot be found
        """
        # Validate input
        if not isinstance(other, ZoningSystem):
            raise ValueError(
                f"other is not the correct type. Expected ZoningSystem, got " f"{type(other)}"
            )

        # Get a numpy array to define the translation
        translation_df = self._get_translation_definition(other, weighting)
        translation = ctk.pandas_utils.long_to_wide_infill(
            df=translation_df,
            index_col=self._translate_base_zone_col % self.name,
            columns_col=self._translate_base_zone_col % other.name,
            values_col=self._translate_base_trans_col % (self.name, other.name),
            index_vals=self.unique_zones,
            column_vals=other.unique_zones,
            infill=self._translate_infill,
        )

        return translation.values

    def save(self, path: PathLike, mode: str):
        """
        Save zoning data as a dataframe and a yml file.

        The dataframe will be saved to either a csv or a DVector Hdf file. If
        hdf, the key is 'zoning']. The dataframe will contain a minimum of
        'zone_id' and 'zone_name', with optional extra columns depending on
        whether they exist in the saved object. The yml will contain the zoning
        metadata, which at a minimum contains the zone name.
        """
        out_path = Path(path)
        save_df = self.unique_zones
        if self._internal_zones is not None:
            save_df["internal"] = save_df["zone_id"].isin(self.internal_zones["zone_id"])
        if self._external_zones is not None:
            save_df["external"] = save_df["zone_id"].isin(self.external_zones["zone_id"])
        if self._zone_descriptions is not None:
            save_df["descriptions"] = self.zone_descriptions
        if mode.lower() == "hdf":
            with pd.HDFStore(out_path / "DVector.h5") as store:
                store["zoning"] = save_df
        elif mode.lower() == "csv":
            save_df.to_csv(out_path / "zoning.csv")
        else:
            raise ValueError("Mode can only be 'hdf' or 'csv', not " f"{mode}.")
        self._metadata.save_yaml(out_path / "zoning_meta.yml")

    @classmethod
    def load(cls, in_path: PathLike, mode: str):
        """Creates a ZoningSystem instance from path_or_instance_dict

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
        if ~os.path.isfile(in_path / "metadata.yml"):
            return None
        zoning_meta = ZoningSystemMetaData.load_yaml(in_path / "zoning_meta.yml")
        if mode.lower() == "hdf":
            with pd.HDFStore(in_path / "DVector.hdf", "r") as store:
                zoning = store["zoning"]
        elif mode.lower() == "csv":
            zoning = pd.read_csv(in_path / "zoning.csv")
        else:
            raise ValueError("Mode can only be 'hdf' or 'csv', not " f"{mode}.")
        int_zones = None
        ext_zones = None
        descriptions = None
        if "internal" in zoning.columns:
            int_zones = zoning.loc[zoning["internal"], "zone_id"].values
        if "external" in zoning.columns:
            ext_zones = zoning.loc[zoning["external"], "zone_id"].values
        if "descriptions" in zoning.columns:
            descriptions = zoning["descriptions"].values

        return cls(
            name=zoning_meta.name,
            unique_zones=zoning["zones"].values,
            metadata=zoning_meta,
            internal_zones=int_zones,
            external_zones=ext_zones,
            descriptions=descriptions,
        )

    def get_metadata(self) -> ZoningSystemMetaData:
        """
        Gets metadata for a zoning system's shapefile.  At the moment this only consists of
        the name of the zone ID column and the path to where the shapefile is saved
        Returns:
            MetaData: The metadata for the zoning system
        """
        import_home = os.path.join(
            ZoningSystem._zoning_definitions_path, self.name, "metadata.yml"
        )
        metadata = ZoningSystemMetaData.load_yaml(import_home)
        return metadata

    @staticmethod
    def old_to_new_zoning(
        old_dir: PathLike, new_dir: PathLike, mode: str = "csv", return_zoning: bool = False
    ):
        """
        Converts zoning info stored in the old format to the new format.
        Optionally returns the zoning as well, but this is primarily designed
        for read in -> write out.

        Parameters
        ----------
        old_dir: Directory containing the zoning data in the old format (i.e.
        in normits_demand/core/definitions/zoning_systems
        new_dir: Directory for the reformatted zoning to be saved in. It will
        be saved in a sub-directory named for the zoning system.
        mode: Whether to save as a csv or HDF. Passed directly to save method
        return_zoning: Whether to return the zoning as well as saving.
        """
        old_dir = Path(old_dir)
        # read zones, expect at least zone_id and zone_name, possibly zone_desc too
        zones = pd.read_csv(old_dir / "zones.csv.bz2")
        if "zone_desc" in zones.columns:
            description = zones["zone_desc"]
        else:
            description = None
        try:
            metadata = ZoningSystemMetaData.load_yaml(old_dir / "metadata.yml")
            name = metadata.name
        except FileNotFoundError:
            name = os.path.split(old_dir)[-1]
            metadata = ZoningSystemMetaData(name=name)
        try:
            external = pd.read_csv(old_dir / "external_zones.csv.bz2")
        except FileNotFoundError:
            external = None
            warnings.warn("No external zoning info found.")
        try:
            internal = pd.read_csv(old_dir / "internal_zones.csv.bz2")
        except FileNotFoundError:
            internal = None
            warnings.warn("No internal zoning info found.")
        zoning = ZoningSystem(
            name=name,
            unique_zones=zones[["zone_id", "zone_name"]],
            metadata=metadata,
            zone_descriptions=description,
            internal_zones=internal,
            external_zones=external,
        )
        zoning.save(new_dir, mode=mode)
        if return_zoning:
            return zoning





class ZoningSystemMetaData(ctk.BaseConfig):
    """
    Class to store metadata relating to zoning systems in normits_demand
    """

    name: str
    shapefile_id_col: Optional[str]
    shapefile_path: Optional[Path]
