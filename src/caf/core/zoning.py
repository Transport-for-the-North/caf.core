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


LOG = logging.get_logger(__name__)


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
    _valid_ftypes = ['.csv', '.pbz2', '.csv.bz2', '.bz2']
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

    _translation_dir = os.path.join(
        _zoning_definitions_path,
        '_translations'
    )

    _translate_infill = 0
    _translate_base_zone_col = "%s_zone_id"
    _translate_base_trans_col = "%s_to_%s"

    _default_weighting_suffix = 'correspondence'
    _weighting_suffix = {
        'population': 'population_weight',
        'employment': 'employment_weight',
        'no_weight': 'no_weighting',
        'average': 'weighted_average',
    }

    possible_weightings = list(_weighting_suffix.keys()) + [None]

    def __init__(self,
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
        self._unique_zones = np.sort(unique_zones)
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
    def unique_zones(self) -> np.ndarray:
        """A numpy array of the unique zones in order"""
        return self._unique_zones

    @property
    def zone_descriptions(self) -> np.ndarray:
        """A numpy array of the unique zone names in order"""
        if self._zone_descriptions is None:
            raise ZoningError(
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
            raise ZoningError(
                f"No definition for internal zones has been set for this "
                f"zoning system. Name: {self.name}"
            )
        return self._internal_zones

    @property
    def external_zones(self) -> np.ndarray:
        """A numpy array of the external zones in order"""
        if self._external_zones is None:
            raise ZoningError(
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
        return len(self.unique_zones)

    def _get_weighting_suffix(self, weighting: str = None) -> str:
        """
        Takes a weighting name and converts it into a file suffix
        """
        if weighting is None:
            return self._default_weighting_suffix
        return self._weighting_suffix[weighting]

    def _get_translation_definition(self,
                                    other: ZoningSystem,
                                    weighting: str = "spatial",
                                    trans_cache: Path = Path(r"I:\Data\Zone Translations\cache")
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
                raise FileNotFoundError("A translation for this weighting has not been found, but the folder "
                            "exists so there is probably a translation with a different weighting. "
                            f"Files in folder are : {os.listdir(home_dir / folder)}. Please choose"
                            f" one of these or generate your own translation using caf.space.") from error
        elif (self._metadata is not None) | (other._metadata is not None):
            LOG.warning("A translation for these zones does not exist. Trying to generate a "
                        "translation using caf.space. This will be spatial regardless of the "
                        "input weighting. For a different weighting make your own.")
            try:
                import caf.space as cs
            except ModuleNotFoundError:
                raise ImportError("caf.space is not installed in this environment. A translation"
                                  " cannot be found or generated.")
            zone_1 = cs.TransZoneSystemInfo(name=self.name, shapefile=self._metadata.shapefile_path, id_col=self._metadata.shapefile_id_col)
            zone_2 = cs.TransZoneSystemInfo(name=other.name, shapefile=other._metadata.shapefile_path, id_col=other._metadata.shapefile_id_col)
            conf = cs.ZoningTranslationInputs(zone_1=zone_1, zone_2=zone_2)
            trans = cs.ZoneTranslation(conf).spatial_translation()
        else:
            raise NotImplementedError("")
        return trans

    def _check_translation_zones(self,
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
            unique_zones=self.unique_zones.copy(),
        )

    def translate(self,
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
                f"other is not the correct type. Expected ZoningSystem, got "
                f"{type(other)}"
            )

        if weighting not in self.possible_weightings:
            raise ValueError(
                f"{weighting} is not a valid weighting for a translation. "
                f"Expected one of: {self.possible_weightings}"
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
            infill=self._translate_infill
        )

        return translation.values

    def save(self, path: PathLike, mode: str):
        """

        """
        out_path = Path(path)
        save_df = pd.DataFrame(data=self.unique_zones, columns=['zones'])
        if self._internal_zones is not None:
            internal = pd.Series(self.internal_zones)
            save_df['internal'] = save_df['zones'].isin(internal)
        if self._external_zones is not None:
            external = pd.Series(self.external_zones)
            save_df['external'] = save_df['zones'].isin(external)
        if self._zone_descriptions is not None:
            save_df['descriptions'] = self.zone_descriptions
        #TODO decide whether to save to the same hdf file as data with a different key
        #TODO or to a separate csv in the same folder
        if mode.lower() == 'hdf':
            with pd.HDFStore(out_path / 'DVector.h5') as store:
                store['zoning'] = save_df
        elif mode.lower() == 'csv':
            save_df.to_csv(out_path / 'zoning.csv')
        else:
            raise ValueError("Mode can only be 'hdf' or 'csv', not "
                             f"{mode}.")
        self._metadata.save_yaml(out_path / 'zoning_meta.yml')



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

        #make sure in_path is a Path
        in_path = Path(in_path)
        # If this file exists the zoning should be in the hdf and vice versa
        if ~os.path.isfile(in_path / "metadata.yml"):
            return None
        zoning_meta = ZoningSystemMetaData.load_yaml(in_path / "zoning_meta.yml")
        if mode.lower() == 'hdf':
            with pd.HDFStore(in_path / "DVector.hdf", 'r') as store:
                zoning = store['zoning']
        elif mode.lower() == 'csv':
            zoning = pd.read_csv(in_path / "zoning.csv")
        else:
            raise ValueError("Mode can only be 'hdf' or 'csv', not "
                             f"{mode}.")
        int_zones = None
        ext_zones = None
        descriptions = None
        if 'internal' in zoning.columns:
            int_zones = zoning.loc[zoning['internal'], 'zones'].values
        if 'external' in zoning.columns:
            ext_zones = zoning.loc[zoning['external'], 'zones'].values
        if 'descriptions' in zoning.columns:
            descriptions = zoning['descriptions'].values

        return cls(name=zoning_meta.name,
                   unique_zones=zoning['zones'].values,
                   metadata=zoning_meta,
                   internal_zones=int_zones,
                   external_zones=ext_zones,
                   descriptions=descriptions
                   )

    def get_metadata(self) -> ZoningSystemMetaData:
        """
        Gets metadata for a zoning system's shapefile.  At the moment this only consists of
        the name of the zone ID column and the path to where the shapefile is saved
        Returns:
            MetaData: The metadata for the zoning system
        """
        import_home = os.path.join(ZoningSystem._zoning_definitions_path, self.name,
                                   'metadata.yml')
        metadata = ZoningSystemMetaData.load_yaml(import_home)
        return metadata


class BalancingZones:
    """Stores the zoning systems for the attraction model balancing.

    Allows a different zone system to be defined for each segment
    and a default zone system. An instance of this class can be
    iterated through to give the groups of segments defined for
    each unique zone system.

    Parameters
    ----------
    segmentation : SegmentationLevel
        Segmentation level of the attractions being balanced.
    default_zoning : ZoningSystem
        Default zoning system to use for any segments which aren't
        given in `segment_zoning`.
    segment_zoning : Dict[str, ZoningSystem]
        Dictionary containing the name of the segment (key) and
        the zoning system for that segment (value).

    Raises
    ------
    ValueError
        If `segmentation` isn't an instance of `SegmentationLevel`.
        If `default_zoning` isn't an instance of `ZoningSystem`.
    """

    OUTPUT_FILE_SECTIONS = {
        "main": "BALANCING ZONES PARAMETERS",
        "zone_groups": "BALANCING ZONES GROUPS",
    }

    def __init__(
            self,
            segmentation: nd.SegmentationLevel,
            default_zoning: ZoningSystem,
            segment_zoning: Dict[str, ZoningSystem]
    ) -> None:
        # Initialise the class logger
        self._logger = nd.get_logger(f"{self.__module__}.{self.__class__.__name__}")

        # Validate inputs
        if not isinstance(segmentation, nd.SegmentationLevel):
            raise ValueError(f"segmentation should be SegmentationLevel not {type(segmentation)}")

        if not isinstance(default_zoning, ZoningSystem):
            raise ValueError(f"default_zoning should be ZoningSystem not {type(default_zoning)}")

        # Assign attributes
        self._segmentation = segmentation
        self._default_zoning = default_zoning
        self._segment_zoning = self._check_segments(segment_zoning)
        self._unique_zoning = None

    def _check_segments(
            self, segment_zoning: Dict[str, ZoningSystem],
    ) -> Dict[str, ZoningSystem]:
        """Check `segment_zoning` types and return dictionary of segments.

        Only adds value to dictionary if it is a segment name from
        `self._segmentation` and it has a `ZoningSystem` defined.

        Parameters
        ----------
        segment_zoning : Dict[str, ZoningSystem]
            Dictionary containing the name of the segment (key)
            and the zoning system for that segment (value).

        Returns
        -------
        Dict[str, ZoningSystem]
            Dictionary containing segment names and the defined `ZoningSystem`,
            does not include any segment names which aren't defined or which
            aren't present in `self._segmentation.segment_names`.
        """
        segments = {}
        for nm, zoning in segment_zoning.items():
            if nm not in self._segmentation.segment_names:
                self._logger.warning(
                    "%r not a segment in %s segmentation, ignoring",
                    nm,
                    self._segmentation.name,
                )
                continue
            if not isinstance(zoning, ZoningSystem):
                self._logger.error(
                    "%s segment zoning is %s not ZoningSystem, "
                    "using default zoning instead",
                    nm,
                    type(zoning),
                )
                continue
            segments[nm] = zoning
        defaults = [s for s in self._segmentation.segment_names if s not in segments]
        if defaults:
            self._logger.info(
                "default zoning (%s) used for %s segments",
                self._default_zoning.name,
                len(defaults),
            )
        return segments

    @property
    def segmentation(self) -> nd.SegmentationLevel:
        """nd.SegmentationLevel: Segmentation level of balancing zones."""
        return self._segmentation

    @property
    def unique_zoning(self) -> Dict[str, ZoningSystem]:
        """Dict[str, ZoningSystem]: Dictionary containing a lookup of all
            the unique `ZoningSystem` provided for the different segments.
            The keys are the zone system name and values are the
            `ZoningSystem` objects.
        """
        if self._unique_zoning is None:
            self._unique_zoning = dict()
            for zoning in self._segment_zoning.values():
                if zoning.name not in self._unique_zoning:
                    self._unique_zoning[zoning.name] = zoning
            self._unique_zoning[self._default_zoning.name] = self._default_zoning
        return self._unique_zoning

    def get_zoning(self, segment_name: str) -> ZoningSystem:
        """Return `ZoningSystem` for given `segment_name`

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

    def zoning_groups(self) -> Tuple[ZoningSystem, List[str]]:
        """Iterates through the unique zoning systems and provides list of segments.

        Yields
        ------
        ZoningSystem
            Zone system for this group of segments.
        List[str]
            List of segment names which use this zone system.
        """
        zone_name = lambda s: self.get_zoning(s).name
        zone_ls = sorted(self._segmentation.segment_names, key=zone_name)
        for zone_name, segments in itertools.groupby(zone_ls, key=zone_name):
            zoning = self.unique_zoning[zone_name]
            yield zoning, list(segments)

    def __iter__(self) -> Tuple[ZoningSystem, List[str]]:
        """See `BalancingZones.zoning_groups`."""
        return self.zoning_groups()

    def save(self, path: Path) -> None:
        """Saves balancing zones to output file.

        Output file is saved in format defined by
        `configparser`.

        Parameters
        ----------
        path : Path
            Path to output file to save.
        """
        config = configparser.ConfigParser()
        config[self.OUTPUT_FILE_SECTIONS["main"]] = {
            "segmentation": self.segmentation.name,
            "default_zoning": self._default_zoning.name,
        }
        config[self.OUTPUT_FILE_SECTIONS["zone_groups"]] = {
            zs.name: ", ".join(segs) for zs, segs in self.zoning_groups()
        }
        with open(path, "wt") as f:
            config.write(f)
        self._logger.info("Saved balancing zones to: %s", path)

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
        config = configparser.ConfigParser()
        config.read(path)
        params = {}
        params["segmentation"] = nd.get_segmentation_level(
            config.get(cls.OUTPUT_FILE_SECTIONS["main"], "segmentation")
        )
        default_zoning = config.get(cls.OUTPUT_FILE_SECTIONS["main"], "default_zoning")
        params["default_zoning"] = nd.get_zoning_system(default_zoning)
        params["segment_zoning"] = {}
        for zone, segments in config[cls.OUTPUT_FILE_SECTIONS["zone_groups"]].items():
            if zone == default_zoning:
                # Don't bother to define segments which use default zoning
                continue
            params["segment_zoning"].update(
                dict.fromkeys(
                    (s.strip() for s in segments.split(",")),
                    nd.get_zoning_system(zone),
                )
            )
        balancing_zones = BalancingZones(**params)
        balancing_zones._logger.info("Loaded balancing zones from: %s", path)
        return balancing_zones

    @staticmethod
    def build_single_segment_group(
            segmentation: nd.SegmentationLevel,
            default_zoning: ZoningSystem,
            segment_column: str,
            segment_zones: Dict[Any, ZoningSystem]
    ) -> BalancingZones:
        """Build `BalancingZones` for a single segment group.

        Defines different zone systems for all unique values
        in a single segment column.

        Parameters
        ----------
        segmentation : nd.SegmentationLevel
            Segmentation to use for the balancing.
        default_zoning : ZoningSystem
            Default zone system for any undefined segments.
        segment_column : str
            Name of the segment column which will have
            different zone system for each unique value.
        segment_zones : Dict[Any, ZoningSystem]
            The unique segment values for `segment_column` and
            their corresponding zone system. Any values not
            include will use `default_zoning`.

        Returns
        -------
        BalancingZones
            Instance of class with different zone systems for
            each segment corresponding to the `segment_zones`
            given.

        Raises
        ------
        ValueError
            - If `segmentation` is not an instance of `SegmentationLevel`.
            - If `group_name` is not the name of a `segmentation` column.
            - If any keys in `segment_zones` aren't found in the `group_name`
              segmentation column.

        Examples
        --------
        The example below will create an instance for `hb_p_m` attraction balancing with
        the zone system `lad_2020` for all segments with mode 1 and `msoa` for all with mode 2.
        >>> hb_p_m_balancing = AttractionBalancingZones.build_single_segment_group(
        >>>     nd.get_segmentation_level('hb_p_m'),
        >>>     nd.get_zoning_system("gor"),
        >>>     "m",
        >>>     {1: nd.get_zoning_system("lad_2020"), 2: nd.get_zoning_system("msoa")},
        >>> )
        """
        if not isinstance(segmentation, nd.SegmentationLevel):
            raise ValueError(
                f"segmentation should be SegmentationLevel not {type(segmentation)}"
            )
        if segment_column not in segmentation.naming_order:
            raise ValueError(
                f"group_name should be one of {segmentation.naming_order}"
                f" for {segmentation.name} not {segment_column}"
            )
        # Check all segment values refer to a possible value for that column
        unique_params = set(segmentation.segments[segment_column])
        missing = [i for i in segment_zones if i not in unique_params]
        if missing:
            raise ValueError(
                "segment values not present in segment "
                f"column {segment_column}: {missing}"
            )
        segment_zoning = {}
        for segment_params in segmentation:
            value = segment_params[segment_column]
            if value in segment_zones:
                name = segmentation.get_segment_name(segment_params)
                segment_zoning[name] = segment_zones[value]
        return BalancingZones(segmentation, default_zoning, segment_zoning)

    def __eq__(self, other: BalancingZones) -> bool:
        if not isinstance(other, self.__class__):
            return False
        if other.segmentation != self.segmentation:
            return False
        if other._default_zoning != self._default_zoning:
            return False
        for seg_name in self.segmentation.segment_names:
            if other.get_zoning(seg_name) != self.get_zoning(seg_name):
                return False
        return True


# ## FUNCTIONS ##
def _get_zones(name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Finds and reads in the unique zone data for zoning system with name
    """
    # Init
    name_col = ZoningSystem._df_name_col
    desc_col = ZoningSystem._df_desc_col

    # ## DETERMINE THE IMPORT LOCATION ## #
    import_home = os.path.join(ZoningSystem._zoning_definitions_path, name)

    # Make sure the import location exists
    if not os.path.exists(import_home):
        raise nd.NormitsDemandError(
            f"We don't seem to have any data for the zoning system {name}.\n"
            f"Tried looking for the data here: {import_home}"
        )

    # ## READ IN THE UNIQUE ZONES ## #
    file_path = os.path.join(import_home, ZoningSystem._zones_csv_fname)

    # Determine which path to use
    if not file_ops.similar_file_exists(file_path):
        raise nd.NormitsDemandError(
            f"We don't seem to have any zone data for the zoning "
            f"system {name}.\n"
            f"Tried looking for the data here: {file_path}\n"
        )

    # Read in the file
    df = file_ops.read_df(file_path, find_similar=True)
    if name_col not in df:
        raise ZoningError(
            f"Cannot get zoning system with name {name}. The definition file "
            f"was found, but no column named {name_col} exists."
        )

    # Keep just the relevant columns
    if desc_col not in df:
        df[desc_col] = df[name_col].copy()
    df = pd_utils.reindex_cols(df, columns=[name_col, desc_col])

    # Extract the columns and sort
    unsorted_zone_names = df[name_col].values
    unsorted_zone_descs = df[desc_col].fillna("").values
    name_to_desc = dict(zip(unsorted_zone_names, unsorted_zone_descs))

    zone_names = np.sort(df[name_col].values)
    zone_descs = np.array([name_to_desc[x] for x in zone_names])

    # ## READ IN THE INTERNAL AND EXTERNAL ZONES ## #
    internal_zones = None
    external_zones = None

    # Read in the internal zones
    try:
        file_path = os.path.join(import_home, ZoningSystem._internal_zones_fname)
        file_path = file_ops.find_filename(file_path, alt_types=ZoningSystem._valid_ftypes)
        if os.path.isfile(file_path):
            df = file_ops.read_df(file_path)
            internal_zones = np.sort(df[name_col].values)
    except FileNotFoundError:
        warn_msg = (
            f"No internal zones definition found for zoning system '{name}'"
        )
        warnings.warn(warn_msg, UserWarning, stacklevel=3)

    # Read in the external zones
    try:
        file_path = os.path.join(import_home, ZoningSystem._external_zones_fname)
        file_path = file_ops.find_filename(file_path, alt_types=ZoningSystem._valid_ftypes)
        if os.path.isfile(file_path):
            df = file_ops.read_df(file_path)
            external_zones = np.sort(df[name_col].values)
    except FileNotFoundError:
        warn_msg = (
            f"No external zones definition found for zoning system '{name}'"
        )
        warnings.warn(warn_msg, UserWarning, stacklevel=3)

    return zone_names, zone_descs, internal_zones, external_zones


def get_zoning_system(name: str) -> ZoningSystem:
    """
    Creates a ZoningSystem for zoning with name.

    Parameters
    ----------
    name:
        The name of the zoning system to get a ZoningSystem object for.

    Returns
    -------
    zoning_system:
        A ZoningSystem object for zoning system with name
    """
    # TODO(BT): Add some validation on the zone name
    # TODO(BT): Add some caching to this function!
    # Look for zone definitions
    zone_names, zone_desc, internal, external = _get_zones(name)

    # Create the ZoningSystem object and return
    return ZoningSystem(
        name=name,
        unique_zones=zone_names,
        zone_descriptions=zone_desc,
        internal_zones=internal,
        external_zones=external,
    )


class ZoningSystemMetaData(ctk.BaseConfig):
    """
    Class to store metadata relating to zoning systems in normits_demand
    """
    name: str
    shapefile_id_col: Optional[str]
    shapefile_path: Optional[Path]