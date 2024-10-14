# -*- coding: utf-8 -*-
"""Tests for the `ZoningSystem` class."""
import dataclasses
import string
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal, assert_series_equal

from caf.base.zoning import ZoningSystem, ZoningSystemMetaData


@dataclasses.dataclass
class ZoningData:
    """Zoning system dataset."""

    name: str
    data: pd.DataFrame
    subsets: dict[str, list[int]]


@pytest.fixture(name="zoning_data", scope="module")
def fix_zoning_data() -> ZoningData:
    """Define zoning input dataset."""
    ids = list(range(10))
    names = list(string.ascii_lowercase[:10])
    zones = pd.DataFrame(
        {
            "zone_id": ids,
            "zone_name": names,
            "descriptions": [f"{i}-{j}" for i, j in zip(ids, names)],
            "internal": [i < 5 for i in ids],
            "external": [i >= 5 for i in ids],
            "north": [i < 3 for i in ids],
        }
    )

    return ZoningData(
        name="test_zoning",
        data=zones,
        subsets={"internal": [0, 1, 2, 3, 4], "external": [5, 6, 7, 8, 9], "north": [0, 1, 2]},
    )


@pytest.fixture(name="old_zoning_dir")
def fix_old_zoning_dir(zoning_data: ZoningData, tmp_path: Path) -> Path:
    """Save zoning data in old format and return directory path."""
    zoning_dir = tmp_path / zoning_data.name
    zoning_dir.mkdir()

    zones = zoning_data.data[["zone_id", "zone_name", "descriptions"]].copy()
    zones = zones.rename(columns={"descriptions": "zone_desc"})
    zones.to_csv(zoning_dir / "zones.csv.bz2", index=False)

    for subset in zoning_data.subsets:
        data = zoning_data.data.loc[zoning_data.data[subset], "zone_id"].to_frame()
        data.to_csv(zoning_dir / f"{subset}_zones.csv.bz2", index=False)

    return zoning_dir


@pytest.fixture(name="id_only_zoning", scope="module")
def fix_id_only_zoning(zoning_data: ZoningData) -> tuple[ZoningData, ZoningSystem]:
    """Create `ZoningSystem` class containing only zone ID."""
    data = ZoningData(
        name=zoning_data.name + "-id_only",
        data=zoning_data.data["zone_id"].to_frame().copy(),
        subsets={},
    )
    system = ZoningSystem(
        name=data.name,
        unique_zones=data.data.copy(),
        metadata=ZoningSystemMetaData(name=data.name),
    )

    return data, system


@pytest.fixture(name="zoning_descriptions", scope="module")
def fix_zoning_descriptions(zoning_data: ZoningData) -> tuple[ZoningData, ZoningSystem]:
    """Create `ZoningSystem` class with optional name and description columns."""
    data = ZoningData(
        name=zoning_data.name + "-zoning_descriptions",
        data=zoning_data.data[["zone_id", "zone_name", "descriptions"]].copy(),
        subsets={},
    )
    system = ZoningSystem(
        name=data.name,
        unique_zones=data.data.copy(),
        metadata=ZoningSystemMetaData(name=data.name),
    )

    return data, system


@pytest.fixture(name="zoning_subsets", scope="module")
def fix_zoning_subsets(zoning_data: ZoningData) -> tuple[ZoningData, ZoningSystem]:
    """ZoningSystem containing all columns including some subsets."""
    data = ZoningData(
        name=zoning_data.name + "-zoning_subsets",
        data=zoning_data.data.copy(),
        subsets=zoning_data.subsets.copy(),
    )
    system = ZoningSystem(
        name=data.name,
        unique_zones=data.data.copy(),
        metadata=ZoningSystemMetaData(name=data.name, extra_columns=list(data.subsets)),
    )

    return data, system


class TestZoning:
    """Tests for the `ZoningSystem` class."""

    @pytest.mark.parametrize(
        "columns", [["zone_id"], ["zone_id", "zone_name", "descriptions"]]
    )
    @pytest.mark.parametrize("subset", [True, False])
    def test_init(self, zoning_data: ZoningData, columns: list[str], subset: bool) -> None:
        """Test initialising `ZoningSystem` with / without subsets."""
        all_columns = columns.copy()
        if subset:
            all_columns += list(zoning_data.subsets)
        data = (
            zoning_data.data[all_columns]
            .copy()
            .sort_index(axis=0, inplace=False)
            .sort_index(axis=1, inplace=False)
        )

        system = ZoningSystem(
            name=zoning_data.name,
            unique_zones=data,
            metadata=ZoningSystemMetaData(
                name=zoning_data.name,
                extra_columns=list(zoning_data.subsets) if subset else [],
            ),
        )

        data = data.set_index("zone_id")

        assert_array_equal(data.index.values, system.zone_ids, "incorrect zone IDs")
        assert_frame_equal(data, system.zones_data)

        if subset:
            assert sorted(system.subset_columns) == sorted(
                zoning_data.subsets
            ), "incorrect subsets"

        if "zone_name" in columns:
            assert_series_equal(data["zone_name"], system.zone_names())
        if "descriptions" in columns:
            assert_series_equal(data["descriptions"], system.zone_descriptions())

    # def test_init_errors(self, zoning_data: ZoningData) -> None:
    #     """Test initialising ZoningSystem with invalid, or missing, ID column."""
    #     meta = ZoningSystemMetaData(name=zoning_data.name)
    #     # Test missing zone ID column
    #     with pytest.raises(
    #         ValueError, match=r"mandatory ID column \(zone_id\) missing from zones data"
    #     ):
    #         ZoningSystem(
    #             name=zoning_data.name,
    #             unique_zones=pd.DataFrame({"missing": [1, 2, 3]}),
    #             metadata=meta,
    #         )
    #
    #     # Test non-int ID column
    #     with pytest.raises(ValueError, match=r"zone IDs should be integers not object"):
    #         ZoningSystem(
    #             name=zoning_data.name,
    #             unique_zones=pd.DataFrame({"zone_id": ["incorrect", "type"]}),
    #             metadata=meta,
    #         )
    #
    #     # Test duplicate IDs
    #     with pytest.raises(ValueError, match=r"duplicate zone IDs: 1"):
    #         ZoningSystem(
    #             name=zoning_data.name,
    #             unique_zones=pd.DataFrame({"zone_id": [1, 1, 2, 3]}),
    #             metadata=meta,
    #         )

    def test_init_subsets(self, zoning_data: ZoningData) -> None:
        """Test initialising invalid and valid subset columns."""
        meta = ZoningSystemMetaData(name=zoning_data.name)
        # Test subset type conversions
        data = pd.DataFrame(
            {
                "zone_id": [1, 2, 3],
                "str_int": ["1", "0", "1"],
                "str_bool": ["TRUE", "FALSE", "TRUE"],
            }
        )
        system = ZoningSystem(name=meta.name, unique_zones=data, metadata=meta)

        subset = np.array([1, 3])
        assert_array_equal(subset, system.get_subset("str_int"))
        assert_array_equal(subset, system.get_subset("str_bool"))

        # Test invalid subset types
        data = pd.DataFrame(
            {
                "zone_id": [1, 2, 3],
                "invalid_int": [5, 6, 7],
                "invalid_str": ["invalid", "text", "subset"],
            }
        )
        with pytest.raises(
            ValueError, match=r"2 subset columns found which don't contain boolean values:"
        ):
            ZoningSystem(name=meta.name, unique_zones=data, metadata=meta)

    def test_get_subset(self, zoning_subsets: tuple[ZoningData, ZoningSystem]) -> None:
        """Test getting valid subsets and the various errors."""
        data, system = zoning_subsets
        system = system.copy()

        for name, values in data.subsets.items():
            subset = system.get_subset(name)
            assert_array_equal(np.array(values), subset, f"incorrect values for subset {name}")

            subset = system.get_inverse_subset(name)
            assert_array_equal(
                np.array([i for i in data.data["zone_id"] if i not in values]), subset
            )

        with pytest.raises(KeyError):
            system.get_subset("subset_that_doesn't_exist")

        with pytest.raises(ValueError):
            system.get_subset(system._id_column)

        # This shouldn't be done during normal use
        name = "incorrect_dummy_subset"
        system._zones.loc[:, name] = 5
        with pytest.raises(TypeError):
            system.get_subset(name)

    def test_old_to_new_zoning(self, old_zoning_dir: Path, zoning_data: ZoningData) -> None:
        """Test `old_to_new_zoning` method can load in old format and output new."""
        new_dir = old_zoning_dir / "new"
        new_dir.mkdir(exist_ok=True)
        zones = ZoningSystem.old_to_new_zoning(old_zoning_dir, new_dir=new_dir)

        expected = ZoningSystem(
            zoning_data.name,
            zoning_data.data,
            metadata=ZoningSystemMetaData(name=zoning_data.name),
        )
        assert zones == expected, "old zoning data loaded isn't as expected"

        new_dir = new_dir / zoning_data.name
        assert new_dir.is_dir(), "new zoning folder not created"

        out_data = pd.read_csv(new_dir / "zoning.csv")
        for i in (0, 1):
            out_data = out_data.sort_index(axis=i)
            zoning_data.data = zoning_data.data.sort_index(axis=i)

        assert_frame_equal(out_data, zoning_data.data)

        out_meta = ZoningSystemMetaData.load_yaml(new_dir / "zoning_meta.yml")
        assert out_meta == expected.metadata, "incorrect metadata"

    @pytest.mark.parametrize(
        "zone_system_str",
        ["id_only_zoning", "zoning_descriptions", "zoning_subsets"],
    )
    def test_io(self, zone_system_str, main_dir, request) -> None:
        """Test saving and loading ZoningSystem's to / from CSVs.

        HDF I/O makes more sense to be tested with DVec.
        """
        zone_system: ZoningSystem
        _, zone_system = request.getfixturevalue(zone_system_str)
        zone_system.save(main_dir, "csv")
        in_zoning = ZoningSystem.load(main_dir / zone_system.name, "csv")
        assert in_zoning == zone_system, "zone system not equal after save then load"

    def test_zone_trans(
        self,
        test_trans: pd.DataFrame,
        min_zoning_2: ZoningSystem,
        min_zoning: ZoningSystem,
        main_dir: Path,
    ):
        """Test sucessfully obtaining zone_translation data."""
        trans = min_zoning_2.translate(min_zoning, cache_path=main_dir)
        assert trans.equals(test_trans)
        assert min_zoning_2.translation_column_name(min_zoning) == "zone_2_to_zone_1"

    def test_getter(self, id_only_zoning: tuple[ZoningData, ZoningSystem], main_dir):
        """Test finding a zone system based on name."""
        _, system = id_only_zoning
        system.save(main_dir, "csv")
        got_zone = ZoningSystem.get_zoning(system.name, search_dir=main_dir)
        assert got_zone == system, "zoning system not equal after load"
