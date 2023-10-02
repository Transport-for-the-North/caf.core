"""
to test:
load
save
get
translate
"""
import pandas as pd
import pytest
from caf.core import zoning


@pytest.fixture(name="min_zoning", scope="session")
def fix_min_zoning():
    data = {"zone_id": [1, 2, 3, 4, 5], "zone_name": ["a", "b", "c", "d", "e"]}
    name = "zone_1"
    meta = zoning.ZoningSystemMetaData(name=name)
    return zoning.ZoningSystem(name=name, unique_zones=pd.DataFrame(data), metadata=meta)


@pytest.fixture(name="internal_zoning", scope="session")
def fix_internal(min_zoning):
    internal = pd.Series(["a", "c", "d"])
    return zoning.ZoningSystem(
        name=min_zoning.name,
        unique_zones=min_zoning.unique_zones,
        internal_zones=internal,
        metadata=min_zoning.metadata,
    )


@pytest.fixture(name="external_zoning", scope="session")
def fix_external(min_zoning):
    external = pd.Series(["b", "e"])
    return zoning.ZoningSystem(
        name=min_zoning.name,
        unique_zones=min_zoning.unique_zones,
        external_zones=external,
        metadata=min_zoning.metadata,
    )


@pytest.fixture(name="desc_zoning", scope="session")
def fix_desc(min_zoning):
    desc = pd.DataFrame({"zone_name": ["zone a", "zone b", "zone c", "zone d", "zone e"]})
    return zoning.ZoningSystem(
        name=min_zoning.name,
        unique_zones=min_zoning.unique_zones,
        zone_descriptions=desc,
        metadata=min_zoning.metadata,
    )


@pytest.fixture(name="complete_zone_1", scope="session")
def fix_complete_1(min_zoning, internal_zoning, external_zoning, desc_zoning):
    return zoning.ZoningSystem(
        name=min_zoning.name,
        unique_zones=min_zoning.unique_zones,
        zone_descriptions=desc_zoning.zone_descriptions,
        internal_zones=internal_zoning.internal_zones,
        external_zones=external_zoning.external_zones,
        metadata=min_zoning.metadata,
    )







class TestZoning:
    @pytest.mark.parametrize(
        "zone_system_str",
        ["min_zoning", "internal_zoning", "external_zoning", "desc_zoning", "complete_zone_1"],
    )
    def test_io(self, zone_system_str, main_dir, request):
        # hdf i/o makes more sense to be tested with DVec
        zone_system = request.getfixturevalue(zone_system_str)
        zone_system.save(main_dir, "csv")
        in_zoning = zoning.ZoningSystem.load(main_dir / zone_system.name, "csv")
        assert in_zoning == zone_system

    def test_get_trans(self, test_trans, min_zoning_2, min_zoning, main_dir):
        trans = min_zoning._get_translation_definition(min_zoning_2, trans_cache=main_dir)
        assert trans.equals(test_trans)

    def test_zone_trans(self, test_trans, min_zoning_2, min_zoning, main_dir):
        # this is probably a horrible thing to do
        zoning.ZONE_CACHE_HOME = main_dir
        trans = min_zoning_2.translate(min_zoning, cache_path=main_dir)
        assert trans.equals(test_trans)

    def test_getter(self, min_zoning, main_dir):
        min_zoning.save(main_dir, "csv")
        got_zone = zoning.ZoningSystem.get_zoning(min_zoning.name, search_dir=main_dir)
        assert got_zone == min_zoning
