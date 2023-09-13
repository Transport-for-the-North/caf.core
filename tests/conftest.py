# -*- coding: utf-8 -*-
"""
Created on: 13/09/2023
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:

"""
# Built-Ins

# Third Party
import pytest
# Local Imports
# pylint: disable=import-error,wrong-import-position
# Local imports here
from caf.core import segmentation, segments
# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #
@pytest.fixture(name="seg_1", scope='session')
def fixture_seg_1():
    return segmentation.Segment(name='test seg 1',
                                values={'A': 1,
                                        'B': 2,
                                        'C': 3,
                                        'D': 4})

@pytest.fixture(name="seg_2", scope='session')
def fixture_seg_2():
    return segmentation.Segment(name='test seg 2',
                                values={'X': 1,
                                        'Y': 2,
                                        'Z': 3})

@pytest.fixture(name="basic_segmentation", scope='session')
def fixture_basic_segmentation(seg_1, seg_2):
    segs = [seg_1, seg_2]
    order = ['test seg 2', 'test seg 1']
    return segmentation.Segmentation(segments=segs,
                                     naming_order=order)

# # # CLASSES # # #

# # # FUNCTIONS # # #
