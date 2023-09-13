# -*- coding: utf-8 -*-
"""
Created on: 11/09/2023
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:

"""
# Built-Ins
import enum
# Third Party

# Local Imports
# pylint: disable=import-error,wrong-import-position
# Local imports here
from caf.core.segmentation import Segment, Segmentation
# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #
# # # CLASSES # # #
class SegmentsSuper(enum.Enum):
    PURPOSE = 'p'
    TIMEPERIOD = 'tp'
    MODE = 'm'
    GENDER = 'g'
    SOC = 'soc'
    SIC = 'sic'
    CA = 'ca'

    def get_segment(self):
        match self:
            case SegmentsSuper.PURPOSE:
                return Segment(name='purpose',
                               values={'HB Work': 1, 'HB Employers Business (EB)': 2, 'HB Education': 3, 'HB Shopping': 4, 'HB Personal Business (PB)': 5, 'HB Recreation / Social': 6, 'HB Visiting friends and relatives': 7, 'HB Holiday / Day trip': 8, 'NHB Work': 11, 'NHB Employers Business (EB)': 12, 'NHB Education': 13, 'NHB Shopping': 14, 'NHB Personal Business (PB)': 15, 'NHB Recreation / Social': 16, 'NHB Holiday / Day trip': 18}
                               )
            case SegmentsSuper.TIMEPERIOD:
                return Segment(name='time period',
                               values={'Weekday AM peak period (0700 - 0959)': 1, 'Weekday Inter peak period (1000 - 1559)': 2, 'Weekday PM peak period (1600 - 1859)': 3, 'Weekday Off peak (0000 - 0659 and 1900 - 2359)': 4, 'Saturdays (all times of day)': 5, 'Sundays (all times of day)': 6, 'Average Weekday': 7, 'Average Day': 8})
            case SegmentsSuper.MODE:
                return Segment(name='mode',
                               values={'Walk': 1, 'Cycle': 2, 'Car driver': 3, 'Car passenger': 4, 'Bus / Coach': 5, 'Rail / underground': 6})
# # # FUNCTIONS # # #
