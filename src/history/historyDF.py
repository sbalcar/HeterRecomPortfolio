#!/usr/bin/python3

from typing import List

import pandas as pd

from history.aHistory import AHistory #class


class HistoryDF(AHistory):

    ITEM_ID = "itemID"
    POSITION_IN_RECOMMENDATION = "positionInRecommendation"

    def __init__(self):

        historyData:pd.DataFrame = []
        self._historyDF:pd.DataFrame = pd.DataFrame(historyData, columns=[self.ITEM_ID, self.POSITION_IN_RECOMMENDATION])


    def addRecommendation(self, itemID:int, recommendedItemIDs:List[int]):

        new_row:pd.Series = pd.Series({self.ITEM_ID: itemID, self.POSITION_IN_RECOMMENDATION: recommendedItemIDs})
        self._historyDF.append(new_row, ignore_index=True)