#!/usr/bin/python3

from typing import List
from pandas.core.series import Series #class

import pandas as pd

from history.aHistory import AHistory #class

from userBehaviourDescription.userBehaviourDescription import UserBehaviourDescription #class


class HistoryDF(AHistory):

    ITEM_ID = "itemID"
    RECOMMENDED_IDS = "recommendedIds"
    USER_BEHAVIOUR = "userBehaviourSimulator"

    def __init__(self):

        historyData:pd.DataFrame = []
        self._historyDF:pd.DataFrame = pd.DataFrame(historyData, columns=[self.ITEM_ID, self.RECOMMENDED_IDS, self.USER_BEHAVIOUR])


    def addRecommendation(self, itemID:int, recommendedItemIDs:List[int], uObservation:List[bool]):

        new_row:pd.Series = pd.Series({self.ITEM_ID:itemID, self.RECOMMENDED_IDS:recommendedItemIDs, self.USER_BEHAVIOUR:uObservation})
        self._historyDF = self._historyDF.append(new_row, ignore_index=True)


    def getIgnoringValue(self, itemID:int, uBehaviourDesc:UserBehaviourDescription, numberOfItems:int=20, lengthOfHistory:int=20):

        probabilities:List[float] = uBehaviourDesc.getProbabilityOfBehavior(numberOfItems=numberOfItems)
        #print("probabilities: " + str(probabilities))

        def valueOfIgnoring(historyIndex:int):
            rowI:Series = self._historyDF.iloc[historyIndex]
            itemIDs:List[int] = rowI[self.RECOMMENDED_IDS]

            if itemID not in itemIDs:
                return 0.0

            indexI:int = itemIDs.index(itemID)
            return probabilities[indexI]

        return sum(map(valueOfIgnoring, range(len(self._historyDF.index))))


    def print(self):
        print(self._historyDF.head(10))