#!/usr/bin/python3

from typing import List
from typing import Dict

from pandas.core.series import Series #class

from pandas.core.frame import DataFrame #class


class ModelOfIndexes:
    def __init__(self, timeOrderedRatingsDF:DataFrame, relevantDFIndexes:List[int], ratingsClass):
        if type(timeOrderedRatingsDF) is not DataFrame:
            raise ValueError("Argument ratingsDF isn't type DataFrame.")
        if type(relevantDFIndexes) is not list:
            raise ValueError("Argument relevantDFIndexes isn't type list.")
        for relevantDFIndexI in relevantDFIndexes:
            if type(relevantDFIndexI) is not int:
               raise ValueError("Argument relevantDFIndexes don't contain int.")

        self._ratingsDF:DataFrame = timeOrderedRatingsDF
        self._relevantDFIndexes:List[int] = relevantDFIndexes
        self._ratingsClass = ratingsClass


        self._tOrdRatingsDict:Dict[int,DataFrame] = {}

        COL_USERID:str = ratingsClass.getColNameUserID()
        #COL_ITEMID:str = ratingsClass.getColNameItemID()

        userIds:List[int] = list(set([rowI[COL_USERID] for indexDFI, rowI in timeOrderedRatingsDF.iterrows()]))

        userIdI:int
        for userIdI in userIds:
            # select ratings of userIdI
            ratingsUserIDF:DataFrame = timeOrderedRatingsDF.loc[timeOrderedRatingsDF[COL_USERID] == userIdI]
            self._tOrdRatingsDict[userIdI] = ratingsUserIDF
    #

    def __getNextDFIndex(self, dfIndex:int):
        COL_USERID:str = self._ratingsClass.getColNameUserID()
        COL_ITEMID:str = self._ratingsClass.getColNameItemID()

        userId:int = self._ratingsDF[COL_USERID].loc[dfIndex]
        itemId:int = self._ratingsDF[COL_ITEMID].loc[dfIndex]
        #print("userId: " + str(userId))
        #print("itemId: " + str(itemId))

        if not userId in self._tOrdRatingsDict:
            return None
        ratingsOfUserI:DataFrame = self._tOrdRatingsDict[userId]
        indexes:List[int] = list(ratingsOfUserI.index.values.astype(int))

        indexInListOfIndexes:int = indexes.index(dfIndex)

        if indexInListOfIndexes + 1 == len(indexes):
            return None

        return indexes[indexInListOfIndexes+1]



    def getNextDFIndexes(self, dfIndex:int, numberOfIndexes:int):

        dfIndexI:int = dfIndex
        dfIndexes:List[int] = []
        while len(dfIndexes) < numberOfIndexes:
            dfIndexI:int = self.__getNextDFIndex(dfIndexI)
            if dfIndexI == None:
                return dfIndexes
            dfIndexes.append(dfIndexI)

        return dfIndexes


    def getNextItemIDs(self, dfIndex:int, numberOfItems:int):

        COL_ITEMID:str = self._ratingsClass.getColNameItemID()

        dfIndexI:int = dfIndex
        selectedItemIDs:List[int] = []
        while len(selectedItemIDs) < numberOfItems:
            dfIndexI:int = self.__getNextDFIndex(dfIndexI)
            if dfIndexI == None:
                return selectedItemIDs

            itemIdI:int = self._ratingsDF[COL_ITEMID].loc[dfIndexI]

            if not itemIdI in selectedItemIDs:
                selectedItemIDs.append(itemIdI)

        return selectedItemIDs


    def getNextRelevantItemIDs(self, dfIndex:int, numberOfItems:int):

        COL_ITEMID:str = self._ratingsClass.getColNameItemID()

        dfIndexI:int = dfIndex
        selectedItemIDs:List[int] = []
        while len(selectedItemIDs) < numberOfItems:
            dfIndexI:int = self.__getNextDFIndex(dfIndexI)
            if dfIndexI == None:
                return selectedItemIDs
            if not dfIndexI in self._relevantDFIndexes:
                continue

            itemIdI:int = self._ratingsDF[COL_ITEMID].loc[dfIndexI]

            if not itemIdI in selectedItemIDs:
                selectedItemIDs.append(itemIdI)

        return selectedItemIDs


    def getNextRelevantItemIDsExceptItemIDs(self, dfIndex:int, exceptedItemIDs:List[int], numberOfItems:int):

        #print("dfIndex: " + str(dfIndex))
        #print("exceptedItemIDs: " + str(exceptedItemIDs))
        #print("numberOfItems: " + str(numberOfItems))

        COL_ITEMID:str = self._ratingsClass.getColNameItemID()

        dfIndexI:int = dfIndex
        selectedItemIDs:List[int] = []
        while len(selectedItemIDs) < numberOfItems:
            dfIndexI:int = self.__getNextDFIndex(dfIndexI)
            #print("dfIndexI: " + str(dfIndexI))
            if dfIndexI == None:
                return selectedItemIDs
            if not dfIndexI in self._relevantDFIndexes:
                continue
            if dfIndexI in exceptedItemIDs:
                continue

            itemIdI:int = self._ratingsDF[COL_ITEMID].loc[dfIndexI]

            if not itemIdI in selectedItemIDs:
                selectedItemIDs.append(itemIdI)

        return selectedItemIDs

