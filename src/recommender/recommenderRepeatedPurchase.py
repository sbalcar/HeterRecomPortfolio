#!/usr/bin/python3

from pandas.core.frame import DataFrame #class

from typing import List
from typing import Dict
from pandas.core.series import Series #class

from sklearn.preprocessing import normalize

from datasets.aDataset import ADataset #class
from datasets.datasetML import DatasetML #class
from datasets.datasetRetailrocket import DatasetRetailRocket #class
from datasets.retailrocket.events import Events #class
from datasets.datasetST import DatasetST #class

from recommender.aRecommender import ARecommender #class

from datasets.ml.ratings import Ratings #class
from history.aHistory import AHistory #class

import numpy as np


class RecommenderRepeatedPurchase(ARecommender):

    def __init__(self, batchID:str, argumentsDict:Dict[str, object]):
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict is not type dict.")

        self._argumentsDict:dict = argumentsDict

        self.numberOfItems:int
        self.result:Series = None

        self.itemsAddTraIdOfUsersDict:dict[int,List] = {}

        self.theMostOftenRepeatedlyBoughtItemsDF:DataFrame = None

    # ratingsSum:Dataframe<(userId:int, movieId:int, ratings:int, timestamp:int)>
    def train(self, history:AHistory, dataset:ADataset):
        if not isinstance(history, AHistory):
            raise ValueError("Argument history isn't type AHistory.")
        if not isinstance(dataset, ADataset):
            raise ValueError("Argument dataset isn't type ADataset.")
        self.history = history
        self.trainDataset = dataset

        if type(dataset) is DatasetML:
            raise ValueError("Not implemented yet.")

        elif type(dataset) is DatasetRetailRocket:
            self._sortedAsceventsTransCountDF:DataFrame = dataset.getTheMostSold()

            eventsDF = dataset.eventsDF

            # select users in dataset
            userIDs:List[int] = list(eventsDF[Events.COL_VISITOR_ID].unique())

            # select items from dataset of given users having events addtocart or transaction
            for userIdI in userIDs:
                eventsOfUserIDF = eventsDF.loc[eventsDF[Events.COL_VISITOR_ID] == userIdI]
                eventsAddTrOfUserIDF:DataFrame = eventsOfUserIDF.loc[eventsOfUserIDF[Events.COL_EVENT].isin([Events.EVENT_TRANSACTION, Events.EVENT_ADDTOCART])]
                itemIDsI:List[int] = eventsAddTrOfUserIDF[Events.COL_ITEM_ID].tolist()

                self.itemsAddTraIdOfUsersDict[userIdI] = itemIDsI

            # the most often repeatedly bought items
            eventsTransDF:DataFrame = eventsDF.loc[eventsDF[Events.COL_EVENT] == "transaction"]
            events2DF = eventsTransDF.groupby(
                [Events.COL_VISITOR_ID, Events.COL_ITEM_ID], as_index=False)[Events.COL_TIME_STAMP].count()
            self.theMostOftenRepeatedlyBoughtItemsDF = events2DF.loc[events2DF[Events.COL_TIME_STAMP] > 1].sort_values(Events.COL_TIME_STAMP, ascending=False)

        elif type(dataset) is DatasetST:
            raise ValueError("Not implemented yet.")

        else:
            raise ValueError("Argument dataset isn't of expected type.")

    def update(self, ratingsUpdateDF:DataFrame, argumentsDict:Dict[str, object]):
        if type(ratingsUpdateDF) is not DataFrame:
            raise ValueError("Argument ratingsTrainDF isn't type DataFrame.")
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict is not type dict.")

        row = ratingsUpdateDF.iloc[0]
        userID:int = row[Events.COL_VISITOR_ID]
        itemID:int = row[Events.COL_ITEM_ID]

        if not userID in self.itemsAddTraIdOfUsersDict.keys():
            self.itemsAddTraIdOfUsersDict[userID] = []
        self.itemsAddTraIdOfUsersDict[userID].append(itemID)

    def recommend(self, userID: int, numberOfItems: int, argumentsDict: Dict[str, object]):
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict is not type dict.")

        if not self.result is None:
            if self.numberOfItems == numberOfItems:
                return self.result

        if type(self.trainDataset) is DatasetML:
            raise ValueError("Not implemented yet.")
        elif type(self.trainDataset) is DatasetRetailRocket:
            from datasets.retailrocket.events import Events  # class

            itemsOfUser:List[int] = self.itemsAddTraIdOfUsersDict.get(userID, [])
            #print(self.theMostOftenRepeatedlyBoughtItemsDF)
            theMostOftenRepeatedlyBoughtItemsIDs:List[int] = self.theMostOftenRepeatedlyBoughtItemsDF[Events.COL_ITEM_ID].tolist()

            candidates:List[int] = list(set(theMostOftenRepeatedlyBoughtItemsIDs) & set(itemsOfUser))

            itemsAddTraIdReversed:List[int] = itemsOfUser[::-1]

            # we add one to avoid having zero index - zero index -> zero relevance
            theNearestIndexesInHistoryOfAddTraItemIds:List[int] = [itemsAddTraIdReversed.index(cI) +1 for cI in candidates]
            resultSer:Series =  Series(theNearestIndexesInHistoryOfAddTraItemIds, index=candidates)

            resultSer = resultSer.sort_values(ascending=False)

        elif type(self.trainDataset) is DatasetST:
            from datasets.slantour.events import Events  # class
            raise ValueError("Not implemented yet.")
        else:
            raise ValueError("Argument dataset isn't of expected type.")


        if len(resultSer.values) <= 0:
            return Series([], index=[])

        finalScores = normalize(np.expand_dims(resultSer.values, axis=0))[0, :]


        self.numberOfItems:int = numberOfItems
        self.result = Series(finalScores.tolist(), index=resultSer.index)
        return self.result