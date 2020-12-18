#!/usr/bin/python3

from pandas.core.frame import DataFrame #class

from typing import List
from pandas.core.series import Series #class

from sklearn.preprocessing import normalize

from datasets.aDataset import ADataset #class
from datasets.datasetRetailrocket import DatasetRetailRocket #class

from recommender.aRecommender import ARecommender #class

from datasets.ml.ratings import Ratings #class
from datasets.retailrocket.events import Events #class
from history.aHistory import AHistory #class

import numpy as np

class RecommenderTheMostSold(ARecommender):

    def __init__(self, jobID:str, argumentsDict:dict):
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict is not type dict.")

        self._argumentsDict:dict = argumentsDict

        self.numberOfItems:int
        self.result:Series = None

    # ratingsSum:Dataframe<(userId:int, movieId:int, ratings:int, timestamp:int)>
    def train(self, history:AHistory, dataset:DatasetRetailRocket):
        if not isinstance(history, AHistory):
            raise ValueError("Argument history isn't type AHistory.")
        if type(dataset) is not DatasetRetailRocket:
            raise ValueError("Argument dataset isn't type DatasetRetailRocket.")

        eventsTrainDF:DataFrame = dataset.eventsDF

        # ratingsSum:Dataframe<(timestamp:int, visitorid:int, event:str, itemid:int, transactionid:int)>
        eventsTransDF:DataFrame = eventsTrainDF.loc[eventsTrainDF[Events.COL_EVENT] == "transaction"]

        # ratingsSum:Dataframe<(movieId:int, ratings:int)>
        eventsTransSumDF:DataFrame = DataFrame(eventsTransDF.groupby(Events.COL_ITEM_ID)[Events.COL_EVENT].count())

        # sortedAsceventsTransCountDF:Dataframe<(movieId:int, ratings:int)>
        sortedAsceventsTransCountDF:DataFrame = eventsTransSumDF.sort_values(by=Events.COL_EVENT, ascending=False)
        #print(sortedAsceventsTransCountDF)

        self._sortedAsceventsTransCountDF:DataFrame = sortedAsceventsTransCountDF


    def update(self, ratingsUpdateDF:DataFrame):
        pass

    def recommend(self, userID:int, numberOfItems:int=20, argumentsDict:dict={}):
        #print("userID: " + str(userID))

        if not self.result is None:
            if self.numberOfItems == numberOfItems:
                return self.result

        # ratings:Dataframe<(movieId:int, ratings:int)>
        ratingsDF:DataFrame = self._sortedAsceventsTransCountDF[Events.COL_EVENT].head(numberOfItems)

        items:List[int] = list(ratingsDF.index)
        finalScores = normalize(np.expand_dims(ratingsDF, axis=0))[0, :]

        self.numberOfItems:int = numberOfItems
        self.result = Series(finalScores.tolist(),index=items)
        return self.result