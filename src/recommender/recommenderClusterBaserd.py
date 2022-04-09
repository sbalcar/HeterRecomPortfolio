#!/usr/bin/python3

from pandas.core.frame import DataFrame  # class

from typing import List
from typing import Dict
from pandas.core.series import Series  # class

from sklearn.preprocessing import normalize

from datasets.aDataset import ADataset  # class
from datasets.datasetML import DatasetML  # class
from datasets.datasetRetailRocket import DatasetRetailRocket  # class
from datasets.datasetST import DatasetST  # class

from recommender.aRecommender import ARecommender  # class

from datasets.ml.ratings import Ratings  # class
from history.aHistory import AHistory  # class

import numpy as np


class RecommenderClusterBased(ARecommender):

    ARG_RECOMMENDER_NUMERIC_ID:str = "recommenderNumericId"

    def __init__(self, batchID:str, argumentsDict:Dict[str, object]):
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict is not type dict.")

        self._argumentsDict: dict = argumentsDict

        self.recommenderNumericId:int = argumentsDict.get(self.ARG_RECOMMENDER_NUMERIC_ID)

        self.numberOfItems: int
        self.result:Series = None

    # ratingsSum:Dataframe<(userId:int, movieId:int, ratings:int, timestamp:int)>
    def train(self, history:AHistory, dataset: ADataset):
        if not isinstance(history, AHistory):
            raise ValueError("Argument history isn't type AHistory.")
        if not isinstance(dataset, ADataset):
            raise ValueError("Argument dataset isn't type ADataset.")
        self.trainDataset = dataset

        if type(dataset) is DatasetML:
            from datasets.ml.items import Items #class
            allGenres:List[str] = Items.getAllGenres()
            self._sortedAscRatings4CountDF:DataFrame = dataset.getTheMostPopularOfGenre(
                    allGenres[self.recommenderNumericId])

        elif type(dataset) is DatasetRetailRocket:
            self._sortedAsceventsTransCountDF:DataFrame = dataset.getTheMostSold()

        elif type(dataset) is DatasetST:
            self._sortedTheMostCommon = dataset.getTheMostSold()

        else:
            raise ValueError("Argument dataset isn't of expected type.")

    def update(self, ratingsUpdateDF:DataFrame, argumentsDict:Dict[str, object]):
        if type(ratingsUpdateDF) is not DataFrame:
            raise ValueError("Argument ratingsTrainDF isn't type DataFrame.")
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict is not type dict.")

    def recommend(self, userID:int, numberOfItems:int, argumentsDict:Dict[str, object]):
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict is not type dict.")

        if not self.result is None:
            if self.numberOfItems == numberOfItems:
                return self.result

        if type(self.trainDataset) is DatasetML:
            # ratings:Dataframe<(movieId:int, ratings:int)>
            ratingsDF:DataFrame = self._sortedAscRatings4CountDF[Ratings.COL_RATING].head(numberOfItems)

        elif type(self.trainDataset) is DatasetRetailRocket:
            from datasets.retailrocket.events import Events  # class
            # ratings:Dataframe<(eventId:int, ratings:int)>
            ratingsDF:DataFrame = self._sortedAsceventsTransCountDF[Events.COL_EVENT].head(numberOfItems)

        elif type(self.trainDataset) is DatasetST:
            from datasets.slantour.events import Events  # class

            # print(self._sortedTheMostCommon)
            if argumentsDict.get(self.ARG_ALLOWED_ITEMIDS) is not None:
                # ARG_ALLOWED_ITEMIDS contains a list of allowed IDs
                # TODO check type of ARG_ALLOWED_ITEMIDS, should be list
                reducedList = self._sortedTheMostCommon.loc[
                    self._sortedTheMostCommon.index.intersection(argumentsDict[self.ARG_ALLOWED_ITEMIDS])]
            else:
                reducedList = self._sortedTheMostCommon

            ratingsDF:DataFrame = reducedList[Events.COL_USER_ID].head(numberOfItems)

        else:
            raise ValueError("Argument dataset isn't of expected type.")

        items:List[int] = list(ratingsDF.index)
        finalScores = normalize(np.expand_dims(ratingsDF, axis=0))[0, :]

        self.numberOfItems: int = numberOfItems
        self.result = Series(finalScores.tolist(), index=items)
        return self.result