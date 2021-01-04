# !/usr/bin/python3

from typing import List

from pandas.core.frame import DataFrame  # class
from pandas.core.series import Series  # class

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from datasets.ml.ratings import Ratings  # class
from datasets.ml.users import Users  # class
from datasets.ml.items import Items #class

from history.aHistory import AHistory  # class
from history.historyDF import HistoryDF #class

from recommender.aRecommender import ARecommender  # class

from datasets.aDataset import ADataset #class
from datasets.datasetML import DatasetML #class
from datasets.datasetRetailrocket import DatasetRetailRocket #class
from datasets.datasetST import DatasetST #class

from scipy.sparse import csr_matrix, lil_matrix
import numpy as np
import pandas as pd

class RecommenderItemBasedKNN(ARecommender):

    def __init__(self, jobID: str, argumentsDict: dict):
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict is not type dict.")

        self._jobID = jobID
        self._argumentsDict: dict = argumentsDict
        self._KNNs: DataFrame = None
        self._distances = None
        self._sparseRatings: lil_matrix = None
        self._modelKNN: NearestNeighbors = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20,
                                                            n_jobs=-1)
        self._lastRatedItemPerUser: DataFrame = None
        self.counter = 0
        self.update_threshold = 100

    def train(self, history: AHistory, dataset:ADataset):
        if not isinstance(history, AHistory):
            raise ValueError("Argument history isn't type AHistory.")
        if not isinstance(dataset, ADataset):
            raise ValueError("Argument dataset isn't type ADataset.")
        self._trainDataset = dataset

        if type(dataset) is DatasetML:
            ratingsTrainDF:DataFrame = dataset.ratingsDF
            cols = pd.to_numeric(ratingsTrainDF[Users.COL_USERID])
            rows = pd.to_numeric(ratingsTrainDF[Items.COL_MOVIEID])
            rating = pd.to_numeric(ratingsTrainDF[Ratings.COL_RATING])

        elif type(dataset) is DatasetRetailRocket:
            from datasets.retailrocket.events import Events #class
            trainEvents2DF:DataFrame = dataset.eventsDF[[Events.COL_VISITOR_ID, Events.COL_ITEM_ID]]
            trainEventsDF:DatasetST = trainEvents2DF.drop_duplicates()

            cols = pd.to_numeric(trainEventsDF[Events.COL_VISITOR_ID])
            rows = pd.to_numeric(trainEventsDF[Events.COL_ITEM_ID])
            rating = pd.to_numeric(Series([1] * len(trainEventsDF), index=trainEventsDF.index))

        elif type(dataset) is DatasetST:
            from datasets.slantour.events import Events  # class
            trainEvents2DF:DataFrame = dataset.eventsDF[[Events.COL_USER_ID, Events.COL_OBJECT_ID]]
            trainEvents2WKDF:DataFrame = trainEvents2DF.loc[trainEvents2DF[Events.COL_OBJECT_ID] != 0]
            trainEventsDF:DatasetST = trainEvents2WKDF.drop_duplicates()

            cols = pd.to_numeric(trainEventsDF[Events.COL_USER_ID])
            rows = pd.to_numeric(trainEventsDF[Events.COL_OBJECT_ID])
            rating = pd.to_numeric(Series([1] * len(trainEventsDF), index=trainEventsDF.index))


        sparseRatingsCSR:csr_matrix = csr_matrix((rating, (rows, cols)))
        sparseRatingsCSR.eliminate_zeros()

        self._modelKNN.fit(sparseRatingsCSR)

        self._distances, self.KNNs = self._modelKNN.kneighbors(n_neighbors=100)

        self._sparseRatings = sparseRatingsCSR.tolil()

        # get last positive feedback from each user
        if type(dataset) is DatasetML:
            self._lastRatedItemPerUser = ratingsTrainDF[ratingsTrainDF[Ratings.COL_RATING] > 3]\
                .sort_values(Ratings.COL_TIMESTAMP)\
                .groupby(Ratings.COL_USERID).tail(1).set_index(Ratings.COL_USERID)\
                .drop(columns=[Ratings.COL_RATING, Ratings.COL_TIMESTAMP])

        elif type(dataset) is DatasetRetailRocket:
            from datasets.retailrocket.events import Events #class
            self._lastRatedItemPerUser = trainEventsDF.loc[trainEventsDF[Events.COL_EVENT] == "transaction"]\
                .sort_values(Events.COL_TIME_STAMP)\
                .groupby(Events.COL_VISITOR_ID).tail(1).set_index(Events.COL_VISITOR_ID)[[Events.COL_VISITOR_ID, Events.COL_ITEM_ID]]

        elif type(dataset) is DatasetST:
            from datasets.slantour.events import Events  # class
            self._lastRatedItemPerUser = trainEventsDF.loc[trainEventsDF[Events.COL_OBJECT_ID] != 0]\
                .groupby(Events.COL_USER_ID).tail(1).set_index(Events.COL_USER_ID)

    def update(self, updtType:str, ratingsUpdateDF:DataFrame):
        if type(updtType) is not str and not updtType in [self.UPDT_CLICK, self.UPDT_VIEW]:
            raise ValueError("Argument updtType isn't type str.")
        if type(ratingsUpdateDF) is not DataFrame:
            raise ValueError("Argument ratingsTrainDF isn't type DataFrame.")

        # the recommender implements only positive feedback
        if updtType == self.UPDT_VIEW:
            return

        row:DataFrame = ratingsUpdateDF.iloc[0]
        if type(self._trainDataset) is DatasetML:
            userID:int = row[Users.COL_USERID]
            itemID:int = row[Items.COL_MOVIEID]
            rating:int = row[Ratings.COL_RATING]

        elif type(self._trainDataset) is DatasetRetailRocket:
            from datasets.retailrocket.events import Events  # class
            userID:int = row[Events.COL_VISITOR_ID]
            itemID:int = row[Events.COL_ITEM_ID]
            rating:int = 1.0

        elif type(self._trainDataset) is DatasetST:
            from datasets.slantour.events import Events  # class
            userID:int = row[Events.COL_USER_ID]
            itemID:int = row[Events.COL_OBJECT_ID]
            rating:int = 1.0


        self._sparseRatings[itemID, userID] = rating

        # update last positive feedback
        if rating > 3:
            self._lastRatedItemPerUser.loc[userID] = [itemID]

        if self.counter > self.update_threshold:
            sparseRatingsCSR: csr_matrix = self._sparseRatings.tocsr()
            self._modelKNN.fit(sparseRatingsCSR)
            self._distances, self.KNNs = self._modelKNN.kneighbors(n_neighbors=100)
            self.counter = 0
        else:
            self.counter += 1


    def recommend(self, userID: int, numberOfItems: int = 20, argumentsDict: dict = {}):
        # Check if user is known
        if userID not in self._lastRatedItemPerUser.index:
            # TODO: How to behave if yet no rating from user was recorded? Maybe return TOP-N most popular items?
            return Series([], index=[])

        if type(self._trainDataset) is DatasetML:
            COL_ITEMID: str = Items.COL_MOVIEID

        elif type(self._trainDataset) is DatasetRetailRocket:
            from datasets.retailrocket.events import Events  # class
            COL_ITEMID: str = Events.COL_ITEM_ID

        elif type(self._trainDataset) is DatasetST:
            from datasets.slantour.events import Events  # class
            COL_ITEMID: str = Events.COL_OBJECT_ID

        # Get recommendations for user
        lastRatedItemFromUser: int = self._lastRatedItemPerUser.loc[userID][COL_ITEMID]
        result: Series = Series(self.KNNs[lastRatedItemFromUser][:numberOfItems])
        finalScores = Series(self._distances[lastRatedItemFromUser][:numberOfItems])

        if self._jobID == 'test' and type(self._trainDataset) is DatasetML:
            print("Last visited film:")
            itemsDF:DataFrame = self._trainDataset.itemsDF
            print(itemsDF[itemsDF['movieId'] == lastRatedItemFromUser])
            print("List of recommendations:")
            for film in result:
                film_info: DataFrame = itemsDF[itemsDF['movieId'] == film]
                print('\t', film_info['movieTitle'].to_string(header=False),
                      film_info['Genres'].to_string(header=False))
        finalScores = normalize(np.expand_dims(finalScores, axis=0))[0, :]

        return Series(finalScores.tolist(), index=list(result))
