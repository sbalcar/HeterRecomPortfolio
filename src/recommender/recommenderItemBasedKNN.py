# !/usr/bin/python3

from typing import List

from pandas.core.frame import DataFrame  # class
from pandas.core.series import Series  # class

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from datasets.ml.ratings import Ratings  # class
from datasets.ml.users import Users  # class

from history.aHistory import AHistory  # class
from history.historyDF import HistoryDF #class

from recommender.aRecommender import ARecommender  # class

from datasets.aDataset import ADataset #class
from datasets.datasetML import DatasetML #class

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
        self._itemsDF: DataFrame = None
        self.counter = 0
        self.update_threshold = 100

    def train(self, history: AHistory, dataset:ADataset):
        if not isinstance(history, AHistory):
            raise ValueError("Argument history isn't type AHistory.")
        if type(dataset) is not DatasetML:
            raise ValueError("Argument dataset isn't type DatasetML.")

        self._itemsDF = dataset.itemsDF
        ratingsTrainDF = dataset.ratingsDF

        cols:List[int] = pd.to_numeric(ratingsTrainDF[Ratings.COL_USERID])
        rows = pd.to_numeric(ratingsTrainDF[Ratings.COL_MOVIEID])
        rating = pd.to_numeric(ratingsTrainDF[Ratings.COL_RATING])
        sparseRatingsCSR: csr_matrix = csr_matrix((rating, (rows, cols)))
        sparseRatingsCSR.eliminate_zeros()

        self._modelKNN.fit(sparseRatingsCSR)

        self._distances, self.KNNs = self._modelKNN.kneighbors(n_neighbors=100)

        self._sparseRatings = sparseRatingsCSR.tolil()

        # get last positive feedback from each user
        self._lastRatedItemPerUser = \
            ratingsTrainDF[ratingsTrainDF[Ratings.COL_RATING] > 3].sort_values(Ratings.COL_TIMESTAMP) \
                .groupby(Ratings.COL_USERID).tail(1).set_index(Ratings.COL_USERID).drop(columns=[Ratings.COL_RATING, Ratings.COL_TIMESTAMP])


    def update(self, ratingsUpdateDF: DataFrame):
        if type(ratingsUpdateDF) is not DataFrame:
            raise ValueError("Argument ratingsTrainDF isn't type DataFrame.")

        row: DataFrame = ratingsUpdateDF.iloc[0]

        userID: int = row[Ratings.COL_USERID]
        objectID: int = row[Ratings.COL_MOVIEID]
        rating: int = row[Ratings.COL_RATING]

        self._sparseRatings[objectID, userID] = rating

        # update last positive feedback
        if rating > 3:
            self._lastRatedItemPerUser.loc[userID] = [objectID]

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

        # Get recommendations for user
        lastRatedItemFromUser: int = self._lastRatedItemPerUser.loc[userID]['movieId']
        result: Series = Series(self.KNNs[lastRatedItemFromUser][:numberOfItems])
        finalScores = Series(self._distances[lastRatedItemFromUser][:numberOfItems])
        if self._jobID == 'test':
            print("Last visited film:")
            print(self._itemsDF[self._itemsDF['movieId'] == lastRatedItemFromUser])
            print("List of recommendations:")
            for film in result:
                film_info: DataFrame = self._itemsDF[self._itemsDF['movieId'] == film]
                print('\t', film_info['movieTitle'].to_string(header=False),
                      film_info['Genres'].to_string(header=False))
        finalScores = normalize(np.expand_dims(finalScores, axis=0))[0, :]
        return Series(finalScores.tolist(), index=list(result))
