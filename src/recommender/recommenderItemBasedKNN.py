# !/usr/bin/python3

import math
from typing import List
from typing import Dict

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

    ARG_K:str = "k"

    def __init__(self, jobID:str, argumentsDict:Dict[str,str]):
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict is not type dict.")

        self._jobID:str = jobID
        self._argumentsDict:Dict[str,str] = argumentsDict
        self._KNNs:DataFrame = None
        self._distances = None
        self._sparseRatings:lil_matrix = None
        self._modelKNN:NearestNeighbors = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)

        self._userIdToUserIndexDict:Dict[int, int] = {}
        self._userIndexToUserIdDict:Dict[int, int] = {}

        self._itemIdToItemIndexDict:Dict[int, int] = {}
        self._itemIndexToItemIdDict:Dict[int, int] = {}

        self._lastRatedItemPerUser:DataFrame = None
        self.counter:int = 0
        self.update_threshold:int = 5

    def train(self, history:AHistory, dataset:ADataset):
        if not isinstance(history, AHistory):
            raise ValueError("Argument history isn't type AHistory.")
        if not isinstance(dataset, ADataset):
            raise ValueError("Argument dataset isn't type ADataset.")
        self._trainDataset = dataset

        if type(dataset) is DatasetML:
            COL_USERID:str = Users.COL_USERID
            COL_ITEMID:str = Items.COL_MOVIEID
            trainRatingsDF:DataFrame = dataset.ratingsDF
        elif type(dataset) is DatasetRetailRocket:
            from datasets.retailrocket.events import Events #class
            COL_USERID:str = Events.COL_VISITOR_ID
            COL_ITEMID:str = Events.COL_ITEM_ID
            trainEvents2DF:DataFrame = dataset.eventsDF[[Events.COL_VISITOR_ID, Events.COL_ITEM_ID]]
            trainRatingsDF:DatasetST = trainEvents2DF.drop_duplicates()
        elif type(dataset) is DatasetST:
            from datasets.slantour.events import Events  # class
            COL_USERID:str = Events.COL_USER_ID
            COL_ITEMID:str = Events.COL_OBJECT_ID
            trainEvents2DF:DataFrame = dataset.eventsDF[[Events.COL_USER_ID, Events.COL_OBJECT_ID]]
            trainEvents2WKDF:DataFrame = trainEvents2DF.loc[trainEvents2DF[Events.COL_OBJECT_ID] != 0]
            trainRatingsDF:DatasetST = trainEvents2WKDF.drop_duplicates()


        self._userIdToUserIndexDict:Dict[int, int] = {val: i for (i, val) in enumerate(trainRatingsDF[COL_USERID].unique())}
        self._userIndexToUserIdDict:Dict[int, int] = {v: k for k, v in self._userIdToUserIndexDict.items()}

        self._itemIdToItemIndexDict:Dict[int, int] = {val: i for (i, val) in enumerate(trainRatingsDF[COL_ITEMID].unique())}
        self._itemIndexToItemIdDict:Dict[int, int] = {v: k for k, v in self._itemIdToItemIndexDict.items()}

        userIndexes:List[int] = [self._userIdToUserIndexDict[i] for i in trainRatingsDF[COL_USERID]]
        itemIndexes:List[int] = [self._itemIdToItemIndexDict[i] for i in trainRatingsDF[COL_ITEMID]]


        sparseRatingsCSR:csr_matrix = csr_matrix(([1.0]*len(itemIndexes), (itemIndexes, userIndexes)),
                                                 shape=(2*len(self._itemIdToItemIndexDict), 2*len(self._userIdToUserIndexDict)))
        sparseRatingsCSR.eliminate_zeros()

        self._modelKNN.fit(sparseRatingsCSR)

        self._distances, self.KNNs = self._modelKNN.kneighbors(n_neighbors=100)

        self._sparseRatings = sparseRatingsCSR.tolil()

        # get last positive feedback from each user
        if type(dataset) is DatasetML:
            self._lastRatedItemPerUser = trainRatingsDF[trainRatingsDF[Ratings.COL_RATING] > 3]\
                .sort_values(Ratings.COL_TIMESTAMP)\
                .groupby(Ratings.COL_USERID).tail(1).set_index(Ratings.COL_USERID)\
                .drop(columns=[Ratings.COL_RATING, Ratings.COL_TIMESTAMP])

        elif type(dataset) is DatasetRetailRocket:
            from datasets.retailrocket.events import Events #class
            self._lastRatedItemPerUser = trainRatingsDF.loc[trainRatingsDF[Events.COL_EVENT] == "transaction"]\
                .sort_values(Events.COL_TIME_STAMP)\
                .groupby(Events.COL_VISITOR_ID).tail(1).set_index(Events.COL_VISITOR_ID)[[Events.COL_VISITOR_ID, Events.COL_ITEM_ID]]

        elif type(dataset) is DatasetST:
            from datasets.slantour.events import Events  # class
            self._lastRatedItemPerUser = trainRatingsDF.loc[trainRatingsDF[Events.COL_OBJECT_ID] != 0]\
                .groupby(Events.COL_USER_ID).tail(1).set_index(Events.COL_USER_ID)

    def update(self, ratingsUpdateDF:DataFrame, argumentsDict:Dict[str,object]):
        if type(ratingsUpdateDF) is not DataFrame:
            raise ValueError("Argument ratingsTrainDF isn't type DataFrame.")
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")

        row:DataFrame = ratingsUpdateDF.iloc[0]
        if type(self._trainDataset) is DatasetML:
            userID:int = row[Users.COL_USERID]
            itemID:int = row[Items.COL_MOVIEID]
            rating:int = row[Ratings.COL_RATING]

        elif type(self._trainDataset) is DatasetRetailRocket:
            from datasets.retailrocket.events import Events  # class
            userID:int = row[Events.COL_VISITOR_ID]
            itemID:int = row[Events.COL_ITEM_ID]
            rating:int = 4.0

        elif type(self._trainDataset) is DatasetST:
            from datasets.slantour.events import Events  # class
            userID:int = row[Events.COL_USER_ID]
            itemID:int = row[Events.COL_OBJECT_ID]
            rating:int = 4.0

        if not userID in self._userIdToUserIndexDict:
            userIndex:int = len(self._userIdToUserIndexDict)
            self._userIdToUserIndexDict[userID] = userIndex
            self._userIndexToUserIdDict[userIndex] = userID

        if not itemID in self._itemIdToItemIndexDict:
            itemIndex:int = len(self._itemIdToItemIndexDict)
            self._itemIdToItemIndexDict[itemID] = itemIndex
            self._itemIndexToItemIdDict[itemIndex] = itemID

        userIndex:int = self._userIdToUserIndexDict[userID]
        itemIndex:int = self._itemIdToItemIndexDict[itemID]

        self._sparseRatings[itemIndex, userIndex] = rating

        # update last positive feedback
        if rating > 3:
            self._lastRatedItemPerUser.loc[userID] = [itemID]

        if self.counter > self.update_threshold:
            print("update model")
            sparseRatingsCSR: csr_matrix = self._sparseRatings.tocsr()
            self._modelKNN.fit(sparseRatingsCSR)
            self._distances, self.KNNs = self._modelKNN.kneighbors(n_neighbors=100)
            self.counter = 0
        else:
            self.counter += 1


    def recommend(self, userID: int, numberOfItems:int, argumentsDict:Dict[str,object]):
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")

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
        lastRatedItemFromUser:int = self._lastRatedItemPerUser.loc[userID][COL_ITEMID]
        
        #adding currently viewed item (if any) into the user profile
        itemID = argumentsDict.get("itemID", 0)
        if itemID > 0:
           lastRatedItemFromUser = itemID
        
        lastRatedIndex:int = self._itemIdToItemIndexDict[lastRatedItemFromUser]

        rItemIndexes:Series = Series(self.KNNs[lastRatedIndex][:5*numberOfItems])
        rDistances:List[int] = self._distances[lastRatedIndex][:5*numberOfItems]
        # maping Indexes to IDs
        rItemIDsSrs:Series = Series([self._itemIndexToItemIdDict.get(rItemIndexI,-1) for rItemIndexI in rItemIndexes])
        rRatingsSrs:Series = Series([1- dI for dI in rDistances])

        nanIndexes:List[int] = [i for i, v in enumerate(rItemIDsSrs.tolist()) if v == -1]
        rItemIDs:List[int] = rItemIDsSrs.drop(rItemIDsSrs.index[nanIndexes]).tolist()
        rRatings:List[int] = rRatingsSrs.drop(rRatingsSrs.index[nanIndexes]).tolist()

        if argumentsDict.get(self.ARG_ALLOWED_ITEMIDS) is not None:
            allowedIndexes:List[int] = [eI for eI, rItemIdI in enumerate(rItemIDs)
                                        if rItemIdI in argumentsDict[self.ARG_ALLOWED_ITEMIDS]]
            rItemIDs = [rItemIDs[i] for i in allowedIndexes]
            rRatings = [rRatings[i] for i in allowedIndexes]

        rItemIDs = rItemIDs[:numberOfItems]
        rRatings = rRatings[:numberOfItems]

        if self._jobID == 'test' and type(self._trainDataset) is DatasetML:
            print("Last visited film:")
            itemsDF:DataFrame = self._trainDataset.itemsDF
            print(itemsDF[itemsDF['movieId'] == lastRatedItemFromUser])
            print("List of recommendations:")
            for filmI in rItemIndexes:
                film_info:DataFrame = itemsDF[itemsDF[Ratings.COL_MOVIEID] == filmI]
                print('\t', film_info['movieTitle'].to_string(header=False),
                      film_info['Genres'].to_string(header=False))

        if len(rItemIDs) == 0:
            return Series([], index=[])
        rRatings = normalize(np.expand_dims(rRatings, axis=0))[0, :]
        return Series(rRatings, index=rItemIDs)