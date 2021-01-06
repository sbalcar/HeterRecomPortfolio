import pandas as pd
import numpy as np

#algorithm-specific imports
import implicit
import scipy.sparse as sp

from pandas.core.frame import DataFrame #class
from pandas.core.series import Series #class

from typing import List
from typing import Dict

from sklearn.metrics import *
from sklearn.preprocessing import normalize
from recommender.aRecommender import ARecommender  # class

from datasets.aDataset import ADataset #class
from datasets.datasetML import DatasetML #class
from datasets.datasetRetailrocket import DatasetRetailRocket #class
from datasets.datasetST import DatasetST #class

from datasets.ml.ratings import Ratings  # class
from datasets.ml.items import Items #class
from datasets.ml.users import Users #class

from history.aHistory import AHistory #class


class RecommenderBPRMF(ARecommender):

    ARG_FACTORS:str = "factors"
    ARG_ITERATIONS:str = "iterations"
    ARG_LEARNINGRATE:str = "learning_rate"
    ARG_REGULARIZATION:str = "regularization"

    DEBUG_MODE = False

    def __init__(self, jobID:str, argumentsDict:dict):
        if type(jobID) is not str:
            raise ValueError("Argument jobID is not type strt.")
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict is not type dict.")

        self._jobID:str = jobID
        self._argumentsDict:dict = argumentsDict
        self._KNNs:DataFrame = None
        self._itemFeaturesMatrix = None
        self._userFeaturesMatrix  = None
        self._factors = argumentsDict[RecommenderBPRMF.ARG_FACTORS]
        self._iterations = argumentsDict[RecommenderBPRMF.ARG_ITERATIONS]
        self._learningRate = argumentsDict[RecommenderBPRMF.ARG_LEARNINGRATE]
        self._regularization = argumentsDict[RecommenderBPRMF.ARG_REGULARIZATION]
        
        self._updateCounter = 0
        self.updateThreshold = 1000   #maybe use values from argumentsDict

        self.userIdToUserIndexDict:Dict = {}
        self.userIndexToUserIdDict:Dict = {}

        self.itemIdToItemIndexDict:Dict = {}
        self.itemIndexToItemIdDict:Dict = {}

        self._randomState = 42
        

    def train(self, history:AHistory, dataset:ADataset):
        if not isinstance(history, AHistory):
            raise ValueError("Argument history isn't type AHistory.")
        if not isinstance(dataset, ADataset):
            raise ValueError("Argument dataset isn't type ADataset.")
        self._trainDataset:ADataset = dataset

        if type(dataset) is DatasetML:
            COL_RATING:str = Ratings.COL_RATING
            COL_USERID:str = Users.COL_USERID
            COL_ITEMID:str = Items.COL_MOVIEID
            ratingsDF:DataFrame = dataset.ratingsDF.loc[dataset.ratingsDF[COL_RATING] >= 4]

        elif type(dataset) is DatasetRetailRocket:
            from datasets.retailrocket.events import Events  # class
            COL_RATING:str = "rating"
            COL_USERID:str = Events.COL_VISITOR_ID
            COL_ITEMID:str = Events.COL_ITEM_ID

            ratings4DF:DataFrame = dataset.eventsDF[[COL_USERID, COL_ITEMID, Events.COL_EVENT]]
            ratings4DF = ratings4DF.drop_duplicates()

            ratings4DF.loc[ratings4DF[Events.COL_EVENT] == "view", COL_RATING] = 1
            ratings4DF.loc[ratings4DF[Events.COL_EVENT] == "addtocart", COL_RATING] = 2
            ratings4DF.loc[ratings4DF[Events.COL_EVENT] == "transaction", COL_RATING] = 3

            ratingsDF:DataFrame = ratings4DF[[COL_USERID, COL_ITEMID, COL_RATING]]
            ratingsDF = ratingsDF.groupby([COL_USERID, COL_ITEMID], as_index=False)[COL_RATING].max()

        elif type(self._trainDataset) is DatasetST:
            from datasets.slantour.events import Events  # class
            COL_RATING:str = "rating"
            COL_USERID:str = Events.COL_USER_ID
            COL_ITEMID:str = Events.COL_OBJECT_ID

            trainEvents2DF:DataFrame = dataset.eventsDF[[Events.COL_USER_ID, Events.COL_OBJECT_ID]]
            trainEventsDF:DatasetST = trainEvents2DF.drop_duplicates()
            trainEventsDF[COL_RATING] = 1.0
            ratingsDF:DataFrame = trainEventsDF

        ratingsDF[COL_RATING] = 1.0 #flatten ratings

        self.userIdToUserIndexDict:Dict[int, int] = {val: i for (i, val) in enumerate(ratingsDF[COL_USERID].unique())}
        self.userIndexToUserIdDict:Dict[int, int] = {v: k for k, v in self.userIdToUserIndexDict.items()}

        self.itemIdToItemIndexDict:Dict[int, int] = {val: i for (i, val) in enumerate(ratingsDF[COL_ITEMID].unique())}
        self.itemIndexToItemIdDict:Dict[int, int] = {v: k for k, v in self.itemIdToItemIndexDict.items()}

        userIndexes:List[int] = [self.userIdToUserIndexDict[i] for i in ratingsDF[COL_USERID]]
        itemIndexes:List[int] = [self.itemIdToItemIndexDict[i] for i in ratingsDF[COL_ITEMID]]

        maxUID:int = len(userIndexes)-1
        maxOID:int = len(itemIndexes)-1

        self._itemFeaturesMatrix = sp.coo_matrix(
                    (ratingsDF[COL_RATING], (itemIndexes, userIndexes)),
            shape = (maxOID+1, maxUID+1))
        self._itemFeaturesMatrixLIL =  self._itemFeaturesMatrix.tolil()
                   
        self._userFeaturesMatrix = self._itemFeaturesMatrix.T.tocsr()
        
        self.model = implicit.bpr.BayesianPersonalizedRanking(factors=self._factors, 
            iterations=self._iterations, 
            learning_rate=self._learningRate, 
            regularization=self._regularization, 
            random_state=self._randomState)
        self.model.fit(self._itemFeaturesMatrix)
        

    def update(self, ratingsUpdateDF:DataFrame):
        if type(ratingsUpdateDF) is not DataFrame:
            raise ValueError("Argument ratingsTrainDF isn't type DataFrame.")

        # ratingsUpdateDF has only one row
        row = ratingsUpdateDF.iloc[0]

        if type(self._trainDataset) is DatasetML:
            rating = row[Ratings.COL_RATING]
            userID = row[Ratings.COL_USERID]
            itemID = row[Ratings.COL_MOVIEID]
            if rating < 4:
                return

        elif type(self._trainDataset) is DatasetRetailRocket:
            from datasets.retailrocket.events import Events
            rating = 1.0
            userID = row[Events.COL_VISITOR_ID]
            itemID = row[Events.COL_ITEM_ID]

        elif type(self._trainDataset) is DatasetST:
            from datasets.slantour.events import Events  # class
            rating = 1.0
            userID = row[Events.COL_USER_ID]
            itemID = row[Events.COL_OBJECT_ID]


        #using flat ratings
        if not userID in self.userIdToUserIndexDict:
            newUIndex:int = len(self.userIdToUserIndexDict)
            self.userIdToUserIndexDict[userID] = newUIndex
            self.userIndexToUserIdDict[newUIndex] = userID

        if not itemID in self.itemIdToItemIndexDict:
            newIIndex:int = len(self.itemIdToItemIndexDict)
            self.itemIdToItemIndexDict[itemID] = newIIndex
            self.itemIndexToItemIdDict[newIIndex] = itemID

        userIndex:int = self.userIdToUserIndexDict[userID]
        itemIndex:int = self.itemIdToItemIndexDict[itemID]

        #print("itemID: " + str(itemID))
        #print("itemIndex: " + str(itemIndex))
        #print("userID: " + str(userID))
        #print("userIndex: " + str(userIndex))

        self._itemFeaturesMatrixLIL[itemIndex, userIndex] =  1.0 #rating
        #print(item, user)
        #print(self._movieFeaturesMatrixLIL[item, user])
        #print(self._updateCounter)

        self._updateCounter += 1
        if self._updateCounter == self.updateThreshold:
            self._updateCounter = 0
            #print("updating matrix")
            self._itemFeaturesMatrix =  self._itemFeaturesMatrixLIL.tocsr()
            self._userFeaturesMatrix = self._itemFeaturesMatrix.T.tocsr()
            self.model.fit(self._itemFeaturesMatrix)
            #print(self._movieFeaturesMatrix)
                

 

    def recommend(self, userID:int, numberOfItems:int=20, argumentsDict:dict={}):
        #print("userID: " + str(userID))
        if type(userID) is not int and type(userID) is not np.int64:
            raise ValueError("Argument userID isn't type int.")
        if type(numberOfItems) is not int and type(numberOfItems) is not np.int64:
            raise ValueError("Argument numberOfItems isn't type int.")
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")

        if not userID in self.userIdToUserIndexDict:
            self.userIdToUserIndexDict[userID] = len(self.userIdToUserIndexDict)
            self.userIndexToUserIdDict = {v: k for k, v in self.userIdToUserIndexDict.items()}

        userIndex:int = self.userIdToUserIndexDict[userID]
        if(userIndex < self._itemFeaturesMatrix.shape[1]):
            # recommendationOfIndexes:Series
            recommendationOfIndexesListOfTuple:List[tuple] = self.model.recommend(userIndex, self._userFeaturesMatrix, N = numberOfItems)

            recommendations:List[tuple] = [(self.itemIndexToItemIdDict[itemIndexI], rI) for itemIndexI, rI in recommendationOfIndexesListOfTuple]

        else: #cannot recommend for unknown user
            return pd.Series([], index=[])


        # provedu agregaci dle zvolené metody
        if len(recommendations) == 0:
            return pd.Series([], index=[])

        if self.DEBUG_MODE:
            print(type(recommendations))
        recItems = [i[0] for i in recommendations]
        recScores = [i[1] for i in recommendations]

        if self.DEBUG_MODE and type(self._trainDataset) is DatasetML:
            idf = self._trainDataset.itemsDF.set_index("movieId")
            print(idf.loc[recItems])


        # normalize scores into the unit vector (for aggregation purposes)
        # !!! tohle je zasadni a je potreba provest normalizaci u vsech recommenderu - teda i pro most popular!
        finalScores = normalize(np.expand_dims(recScores, axis=0))[0, :]
        return pd.Series(finalScores.tolist(), index=recItems)

