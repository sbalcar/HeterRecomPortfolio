import pandas as pd
import numpy as np

#algorithm-specific imports
import implicit
import scipy.sparse as sp


from pandas.core.frame import DataFrame #class

from typing import List

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
        self._movieFeaturesMatrix = None
        self._userFeaturesMatrix  = None
        self._factors = argumentsDict[RecommenderBPRMF.ARG_FACTORS]
        self._iterations = argumentsDict[RecommenderBPRMF.ARG_ITERATIONS]
        self._learningRate = argumentsDict[RecommenderBPRMF.ARG_LEARNINGRATE]
        self._regularization = argumentsDict[RecommenderBPRMF.ARG_REGULARIZATION]
        
        self._updateCounter = 0
        self.updateThreshold = 1000   #maybe use values from argumentsDict
        
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
        
            maxUID:int = dataset.usersDF[COL_USERID].max()
            maxOID:int = dataset.itemsDF[COL_ITEMID].max()

        elif type(dataset) is DatasetRetailRocket:
            from datasets.retailrocket.events import Events
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

            maxUID:int = dataset.eventsDF[COL_USERID].max()
            maxOID:int = dataset.eventsDF[COL_ITEMID].max()


        if self.DEBUG_MODE:
            print(maxUID, maxOID)

        ratingsDF[COL_RATING] = 1.0 #flatten ratings
        if type(ratingsDF) is not DataFrame:
            raise ValueError("Argument trainRatingsDF is not type DataFrame.") 
                 
        self._movieFeaturesMatrix = sp.coo_matrix((ratingsDF[COL_RATING],
                   (ratingsDF[COL_ITEMID],
                    ratingsDF[COL_USERID])), shape = (maxOID+1, maxUID+1))
        self._movieFeaturesMatrixLIL =  self._movieFeaturesMatrix.tolil()
                   
        self._userFeaturesMatrix = self._movieFeaturesMatrix.T.tocsr()
        
        self.model = implicit.bpr.BayesianPersonalizedRanking(factors=self._factors, 
            iterations=self._iterations, 
            learning_rate=self._learningRate, 
            regularization=self._regularization, 
            random_state=self._randomState)
        self.model.fit(self._movieFeaturesMatrix)
        

    def update(self, ratingsUpdateDF:DataFrame):
        # ratingsUpdateDF has only one row
        row = ratingsUpdateDF.iloc[0]
        rating = row[Ratings.COL_RATING]
        user = row[Ratings.COL_USERID]
        item = row[Ratings.COL_MOVIEID]
        #print(self._movieFeaturesMatrix)
        
        if rating >= 4:
            self._updateCounter += 1
            #using flat ratings
            self._movieFeaturesMatrixLIL[item, user] =  1.0 #rating
            #print(item, user)
            #print(self._movieFeaturesMatrixLIL[item, user])
            #print(self._updateCounter)
            
            if self._updateCounter == self.updateThreshold:
                self._updateCounter = 0
                #print("updating matrix")
                self._movieFeaturesMatrix =  self._movieFeaturesMatrixLIL.tocsr()
                self._userFeaturesMatrix = self._movieFeaturesMatrix.T.tocsr()
                self.model.fit(self._movieFeaturesMatrix)
                #print(self._movieFeaturesMatrix)
                

 

    def recommend(self, userID:int, numberOfItems:int=20, argumentsDict:dict={}):
        #print("userID: " + str(userID))
        if type(userID) is not int and type(userID) is not np.int64:
            raise ValueError("Argument userID isn't type int.")
        if type(numberOfItems) is not int and type(numberOfItems) is not np.int64:
            raise ValueError("Argument numberOfItems isn't type int.")
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")
        
        if(userID < self._movieFeaturesMatrix.shape[1]):
            recommendations = self.model.recommend(userID, self._userFeaturesMatrix, N = numberOfItems)
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

