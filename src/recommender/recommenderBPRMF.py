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

from datasets.ml.ratings import Ratings  # class
from datasets.ml.items import Items #class
from datasets.ml.users import Users #class

from history.aHistory import AHistory #class


class RecommenderBPRMF(ARecommender):

    ARG_USER_PROFILE_STRATEGY:str = "userProfileStrategy"


    DEBUG_MODE = False

    def __init__(self, jobID:str, argumentsDict:dict):
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict is not type dict.")

        self._jobID = jobID
        self._argumentsDict: dict = argumentsDict
        self._KNNs:DataFrame = None
        self._movieFeaturesMatrix = None
        self._userFeaturesMatrix  = None
        self._factors = 20      #use values from argumentsDict
        self._iterations = 10   #use values from argumentsDict
        self._updateCounter = 0
        self.updateThreshold = 1000   #maybe use values from argumentsDict

    def train(self, history:AHistory, dataset:ADataset):
        if type(dataset) is not DatasetML:
            raise ValueError("Argument dataset is not type DatasetML.")

        ratingsDF:DataFrame = dataset.ratingsDF
        usersDF:DataFrame = dataset.usersDF
        itemsDF:DataFrame = dataset.itemsDF

        ratingsDF:DataFrame = ratingsDF.loc[ratingsDF[Ratings.COL_RATING] >= 4]
        
        maxUID = usersDF[Users.COL_USERID].max()
        maxOID = itemsDF[Items.COL_MOVIEID].max()
        print(maxUID, maxOID)
        ratingsDF[Ratings.COL_RATING] = 1.0 #flatten ratings
        if type(ratingsDF) is not DataFrame:
            raise ValueError("Argument trainRatingsDF is not type DataFrame.") 
                 
        self._movieFeaturesMatrix = sp.coo_matrix((ratingsDF[Ratings.COL_RATING], 
                   (ratingsDF[Ratings.COL_USERID], 
                    ratingsDF[Ratings.COL_MOVIEID])), shape = (maxOID, maxUID))
        self._movieFeaturesMatrixLIL =  self._movieFeaturesMatrix.tolil()
                   
        self._userFeaturesMatrix = self._movieFeaturesMatrix.T.tocsr()
        
        self.model = implicit.bpr.BayesianPersonalizedRanking(factors=self._factors, iterations=self._iterations)
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
            self._movieFeaturesMatrixLIL[user, item] =  1.0 #rating
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

        #userProfileStrategy:str = argumentsDict[self.ARG_USER_PROFILE_STRATEGY]
        
        if(userID < self._movieFeaturesMatrix.shape[1]):
            recommendations = self.model.recommend(userID, self._userFeaturesMatrix, N = numberOfItems)
        else: #cannot recommend for unknown user
            return pd.Series([], index=[])


        # provedu agregaci dle zvolené metody
        if len(recommendations) > 0:
            if self.DEBUG_MODE:
                print(type(recommendations))
            recItems = [i[0] for i in recommendations]
            recScores = [i[1] for i in recommendations]

            # print(results[resultList])

            # normalize scores into the unit vector (for aggregation purposes)
            # !!! tohle je zasadni a je potreba provest normalizaci u vsech recommenderu - teda i pro most popular!
            finalScores = normalize(np.expand_dims(recScores, axis=0))[0, :]
            return pd.Series(finalScores.tolist(), index=recItems)

        return pd.Series([], index=[])