import pandas as pd
import numpy as np

from pandas.core.frame import DataFrame #class
from pandas.core.series import Series #class

from typing import List

import random
from sklearn.metrics.pairwise import euclidean_distances, pairwise_distances
from sklearn.metrics import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
from recommender.aRecommender import ARecommender  # class

from datasets.ratings import Ratings  # class

from history.aHistory import AHistory #class


class RecommenderCosineCB(ARecommender):

    ARG_CB_DATA_PATH:str = "cbDataPath"

    ARG_USER_PROFILE_STRATEGY:str = "userProfileStrategy"


    DEBUG_MODE = False

    def __init__(self, jobID:str, argumentsDict:dict):
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict is not type dict.")

        # arguments je dictionary, povinny parametr je cesta k souboru s CB daty
        self._arguments:dict = argumentsDict
        # "../../../../data/cbDataOHE.txt" nebo "../../../../data/cbDataTFIDF.txt"
        self.cbDataPath:str = self._arguments[self.ARG_CB_DATA_PATH]
        print(self.cbDataPath)

        self.dfCBFeatures = pd.read_csv(self.cbDataPath, sep=",", header=0, index_col=0)
        dfCBSim = 1 - pairwise_distances(self.dfCBFeatures, metric="cosine")
        np.fill_diagonal(dfCBSim, 0.0)
        self.cbData:DataFrame = DataFrame(data=dfCBSim, index=self.dfCBFeatures.index, columns=self.dfCBFeatures.index)
        self.userProfiles:dict = {}

    def train(self, history:AHistory, ratingsDF:DataFrame, usersDF:DataFrame, itemsDF:DataFrame):
        if type(ratingsDF) is not DataFrame:
            raise ValueError("Argument trainRatingsDF is not type DataFrame.")

        # ratingsSum:Dataframe<(userId:int, movieId:int, ratings:int, timestamp:int)>
        ratingsDF:DataFrame = ratingsDF.loc[ratingsDF[Ratings.COL_RATING] >= 4]
        self.ratingsGroupDF:DataFrame = ratingsDF.groupby(Ratings.COL_USERID)[Ratings.COL_MOVIEID]
        # userProfileDF:DataFrame[userID:int, itemIDs:List[int]]
        userProfileDF:DataFrame = self.ratingsGroupDF.aggregate(lambda x: list(x))
        self.userProfiles:dict = userProfileDF.to_dict()
        s = ""

    def update(self, ratingsUpdateDF:DataFrame):
        # ratingsUpdateDF has only one row
        row = ratingsUpdateDF.iloc[0]
        rating = row[Ratings.COL_RATING]
        if rating >= 4:
            # only positive feedback
            userID = row[Ratings.COL_USERID]
            objectID = row[Ratings.COL_MOVIEID]
            userTrainData = self.userProfiles.get(userID, [])
            userTrainData.append(objectID)
            self.userProfiles[userID] = userTrainData
            s = ""

    def resolveUserProfile(self, userProfileStrategy:str, userTrainData:List[int]):
        rec:str = userProfileStrategy
        if self.DEBUG_MODE:
            print(rec)
        if (len(userTrainData) > 0):
            if (rec == "mean") | (rec == "max"):
                weights = [1.0] * len(userTrainData)
            elif rec == "last":
                userTrainData = userTrainData[-1:]
                weights = [1.0]
            elif rec == "window3":
                userTrainData = userTrainData[-3:]
                weights = [1 / len(userTrainData) * i for i in range(1, (len(userTrainData) + 1))]
            elif rec == "window5":
                userTrainData = userTrainData[-5:]
                weights = [1 / len(userTrainData) * i for i in range(1, (len(userTrainData) + 1))]
            elif rec == "window10":
                userTrainData = userTrainData[-10:]
                weights = [1 / len(userTrainData) * i for i in range(1, (len(userTrainData) + 1))]

            if rec == "max":
                agg = np.max
            else:
                agg = np.mean

            if self.DEBUG_MODE:
                print((userTrainData, weights, agg))
            return (userTrainData, weights, agg)

        return ([], [], "")

    def recommend(self, userID:int, numberOfItems:int=20, argumentsDict:dict={}):
        #print("userID: " + str(userID))
        if type(userID) is not int and type(userID) is not np.int64:
            raise ValueError("Argument userID isn't type int.")
        if type(numberOfItems) is not int and type(numberOfItems) is not np.int64:
            raise ValueError("Argument numberOfItems isn't type int.")
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")

        userProfileStrategy:str = argumentsDict[self.ARG_USER_PROFILE_STRATEGY]

        userTrainData:List[int] = self.userProfiles.get(userID, [])
        objectIDs:List[int]
        weights:List[float]
        objectIDs, weights, aggregation = self.resolveUserProfile(userProfileStrategy, userTrainData)

        simList:List = []

        # provedu agregaci dle zvolené metody
        if len(objectIDs) > 0:
            results = self.cbData.loc[objectIDs]
            weights = np.asarray(weights)
            weights = weights[:, np.newaxis]
            results = results * weights
            results = aggregation(results, axis=0)

            if self.DEBUG_MODE:
                print(type(results))
            results.sort_values(ascending=False, inplace=True, ignore_index=False)
            resultList = results.iloc[0:numberOfItems]

            # print(results[resultList])

            # normalize scores into the unit vector (for aggregation purposes)
            # !!! tohle je zasadni a je potreba provest normalizaci u vsech recommenderu - teda i pro most popular!
            finalScores = resultList.values
            finalScores = normalize(np.expand_dims(finalScores, axis=0))[0, :]

            return pd.Series(finalScores.tolist(), index=list(resultList.index))

        return pd.Series([], index=[])