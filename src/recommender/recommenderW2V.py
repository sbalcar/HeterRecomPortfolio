#!/usr/bin/python3

import pickle
import os

import pandas as pd
import numpy as np

from configuration.configuration import Configuration #class

from typing import List

from recommender.w2v import word2vec

from pandas.core.frame import DataFrame #class

from sklearn.metrics import *
from sklearn.preprocessing import normalize
from recommender.aRecommender import ARecommender  #class

from datasets.ml.ratings import Ratings  #class
from history.aHistory import AHistory #class


class RecommenderW2V(ARecommender):

    # mandatory argument
    ARG_TRAIN_VARIANT:str = "trainVariant"

    # mandatory argument
    ARG_USER_PROFILE_STRATEGY:str = "userProfileStrategy"

    # mandatory argument
    ARG_DATASET_ID:str = "datasetId"

    DEBUG_MODE = False

    # ratingsSum:Dataframe<(userId:int, movieId:int, ratings:int, timestamp:int)>
    def __init__(self, jobID:str, argumentsDict:dict):
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict is not type dict.")
        self._jobID:str = jobID
        self._arguments:dict = argumentsDict

        self.trainVariant:str = self._arguments[self.ARG_TRAIN_VARIANT]
        self.userProfiles:dict = {}

    def __getTrainVariant(self, trainVariant:str, trainDF:DataFrame):
        if trainVariant == "all":
            return trainDF
        elif trainVariant == "positive":
            return trainDF.loc[trainDF[Ratings.COL_RATING] >= 4]
        elif trainVariant == "posneg":
            return trainDF

    def train(self, history:AHistory, ratingsDF:DataFrame, usersDF:DataFrame, itemsDF:DataFrame):
        if not isinstance(history, AHistory):
            raise ValueError("Argument history isn't type AHistory.")
        if type(ratingsDF) is not DataFrame:
            raise ValueError("Argument ratingsDF isn't type DataFrame.")
        if type(usersDF) is not DataFrame:
            raise ValueError("Argument usersDF isn't type DataFrame.")
        if type(itemsDF) is not DataFrame:
            raise ValueError("Argument itemsDF isn't type DataFrame.")

        t:DataFrame = self.__getTrainVariant(self.trainVariant, ratingsDF)
        t[Ratings.COL_MOVIEID] = t[Ratings.COL_MOVIEID].astype("str")
        t_sequences:DataFrame = t.groupby(Ratings.COL_USERID)[Ratings.COL_MOVIEID].apply(" ".join)
        if self.DEBUG_MODE:
            print(t_sequences)
        # t_sequences.set_index(Ratings.COL_USERID, inplace=True)
        w2vTrainData:List[str] = t_sequences.values.tolist()

        e:int = 64
        w:int = 3

        datasetId:str = self._arguments[self.ARG_DATASET_ID]
        #print("datasetId: " + str(datasetId))

        self.model = self.__load_obj("model", datasetId, self.trainVariant, e, w)
        self.dictionary = self.__load_obj("dictionary", datasetId, self.trainVariant, e, w)
        self.rev_dict = self.__load_obj("rev_dict", datasetId, self.trainVariant, e, w)

        if self.model is None:
            model, rev_dict, dictionary = word2vec.word2vecRun(w, e, w2vTrainData)
            dictionary = dict([((int(i), j) if i != "RARE" else (-1, j)) for i, j in dictionary.items()])
            rev_dict = dict(zip(dictionary.values(), dictionary.keys()))
            self.__save_obj(model, "model", datasetId, self.trainVariant, e, w)
            self.__save_obj(dictionary, "dictionary", datasetId, self.trainVariant, e, w)
            self.__save_obj(rev_dict, "rev_dict", datasetId, self.trainVariant, e, w)
            self.model = model
            self.dictionary = dictionary
            self.rev_dict = rev_dict

        # ratingsSum:Dataframe<(userId:int, movieId:int, ratings:int, timestamp:int)>

        self.ratingsGroupDF = t.groupby(Ratings.COL_USERID)[Ratings.COL_MOVIEID]
        userProfileDF = self.ratingsGroupDF.aggregate(lambda x: list(x))
        self.userProfiles = userProfileDF.to_dict()

    def update(self, ratingsUpdateDF:DataFrame):
        # ratingsUpdateDF has only one row
        ratingsUpdateDF:DataFrame = self.__getTrainVariant(self.trainVariant, ratingsUpdateDF)
        if ratingsUpdateDF.shape[0] > 0:
            row = ratingsUpdateDF.iloc[0]
            rating = row[Ratings.COL_RATING]
            userID = row[Ratings.COL_USERID]
            objectID = row[Ratings.COL_MOVIEID]
            userTrainData = self.userProfiles.get(userID, [])
            userTrainData.append(objectID)
            self.userProfiles[userID] = userTrainData

    def __resolveUserProfile(self, userProfileStrategy:str, userTrainData):
        objectIDs:List[int] = [int(i) for i in userTrainData]
        w2vObjects = [self.dictionary[i] for i in objectIDs if i in self.dictionary]

        rec:str = userProfileStrategy
        if self.DEBUG_MODE:
            print(rec)
        if (len(w2vObjects) > 0):
            if (rec == "mean") | (rec == "max"):
                weights = [1.0] * len(w2vObjects)
            elif rec == "last":
                w2vObjects = w2vObjects[-1:]
                weights = [1.0]
            elif rec == "window3":
                w2vObjects = w2vObjects[-3:]
                weights = [1 / len(w2vObjects) * i for i in range(1, (len(w2vObjects) + 1))]
            elif rec == "window5":
                w2vObjects = w2vObjects[-5:]
                weights = [1 / len(w2vObjects) * i for i in range(1, (len(w2vObjects) + 1))]
            elif rec == "window10":
                w2vObjects = w2vObjects[-10:]
                weights = [1 / len(w2vObjects) * i for i in range(1, (len(w2vObjects) + 1))]

            if rec == "max":
                agg = np.max
            else:
                agg = np.mean

            if self.DEBUG_MODE:
                print((w2vObjects, weights, agg))
            return (w2vObjects, weights, agg)

        return ([], [], "")


    def recommend(self, userID:int, numberOfItems:int, argumentsDict:dict):
        #print("userID: " + str(userID))
        if type(userID) is not int and type(userID) is not np.int64:
            raise ValueError("Argument userID isn't type int.")
        if type(numberOfItems) is not int and type(numberOfItems) is not np.int64:
            raise ValueError("Argument numberOfItems isn't type int.")
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")

        userProfileStrategy:str = argumentsDict[self.ARG_USER_PROFILE_STRATEGY]
        #print("userProfileStrategy: " + str(userProfileStrategy))

        userTrainData = self.userProfiles.get(userID, [])
        w2vObjects, weights, aggregation = self.__resolveUserProfile(userProfileStrategy, userTrainData)
        simList:List = []

        # provedu agregaci dle zvolené metody
        if len(w2vObjects) > 0:
            embeds = self.model[w2vObjects]
            results = 1 - pairwise_distances(embeds, self.model, metric="cosine")

            weights = np.asarray(weights).reshape((-1, 1))
            results = results * weights
            results = aggregation(results, axis=0)

            if self.DEBUG_MODE:
                print(type(results))
            # check for a variant with negative preference (only positive objects recommended)
            # approximative solution - might result in less objects
            resultList = (-results).argsort()[0:(numberOfItems * 3)]
            resultingOIDs = [self.rev_dict[i] for i in resultList if self.rev_dict[i] > 0]
            resultingOIDs = resultingOIDs[0:numberOfItems]
            resultList = [self.dictionary[i] for i in resultingOIDs]

            # print(results[resultList])

            # normalize scores into the unit vector (for aggregation purposes)
            # !!! tohle je zasadni a je potreba provest normalizaci u vsech recommenderu - teda i pro most popular!
            finalScores = results[resultList]
            finalScores = normalize(np.expand_dims(finalScores, axis=0))[0, :]

            return pd.Series(finalScores.tolist(), index=resultingOIDs)

        return pd.Series([], index=[])

    def __save_obj(self, obj, name:str, datasetId:str, trainVariant:str, e:int, w:int):
        fileName:str = Configuration.modelDirectory + os.sep + name + "_{0}_{1}_{2}_{3}".format(datasetId, trainVariant, e, w)+ '.pkl'
        with open(fileName, 'wb') as f:
            pickle.dump(obj, f)

    def __load_obj(self, name:str, datasetId:str, trainVariant:str, e:int, w:int):
        fileName:str = Configuration.modelDirectory + os.sep + name + "_{0}_{1}_{2}_{3}".format(datasetId, trainVariant, e, w)+ '.pkl'
        print(fileName)
        if not os.path.isfile(fileName):
            return None
        with open(fileName, 'rb') as f:
            return pickle.load(f)