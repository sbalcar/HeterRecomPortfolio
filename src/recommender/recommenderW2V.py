#!/usr/bin/python3

import pickle
import os

import pandas as pd
import numpy as np

from configuration.configuration import Configuration #class

from typing import List
from typing import Dict

from datasets.aDataset import ADataset #class
from datasets.datasetML import DatasetML #class
from datasets.datasetRetailrocket import DatasetRetailRocket #class
from datasets.datasetST import DatasetST #class

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
    ARG_VECTOR_SIZE:str = "vectorSize"
    ARG_WINDOW_SIZE:str = "windowSize"
    ARG_ITERATIONS:str = "iterations"

    # mandatory argument
    ARG_USER_PROFILE_STRATEGY:str = "userProfileStrategy"
    ARG_USER_PROFILE_SIZE:str = "userProfileSize"

    DEBUG_MODE = False

    # ratingsSum:Dataframe<(userId:int, movieId:int, ratings:int, timestamp:int)>
    def __init__(self, jobID:str, argumentsDict:Dict[str,object]):
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict is not type dict.")
        self._jobID:str = jobID
        self._arguments:dict = argumentsDict

        self.trainVariant:str = self._arguments[self.ARG_TRAIN_VARIANT]
        self.vectorSize:int = self._arguments[self.ARG_VECTOR_SIZE]
        self.windowSize:int = self._arguments[self.ARG_WINDOW_SIZE]
        self.iterations:int = self._arguments[self.ARG_ITERATIONS]

        self.userProfiles:dict = {}

    def __getTrainVariant(self, trainVariant:str, trainDF:DataFrame):
        if trainVariant == "all":
            return trainDF
        elif trainVariant == "positive":
            return trainDF.loc[trainDF[Ratings.COL_RATING] >= 4]
        elif trainVariant == "posneg":
            return trainDF
    
    def train(self, history:AHistory, dataset:ADataset):
        if not isinstance(history, AHistory):
            raise ValueError("Argument history isn't type AHistory.")
        if not isinstance(dataset, ADataset):
            raise ValueError("Argument dataset isn't type ADataset.")
        self._trainDataset = dataset

        if type(dataset) is DatasetML:
            trainRatingsDF:DataFrame = dataset.ratingsDF
            COL_USERID:str = Ratings.COL_USERID
            COL_ITEMID:str = Ratings.COL_MOVIEID

        elif type(dataset) is DatasetRetailRocket:
            from datasets.retailrocket.events import Events  # class
            trainRatingsDF:DataFrame = dataset.eventsDF
            COL_USERID:str = Events.COL_VISITOR_ID
            COL_ITEMID:str = Events.COL_ITEM_ID

        elif type(self._trainDataset) is DatasetST:
            from datasets.slantour.events import Events  # class
            trainRatingsDF:DataFrame = dataset.eventsDF
            COL_USERID:str = Events.COL_USER_ID
            COL_ITEMID:str = Events.COL_OBJECT_ID

        t:DataFrame = self.__getTrainVariant(self.trainVariant, trainRatingsDF)
        t[COL_ITEMID] = t[COL_ITEMID].astype("str")
        t_sequences:DataFrame = t.groupby(COL_USERID)[COL_ITEMID].apply(" ".join)

        if self.DEBUG_MODE:
            print(t_sequences)
        # t_sequences.set_index(Ratings.COL_USERID, inplace=True)
        w2vTrainData:List[str] = t_sequences.values.tolist()

        e:int = self.vectorSize #64
        w:int = self.windowSize #3
        i:int = self.iterations #100000

        #datasetId = "ml1mDiv90"
        self.model = self.__load_obj("model", dataset.datasetID, self.trainVariant, e, w, i)
        self.dictionary = self.__load_obj("dictionary", dataset.datasetID, self.trainVariant, e, w, i)
        self.rev_dict = self.__load_obj("rev_dict", dataset.datasetID, self.trainVariant, e, w, i)

        if self.model is None or self.dictionary is None or self.rev_dict is None:
            model, rev_dict, dictionary = word2vec.word2vecRun(w, e, i, w2vTrainData)
            dictionary = dict([((int(i), j) if i != "RARE" else (-1, j)) for i, j in dictionary.items()])
            rev_dict = dict(zip(dictionary.values(), dictionary.keys()))
            self.__save_obj(model, "model", dataset.datasetID, self.trainVariant, e, w, i )
            self.__save_obj(dictionary, "dictionary", dataset.datasetID, self.trainVariant, e, w, i)
            self.__save_obj(rev_dict, "rev_dict", dataset.datasetID, self.trainVariant, e, w, i)
            self.model = model
            self.dictionary = dictionary
            self.rev_dict = rev_dict

        self.ratingsGroupDF = t.groupby(COL_USERID)[COL_ITEMID]
        userProfileDF:DataFrame = self.ratingsGroupDF.aggregate(lambda x: list(x))
        self.userProfiles = userProfileDF.to_dict()

    def update(self, ratingsUpdateDF:DataFrame, argumentsDict:Dict[str,object]):
        if type(ratingsUpdateDF) is not DataFrame:
            raise ValueError("Argument ratingsTrainDF isn't type DataFrame.")
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")

        if type(self._trainDataset) is DatasetML:
            COL_USERID:str = Ratings.COL_USERID
            COL_ITEMID:str = Ratings.COL_MOVIEID

        elif type(self._trainDataset) is DatasetRetailRocket:
            from datasets.retailrocket.events import Events  # class
            COL_USERID:str = Events.COL_VISITOR_ID
            COL_ITEMID:str = Events.COL_ITEM_ID

        elif type(self._trainDataset) is DatasetST:
            from datasets.slantour.events import Events  # class
            COL_USERID:str = Events.COL_USER_ID
            COL_ITEMID:str = Events.COL_OBJECT_ID

        # ratingsUpdateDF has only one row
        ratingsUpdateDF:DataFrame = self.__getTrainVariant(self.trainVariant, ratingsUpdateDF)
        if ratingsUpdateDF.shape[0] > 0:
            row = ratingsUpdateDF.iloc[0]
            userID:int = row[COL_USERID]
            objectID:int = row[COL_ITEMID]
            userTrainData:List[int] = self.userProfiles.get(userID, [])
            userTrainData.append(objectID)
            self.userProfiles[userID] = userTrainData

    def __resolveUserProfile(self, userProfileStrategy:str,userProfileSize:int, userTrainData:List[int]):
        objectIDs:List[int] = [int(i) for i in userTrainData]
        w2vObjects = [self.dictionary[i] for i in objectIDs if i in self.dictionary]

        rec:str = userProfileStrategy
        if self.DEBUG_MODE:
            print(rec)

        if (len(userTrainData) > 0):
            if(userProfileSize > 0 ):
                val = -1 * userProfileSize    
                w2vObjects = w2vObjects[val:]      
                  
            if (rec == "mean") | (rec == "max"):
                weights = [1.0] * len(w2vObjects)
            elif rec == "weightedMean":
                weights = [1 / len(w2vObjects) * i for i in range(1, (len(w2vObjects) + 1))]

            if rec == "max":
                agg = np.max
            else:
                agg = np.mean

            return (w2vObjects, weights, agg)            

        return ([], [], "")     
        

    def recommend(self, userID:int, numberOfItems:int, argumentsDict:Dict[str,object]):
        #print("userID: " + str(userID))
        if type(userID) is not int and type(userID) is not np.int64:
            raise ValueError("Argument userID isn't type int.")
        if type(numberOfItems) is not int and type(numberOfItems) is not np.int64:
            raise ValueError("Argument numberOfItems isn't type int.")
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")

        userProfileStrategy:str = argumentsDict[self.ARG_USER_PROFILE_STRATEGY]
        userProfileSize:str = argumentsDict[self.ARG_USER_PROFILE_SIZE]

        userTrainData = self.userProfiles.get(userID, [])
        
        #adding currently viewed item (if any) into the user profile
        itemID = argumentsDict.get("itemID", 0)
        if itemID > 0:
           userTrainData.append(itemID)
        
        w2vObjects, weights, aggregation = self.__resolveUserProfile(userProfileStrategy, userProfileSize, userTrainData)

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
            resultList = (-results).argsort()[0:(numberOfItems * 10)]
            resultingOIDs = [self.rev_dict[i] for i in resultList if self.rev_dict[i] > 0]
            
            if argumentsDict.get(self.ARG_ALLOWED_ITEMIDS) is not None:
                # ARG_ALLOWED_ITEMIDS contains a list of allowed IDs
                # TODO check type of ARG_ALLOWED_ITEMIDS, should be list
                resultingOIDs = [key for key in resultingOIDs if key in argumentsDict[self.ARG_ALLOWED_ITEMIDS]]          
            
            resultingOIDs = resultingOIDs[0:numberOfItems]
            resultList = [self.dictionary[i] for i in resultingOIDs]

            # print(results[resultList])

            # normalize scores into the unit vector (for aggregation purposes)
            # !!! tohle je zasadni a je potreba provest normalizaci u vsech recommenderu - teda i pro most popular!
            finalScores = results[resultList]
            finalScores = normalize(np.expand_dims(finalScores, axis=0))[0, :]

            return pd.Series(finalScores.tolist(), index=resultingOIDs)

        return pd.Series([], index=[])

    def __save_obj(self, obj, name:str, datasetId:str, trainVariant:str, e:int, w:int, i:int):
        fileName:str = Configuration.modelDirectory + os.sep + name + "_{0}_{1}_{2}_{3}_{4}".format(datasetId, trainVariant, e, w, i)+ '.pkl'
        print("saveObject: " + str(fileName))
        with open(fileName, 'wb') as f:
            pickle.dump(obj, f)

    def __load_obj(self, name:str, datasetId:str, trainVariant:str, e:int, w:int, i:int):
        fileName:str = Configuration.modelDirectory + os.sep + name + "_{0}_{1}_{2}_{3}_{4}".format(datasetId, trainVariant, e, w, i)+ '.pkl'
        print("loadObject: " + str(fileName))
        if not os.path.isfile(fileName):
            return None
        with open(fileName, 'rb') as f:
            return pickle.load(f)