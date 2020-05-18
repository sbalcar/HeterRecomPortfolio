#!/usr/bin/python3

import pandas as pd
import numpy as np
from recommender.w2v import word2vec

from pandas.core.frame import DataFrame #class

from sklearn.metrics import *
from sklearn.preprocessing import normalize
from recommender.aRecommender import ARecommender  # class

from datasets.ratings import Ratings  # class


class RecommenderW2V(ARecommender):

    ARG_TRAIN_VARIANT:str = "trainVariant"

    # ratingsSum:Dataframe<(userId:int, movieId:int, ratings:int, timestamp:int)>
    def __init__(self, argumentsDict):
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict is not type dict.")
        self._arguments:dict = argumentsDict

        self.trainVariant = self._arguments[self.ARG_TRAIN_VARIANT]
        self.userProfiles = {}

    def getTrainVariant(self, trainDF:DataFrame):
        if self.trainVariant == "all":
            return trainDF
        elif self.trainVariant == "positive":
            return trainDF.loc[trainDF[Ratings.COL_RATING] >= 4]
        elif self.trainVariant == "posneg":
            trainDF[Ratings.COL_MOVIEID].loc[trainDF[Ratings.COL_RATING] < 4] = -trainDF[Ratings.COL_MOVIEID]
            return trainDF

    def train(self, historyDF:DataFrame, ratingsDF:DataFrame, usersDF:DataFrame, itemsDF:DataFrame):
        if type(ratingsDF) is not DataFrame:
            raise ValueError("Argument ratingsDF is not type DataFrame.")

        t = self.getTrainVariant(ratingsDF)
        t[Ratings.COL_MOVIEID] = t[Ratings.COL_MOVIEID].astype("str")
        t_sequences = t.groupby(Ratings.COL_USERID)[Ratings.COL_MOVIEID].apply(" ".join)
        print(t_sequences)
        # t_sequences.set_index(Ratings.COL_USERID, inplace=True)
        w2vTrainData = t_sequences.values.tolist()

        w = 3
        e = 64
        model, rev_dict, dictionary = word2vec.word2vecRun(w, e, w2vTrainData)
        dictionary = dict([((int(i), j) if i != "RARE" else (-1, j)) for i, j in dictionary.items()])
        rev_dict = dict(zip(dictionary.values(), dictionary.keys()))
        self.model = model
        self.dictionary = dictionary
        self.rev_dict = rev_dict

        # ratingsSum:Dataframe<(userId:int, movieId:int, ratings:int, timestamp:int)>

        self.ratingsGroupDF = t.groupby(Ratings.COL_USERID)[Ratings.COL_MOVIEID]
        userProfileDF = self.ratingsGroupDF.aggregate(lambda x: list(x))
        self.userProfiles = userProfileDF.to_dict()

    def update(self, ratingsUpdateDF:DataFrame):
        # ratingsUpdateDF has only one row
        ratingsUpdateDF = self.getTrainVariant(ratingsUpdateDF)
        if ratingsUpdateDF.shape[0] > 0:
            row = ratingsUpdateDF.iloc[0]
            rating = row[Ratings.COL_RATING]
            userID = row[Ratings.COL_USERID]
            objectID = row[Ratings.COL_MOVIEID]
            userTrainData = self.userProfiles.get(userID, [])
            userTrainData.append(objectID)
            self.userProfiles[userID] = userTrainData

    def resolveUserProfile(self, userProfileStrategy, userTrainData):
        objectIDs = [int(i) for i in userTrainData]
        w2vObjects = [self.dictionary[i] for i in objectIDs if i in self.dictionary]

        rec = userProfileStrategy
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

            print((w2vObjects, weights, agg))
            return (w2vObjects, weights, agg)

        return ([], [], "")

    def recommend(self, userID:int, numberOfItems:int, userProfileStrategy:str):
        userTrainData = self.userProfiles.get(userID, [])
        w2vObjects, weights, aggregation = self.resolveUserProfile(userProfileStrategy, userTrainData)
        simList = []

        # provedu agregaci dle zvolené metody
        if len(w2vObjects) > 0:
            embeds = self.model[w2vObjects]
            results = 1 - pairwise_distances(embeds, self.model, metric="cosine")

            weights = np.asarray(weights).reshape((-1, 1))
            results = results * weights
            results = aggregation(results, axis=0)

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

            return pd.Series(resultingOIDs, finalScores.tolist(), name="rating")

        return pd.Series([], [],name="rating")