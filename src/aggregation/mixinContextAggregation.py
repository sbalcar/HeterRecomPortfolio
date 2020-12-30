#!/usr/bin/python3

import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import PolynomialFeatures

from typing import List

from pandas.core.frame import DataFrame  # class
from pandas.core.series import Series  # class

from aggregation.aAggregation import AAgregation # class

from history.aHistory import AHistory  # class
from datasets.ml.ratings import Ratings  # class


class MixinContextAggregation(AAgregation):
    ARG_SELECTOR: str = "selector"
    ARG_ITEMS: str = "items"
    ARG_USERS: str = "users"
    ARG_DATASET: str = "dataset"
    SIZE_OF_USER_CLICKING_HISTORY = 20

    dictOfGenreIndexes: dict = {"Action": 0, "Adventure": 1, "Animation": 2, "Children's": 3, "Comedy": 4, "Crime": 5,
                                "Documentary": 6, "Drama": 7, "Fantasy": 8, "Film-Noir": 9, "Horror": 10,
                                "Musical": 11, "Mystery": 12, "Romance": 13, "Sci-Fi": 14, "Thriller": 15, "War": 16,
                                "Western": 17}

    def __init__(self, history: AHistory, argumentsDict: dict):
        if not isinstance(history, AHistory):
            raise ValueError("Argument history isn't type AHistory.")
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")

        self._selector = argumentsDict[self.ARG_SELECTOR]
        # what was recommended in previous iteration (key is userID, value is dict of recommenders and their recommendations)
        self._listOfPreviousRecommendations = dict()
        # what was clicked by user (key is userID, value is list of clicked items)
        self._listOfPreviousClickedItems = dict()
        self._contextDim: int = 0
        self._b: dict = None
        self._A: dict = None
        self._context = None
        self._inverseA: dict = None
        self._INVERSE_CALCULATION_THRESHOLD: int = 100
        self._inverseCounter: int = 0
        self.dataset_name: str = argumentsDict[self.ARG_DATASET]
        self.items: DataFrame = self._preprocessItems(argumentsDict[self.ARG_ITEMS])
        self.users: DataFrame = self._preprocessUsers(argumentsDict[self.ARG_USERS])

    def _preprocessUsers(self, users:DataFrame):
        if self.dataset_name == "ml":
            return self._onehotUsersOccupationML(users)
        else:
            raise ValueError("Dataset " + self.dataset_name + " is not supported!")

    def _onehotUsersOccupationML(self, users:DataFrame):
        one_hot_encoding = pd.get_dummies(users['occupation'])
        users.drop(['occupation'], axis=1, inplace=True)
        return pd.concat([users, one_hot_encoding], axis=1)

    def _preprocessItems(self, items:DataFrame):
        if self.dataset_name == "ml":
            return self._onehotItemsGenresML(items)
        else:
            raise ValueError("Dataset " + self.dataset_name + " is not supported!")

    def _onehotItemsGenresML(self, items:DataFrame):
        one_hot_encoding = items["Genres"].str.get_dummies(sep='|')
        one_hot_encoding.drop(one_hot_encoding.columns[0],axis=1, inplace=True)
        tmp = items.drop(['Genres'], axis=1, inplace=False)
        return pd.concat([tmp, one_hot_encoding], axis=1)

    # methodsResultDict:{String:pd.Series(rating:float[], itemID:int[])},
    # modelDF:pd.DataFrame[numberOfVotes:int], numberOfItems:int
    def run(self, methodsResultDict: dict, modelDF: DataFrame, userID: int, numberOfItems: int = 20):
        # TODO: CHECK DATA INTEGRITY!

        if self._listOfPreviousClickedItems.get(userID) is None:
            self._listOfPreviousClickedItems[userID] = []
        # initialize A and b if not done already
        if self._b is None or self._A is None:
            # update context
            self._context = self._calculateContext(userID)

            self._b: dict = {}
            self._A: dict = {}
            self._inverseA = {}
            for recommender, value in methodsResultDict.items():
                self._b[recommender] = np.zeros(self._contextDim)
                self._A[recommender] = np.identity(self._contextDim)
                self._inverseA[recommender] = np.identity(self._contextDim)

        # else update b's
        else:
            listOfClickedItems = self._getListOfPreviousClickedItems(userID)[-20:]

            if self._listOfPreviousRecommendations.get(userID) is None:
                self._listOfPreviousRecommendations[userID] = dict()

            # calculate reward for each recommender based on previous recommendation
            dictOfRewards: dict = {}
            counter = 0
            for recommender, value in self._listOfPreviousRecommendations[userID].items():
                # TODO: Is performance OK here? This is just sketch
                # calculate size of intersection between listOfClickedItems and self._listOfPreviousRecommendations
                succesfulRecomendations = len(list(set(listOfClickedItems)
                                                   .intersection(value.index)))

                # give recommender 'points' based on the size of intersection
                # TODO: I don't take into account previous relevance or previous votes, maybe needs to be improved in future
                dictOfRewards[recommender] = succesfulRecomendations
                counter += 1

            # update b's
            if len(dictOfRewards) > 0:
                for recommender, value in self._b.items():
                    reward = dictOfRewards[recommender] / counter
                    self._b[recommender] = self._b[recommender] + (reward * self._context)

            # update context for next iteration (we had to keep previous context because we needed to update b's)
            self._context = self._calculateContext(userID)

        # check data integrity - if contextDim is OK
        for name, value in methodsResultDict.items():
            if (self._A is not None and self._contextDim != len(self._A[name])) or \
                    (self._b is not None and self._contextDim != len(self._b[name])):
                raise ValueError("Context dimension should be same in every iteration!")

        # update recommender's votes
        updatedVotes = dict()
        totalOriginalVotes = 0
        totalUpdatedVotes = 0
        for recommender, votes in modelDF.iterrows():
            # Calculate change rate
            ridgeRegression = self._inverseA[recommender].dot(self._b[recommender])
            UCB = self._context.T.dot(self._inverseA[recommender]).dot(self._context)
            change_rate = (ridgeRegression.T.dot(self._context) + math.sqrt(UCB))

            # update votes
            updatedVotes[recommender] = change_rate * votes['votes']
            totalOriginalVotes += modelDF.at[recommender, 'votes']
            totalUpdatedVotes += updatedVotes[recommender]

        # normalize updated votes and save it to modelDF
        rate = (totalOriginalVotes / totalUpdatedVotes) + 1e-12     # totalOriginalVottes converges to 0 so for now we add
                                                                    # 1e-12 to the rate to avoid this
        for recommender, votes in modelDF.iterrows():
            modelDF.at[recommender, 'votes'] = updatedVotes[recommender] * rate

        itemsWithResposibilityOfRecommenders: List[int, np.Series[int, str]] = \
            super().run(methodsResultDict, modelDF, userID, numberOfItems=numberOfItems)

        # update A's
        for recommender, value in self._A.items():

            # get relevance of items, which were recommended by recommender and are in itemsWithResposibilityOfRecommenders
            relevanceSum = 0
            for recommendedItemID in itemsWithResposibilityOfRecommenders:
                if recommendedItemID in methodsResultDict[recommender].index:
                    relevanceSum += methodsResultDict[recommender][recommendedItemID]
            self._A[recommender] += self._context.dot(self._context.T) * relevanceSum

        # recompute inverse A's if threshold is hit
        if self._inverseCounter > self._INVERSE_CALCULATION_THRESHOLD:
            for recommender, value in self._inverseA.items():
                self._inverseA[recommender] = np.linalg.inv(self._A[recommender])
            self._inverseCounter = 0
        self._inverseCounter += 1
        self._listOfPreviousRecommendations[userID] = methodsResultDict

        # set clicked items to None (because next time there could be different userID)
        return itemsWithResposibilityOfRecommenders

    def _getListOfPreviousClickedItems(self, userID: int):
        if self.dataset_name == "ml":
            return self._getListOfPreviousClickedItemsML(userID)
        else:
            raise ValueError("Dataset " + self.dataset_name + " is not supported!")

    def _getListOfPreviousClickedItemsML(self, userID: int):
        # for now, return whole list of every clicked item (calculated already in context)
        return self._listOfPreviousClickedItems[userID]

    def _calculateContext(self, userID):
        if self.dataset_name == "ml":
            return self._calculateContextML(userID)
        else:
            raise ValueError("Dataset " + self.dataset_name + " is not supported!")

    def _calculateContextML(self, userID):

        # get user data
        user = self.users.loc[self.users['userId'] == userID]

        # init result
        result = np.zeros(2)

        # add seniority of user into the context (filter only clicked items)
        previousClickedItemsOfUser = self._getListOfPreviousClickedItems(userID)
        historySize = len(previousClickedItemsOfUser)
        if historySize < 3:
            result[0] = 1
        elif historySize < 5:
            result[0] = 2
        elif historySize < 10:
            result[0] = 3
        elif historySize < 30:
            result[0] = 4
        elif historySize < 50:
            result[0] = 5
        else:
            result[0] = 6

        # add log to the base 2 of historySize to context
        if historySize != 0:
            result[1] = math.log(historySize, 2)
        else:
            result[1] = -1

        # get last 20 movies from user and aggregate their genres
        last20MoviesList = previousClickedItemsOfUser[-20:]

        # aggregation
        itemsIDs = last20MoviesList
        items = self.items.iloc[list(itemsIDs)]
        itemsGenres = items.drop(items.columns[[0,1]], axis=1).sum()
        result = np.append(result, itemsGenres)

        # create polynomial features from [seniority]*[genres]
        poly = PolynomialFeatures(2)
        result = poly.fit_transform(result.reshape(-1, 1))
        result = result.flatten()

        # add user gender to the context
        userFeatures = []
        userFeatures.append(1 if user['gender'].item() == 'F' else -1)

        # append age and onehot occupation
        tmp = user.T.drop(labels=['userId', 'gender', 'zipCode']).to_numpy().flatten()
        userFeatures = np.concatenate([userFeatures, tmp])

        result = np.concatenate([result, userFeatures])

        # adjust context dimension attribute
        self._contextDim = len(result)

        return result

    def update(self, ratingsUpdateDF:DataFrame):
        if type(ratingsUpdateDF) is not DataFrame:
            raise ValueError("Argument ratingsTrainDF isn't type DataFrame.")

        row: DataFrame = ratingsUpdateDF.iloc[0]

        userID: int = row[Ratings.COL_USERID]
        objectID: int = row[Ratings.COL_MOVIEID]

        if self._listOfPreviousClickedItems.get(userID) is None:
            self._listOfPreviousClickedItems[userID] = [objectID]
        else:
            self._listOfPreviousClickedItems[userID].append(objectID)
            if len(self._listOfPreviousClickedItems[userID]) > self.SIZE_OF_USER_CLICKING_HISTORY:
                self._listOfPreviousClickedItems[userID].pop(0)



    # methodsResultDict:{String:Series(rating:float[], itemID:int[])},
    # modelDF:DataFrame<(methodID:str, votes:int)>, numberOfItems:int
    def runWithResponsibility(self, methodsResultDict: dict, modelDF: DataFrame, userID: int, numberOfItems: int = 20):
        # TODO: CHECK DATA INTEGRITY!
        # methodsResultNewDict: dict[str, pd.Series] = self._penaltyTool.runPenalization(
        #        userID, methodsResultDict, self._history)

        itemsWithResposibilityOfRecommenders: List[int, Series[int, str]] = super().runWithResponsibility(
            methodsResultDict, modelDF, userID, numberOfItems=numberOfItems)

        return itemsWithResposibilityOfRecommenders
