#!/usr/bin/python3

import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import PolynomialFeatures

from typing import List

from pandas.core.frame import DataFrame #class
from pandas.core.series import Series #class

from aggregation.aggrFuzzyDHondt import AggrFuzzyDHondt #class

from history.aHistory import AHistory #class


class AggrContextFuzzyDHondt(AggrFuzzyDHondt):

    ARG_SELECTOR:str = "selector"
    ARG_ITEMS:str = "items"
    ARG_USERS:str = "users"
    dictOfGenreIndexes: dict = {"Action": 0, "Adventure": 1, "Animation": 2, "Children's": 3, "Comedy": 4, "Crime": 5,
                                "Documentary": 6, "Drama": 7, "Fantasy": 8, "Film-Noir": 9, "Horror": 10,
                                "Musical": 11, "Mystery": 12, "Romance": 13, "Sci-Fi": 14, "Thriller": 15, "War": 16,
                                "Western": 17}

    def __init__(self, history:AHistory, argumentsDict:dict):
        if not isinstance(history, AHistory):
            raise ValueError("Argument history isn't type AHistory.")
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")

        self._selector = argumentsDict[self.ARG_SELECTOR]
        self._history = history
        self._listOfPreviousRecommendations = None
        self._contextDim:int = 21
        self._b:dict = None
        self._A:dict = None
        self._context = None
        self._inverseA:dict = None
        self._INVERSE_CALCULATION_THRESHOLD: int = 100
        self._inverseCounter = 0
        self._lastTimestamp:dict = {}
        self.items = argumentsDict[self.ARG_ITEMS]
        self.users = argumentsDict[self.ARG_USERS]

    # methodsResultDict:{String:pd.Series(rating:float[], itemID:int[])},
    # modelDF:pd.DataFrame[numberOfVotes:int], numberOfItems:int
    def run(self, methodsResultDict:dict, modelDF:DataFrame, userID:int, numberOfItems:int=20):
        # TODO: CHECK DATA INTEGRITY!

        # create lastTimestamp for new user
        if userID not in self._lastTimestamp:
            self._lastTimestamp[userID] = 0

        # update context
        self._context = self.__calculateContext(userID)

        # check contextDim is OK
        if self._contextDim != len(self._A) or self._contextDim != len(self._b):
            raise ValueError("Context dimension should be same in every iteration!")

        # initialize A and b if not done already
        if self._b is None or self._A is None or self._context is None:
            self._b: dict = {}
            self._A: dict = {}
            self._inverseA = {}
            for recommender, value in methodsResultDict.items():
                self._b[recommender] = np.zeros(self._contextDim)
                self._A[recommender] = np.identity(self._contextDim)
                self._inverseA[recommender] = np.identity(self._contextDim)

        # else update b's
        else:
            ListOfClickedItems = self.__getListOfPreviousClickedItems(userID)

            if self._listOfPreviousRecommendations is None:
                raise ValueError("self._listOfPreviousRecommendations has to contain previous recommendations!")
            dictOfRewards:dict = {}
            counter = 0
            for recommender, value in methodsResultDict.items():
                # TODO: Is performance OK here? This is just sketch
                # calculate size of intersection between ListOfClickedItems and self._listOfPreviousRecommendations
                succesfulRecomendations = len(list(set(ListOfClickedItems).intersection(self._listOfPreviousRecommendations)))

                # give recommender 'points' based on the size of intersection
                dictOfRewards[recommender] = succesfulRecomendations
                counter += 1
                # TODO: I don't take into account previous relevance or previous votes, maybe needs to be improved in future

            # update b's
            for recommender, value in self._b.items():
                reward = dictOfRewards[recommender] / counter
                self._b[recommender] += reward * self._context

        # update recommender's votes
        for recommender, votes in modelDF.iterrows():

            # Calculate change rate
            ridgeRegression = self._inverseA[recommender].dot(self._b[recommender])
            UCB = self._context.T.dot(self._inverseA[recommender]).dot(self._context)
            change_rate = ridgeRegression.T.dot(self._context) + math.sqrt(UCB)

            # update votes
            modelDF.at[recommender, 'votes'] = change_rate * votes['votes']

        itemsWithResposibilityOfRecommenders:List[int,np.Series[int,str]] =\
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
        self._listOfPreviousRecommendations = itemsWithResposibilityOfRecommenders
        return itemsWithResposibilityOfRecommenders

    def __getListOfPreviousClickedItems(self, userID:int):

        # get list of history with user
        previousItemsOfUser = self._history.getPreviousRecomOfUser(userID)

        # if we have no history about the user
        if len(previousItemsOfUser) == 0:
            return list()

        # data indexes in previousItemsOfUser
        ITEM_INDEX = 2
        CLICKED_INDEX = 5
        TIMESTAMP_INDEX = 6

        # sort history by timestamp
        sortedItemsByTimestamp = sorted(previousItemsOfUser, key=lambda x: x[TIMESTAMP_INDEX], reverse=True)

        # get newest timestamp
        newLastTimestamp = sortedItemsByTimestamp[0][TIMESTAMP_INDEX]

        # check data integrity of lastTimestamp
        if userID not in self._lastTimestamp:
            raise ValueError("Timestamp has to be initialized for each user!")
        # if there is no new history from user
        if self._lastTimestamp[userID] == newLastTimestamp:
            return list()

        # get last interactions with the user (iteratively based on history timestamps)
        iterator = 0
        result = list()
        while iterator < len(sortedItemsByTimestamp) and sortedItemsByTimestamp[iterator][TIMESTAMP_INDEX] > self._lastTimestamp[userID]:
            if sortedItemsByTimestamp[iterator][CLICKED_INDEX] and sortedItemsByTimestamp[iterator][ITEM_INDEX] not in result:
                result.append(sortedItemsByTimestamp[iterator][ITEM_INDEX])
            iterator += 1

        self._lastTimestamp[userID] = newLastTimestamp

        return result

    def __calculateContext(self, userID):
        # get user data
        user = self.users.iloc[userID]

        # init result
        result = np.zeros(len(self.dictOfGenreIndexes) + 1)

        # add seniority of user into the context (filter only clicked items)
        CLICKED_INDEX = 5
        previousClickedItemsOfUser = list(filter(lambda x: x[CLICKED_INDEX], self._history.getPreviousRecomOfUser(userID)))
        historySize = len(previousClickedItemsOfUser)
        if historySize < 10:
            result[len(self.dictOfGenreIndexes)] = 1
        elif historySize < 30:
            result[len(self.dictOfGenreIndexes)] = 2
        elif historySize < 50:
            result[len(self.dictOfGenreIndexes)] = 3
        else:
            result[len(self.dictOfGenreIndexes)] = 4

        # get last 20 movies from user and aggregate their genres
        TIMESTAMP_INDEX = 6
        sortedItemsByTimestamp = sorted(previousClickedItemsOfUser, key=lambda x: x[TIMESTAMP_INDEX], reverse=True)
        last20MoviesList = sortedItemsByTimestamp[:20]

        # aggregation
        ITEM_INDEX = 2
        for item in last20MoviesList:
            itemID = item[ITEM_INDEX]
            genres = self.items.iloc[itemID]['Genres'].split("|")
            for genre in genres:
                result[self.dictOfGenreIndexes[genre]] += 1

        # create polynomial features from [seniority]*[genres]
        poly = PolynomialFeatures(2)
        result = poly.fit_transform(result.reshape(-1, 1))
        result = result.flatten()

        # add user gender, age and occupation to the context
        userFeatures = []
        userFeatures.append(self.users.iloc[userID]['age'])
        userFeatures.append(1 if self.users.iloc[userID]['gender'] == 'F' else -1)
        userFeatures.append(self.users.iloc[userID]['occupation'])

        result = np.append(result, userFeatures)

        # adjust context dimension attribute
        self._contextDim = len(result)

        return result

    # methodsResultDict:{String:Series(rating:float[], itemID:int[])},
    # modelDF:DataFrame<(methodID:str, votes:int)>, numberOfItems:int
    def runWithResponsibility(self, methodsResultDict:dict, modelDF:DataFrame, userID:int, numberOfItems:int=20):
        # TODO: CHECK DATA INTEGRITY!
        # methodsResultNewDict: dict[str, pd.Series] = self._penaltyTool.runPenalization(
        #        userID, methodsResultDict, self._history)

        itemsWithResposibilityOfRecommenders:List[int,Series[int,str]] = super().runWithResponsibility(
            methodsResultDict, modelDF, userID, numberOfItems=numberOfItems)

        return itemsWithResposibilityOfRecommenders
