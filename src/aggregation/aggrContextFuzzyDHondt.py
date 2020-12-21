#!/usr/bin/python3

import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import PolynomialFeatures

from typing import List

from pandas.core.frame import DataFrame  # class
from pandas.core.series import Series  # class

from aggregation.aggrFuzzyDHondt import AggrFuzzyDHondt  # class

from history.aHistory import AHistory  # class


class AggrContextFuzzyDHondt(AggrFuzzyDHondt):
    ARG_SELECTOR: str = "selector"
    ARG_ITEMS: str = "items"
    ARG_USERS: str = "users"
    ARG_DATASET: str = "dataset"
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
        self._history = history
        self._listOfPreviousRecommendations = None
        self._listOfPreviousClickedItems = None
        self._contextDim: int = 0
        self._b: dict = None
        self._A: dict = None
        self._context = None
        self._inverseA: dict = None
        self._INVERSE_CALCULATION_THRESHOLD: int = 100
        self._inverseCounter: int = 0
        self.dataset_name: str = argumentsDict[self.ARG_DATASET]
        self.items: DataFrame = self.__preprocessItems(argumentsDict[self.ARG_ITEMS])
        self.users: DataFrame = self.__preprocessUsers(argumentsDict[self.ARG_USERS])

    def __preprocessUsers(self, users:DataFrame):
        if self.dataset_name == "ml":
            return self.__onehotUsersOccupationML(users)
        else:
            raise ValueError("Dataset " + self.dataset_name + " is not supported!")

    def __onehotUsersOccupationML(self, users:DataFrame):
        one_hot_encoding = pd.get_dummies(users['occupation'])
        users.drop(['occupation'], axis=1, inplace=True)
        return pd.concat([users, one_hot_encoding], axis=1)

    def __preprocessItems(self, items:DataFrame):
        if self.dataset_name == "ml":
            return self.__onehotItemsGenresML(items)
        else:
            raise ValueError("Dataset " + self.dataset_name + " is not supported!")

    def __onehotItemsGenresML(self, items:DataFrame):
        one_hot_encoding = items["Genres"].str.get_dummies(sep='|')
        one_hot_encoding.drop(one_hot_encoding.columns[0],axis=1, inplace=True)
        items.drop(['Genres'], axis=1, inplace=True)
        return pd.concat([items, one_hot_encoding], axis=1)

    # methodsResultDict:{String:pd.Series(rating:float[], itemID:int[])},
    # modelDF:pd.DataFrame[numberOfVotes:int], numberOfItems:int
    def run(self, methodsResultDict: dict, modelDF: DataFrame, userID: int, numberOfItems: int = 20):
        # TODO: CHECK DATA INTEGRITY!

        # update context
        self._context = self.__calculateContext(userID)

        # check if contextDim is OK
        for name, value in methodsResultDict.items():
            if (self._A is not None and self._contextDim != len(self._A[name])) or \
                    (self._b is not None and self._contextDim != len(self._b[name])):
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
            dictOfRewards: dict = {}
            counter = 0
            for recommender, value in self._listOfPreviousRecommendations.items():
                # TODO: Is performance OK here? This is just sketch
                # calculate size of intersection between ListOfClickedItems and self._listOfPreviousRecommendations
                succesfulRecomendations = len(list(set(ListOfClickedItems)
                                                   .intersection(value)))

                # give recommender 'points' based on the size of intersection
                # TODO: I don't take into account previous relevance or previous votes, maybe needs to be improved in future
                dictOfRewards[recommender] = succesfulRecomendations
                counter += 1

            # update b's
            for recommender, value in self._b.items():
                reward = dictOfRewards[recommender] / counter
                self._b[recommender] = self._b[recommender] + (reward * self._context)

        # update recommender's votes
        for recommender, votes in modelDF.iterrows():
            # Calculate change rate
            ridgeRegression = self._inverseA[recommender].dot(self._b[recommender])
            UCB = self._context.T.dot(self._inverseA[recommender]).dot(self._context)
            x = 0
            change_rate = ridgeRegression.T.dot(self._context) + math.sqrt(UCB)

            # update votes
            modelDF.at[recommender, 'votes'] = change_rate * votes['votes']

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
        self._listOfPreviousRecommendations = methodsResultDict

        # set clicked items to None (because next time there could be different userID)
        # TODO: an alternative is to calculate this list for every user (ask about performance? Will be every user called? Then it would make sense to keep these lists in memory for every user
        self._listOfPreviousClickedItems = None
        return itemsWithResposibilityOfRecommenders

    def __getListOfPreviousClickedItems(self, userID: int):
        if self.dataset_name == "ml":
            return self.__getListOfPreviousClickedItemsML(userID)
        else:
            raise ValueError("Dataset " + self.dataset_name + " is not supported!")

    def __getListOfPreviousClickedItemsML(self, userID: int):
        # TODO: Function is not needed if Stepan'll add something like "update" function
        # for now, return whole list of every clicked item (calculated already in context)
        return self._listOfPreviousClickedItems[:20]


        # # get list of history with user
        # previousItemsOfUser = self._listOfPreviousClickedItems
        #
        # if previousItemsOfUser is None:
        #
        # # if we have no history about the user
        # if len(previousItemsOfUser) == 0:
        #     return list()
        #
        # return previousItemsOfUser
#
        # # data indexes in previousItemsOfUser
        # ITEM_INDEX = 2
        # CLICKED_INDEX = 5
        # TIMESTAMP_INDEX = 6
#
        # # sort history by timestamp
        # sortedItemsByTimestamp = sorted(previousItemsOfUser, key=lambda x: x[TIMESTAMP_INDEX], reverse=True)
#
        # # get newest timestamp
        # newLastTimestamp = sortedItemsByTimestamp[0][TIMESTAMP_INDEX]
#
        # # check data integrity of lastTimestamp
        # if userID not in self._lastTimestamp:
        #     raise ValueError("Timestamp has to be initialized for each user!")
        # # if there is no new history from user
        # if self._lastTimestamp[userID] == newLastTimestamp:
        #     return list()
#
        # # get last interactions with the user (iteratively based on history timestamps)
        # iterator = 0
        # result = list()
        # while iterator < len(sortedItemsByTimestamp) and sortedItemsByTimestamp[iterator][TIMESTAMP_INDEX] > \
        #         self._lastTimestamp[userID]:
        #     if sortedItemsByTimestamp[iterator][CLICKED_INDEX] and \
        #             sortedItemsByTimestamp[iterator][ITEM_INDEX] not in result:
        #         result.append(sortedItemsByTimestamp[iterator][ITEM_INDEX])
        #     iterator += 1
#
        # self._lastTimestamp[userID] = newLastTimestamp
#
        # return result

    def __calculateContext(self, userID):
        if self.dataset_name == "ml":
            return self.__calculateContextML(userID)
        else:
            raise ValueError("Dataset " + self.dataset_name + " is not supported!")

    def __calculateContextML(self, userID):

        # get user data
        user = self.users.iloc[userID]

        # init result
        result = np.zeros(2)

        # add seniority of user into the context (filter only clicked items)
        CLICKED_INDEX = 5
        previousClickedItemsOfUser = list(
            filter(lambda x: x[CLICKED_INDEX], self._history.getPreviousRecomOfUser(userID)))
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
        result[1] = math.log(historySize, 2)

        # get last 20 movies from user and aggregate their genres
        TIMESTAMP_INDEX = 6
        self._listOfPreviousClickedItems = sorted(previousClickedItemsOfUser, key=lambda x: x[TIMESTAMP_INDEX], reverse=True)
        last20MoviesList = self._listOfPreviousClickedItems[:20]

        # aggregation
        ITEM_INDEX = 2
        itemsIDs = set([x[ITEM_INDEX] for x in last20MoviesList])
        items = self.items.iloc[list(itemsIDs)]
        itemsGenres = items.drop(items.columns[[0,1]], axis=1).sum()
        result = np.append(result, itemsGenres)

        # create polynomial features from [seniority]*[genres]
        poly = PolynomialFeatures(2)
        result = poly.fit_transform(result.reshape(-1, 1))
        result = result.flatten()

        # add user gender to the context
        userFeatures = []
        userFeatures.append(1 if self.users.iloc[userID]['gender'] == 'F' else -1)

        # append age and onehot occupation
        tmp = self.users.iloc[userID].drop(labels=['userId', 'gender', 'zipCode']).to_numpy()
        userFeatures = np.concatenate([userFeatures, tmp])

        result = np.concatenate([result, userFeatures])

        # adjust context dimension attribute
        self._contextDim = len(result)

        return result

    # methodsResultDict:{String:Series(rating:float[], itemID:int[])},
    # modelDF:DataFrame<(methodID:str, votes:int)>, numberOfItems:int
    def runWithResponsibility(self, methodsResultDict: dict, modelDF: DataFrame, userID: int, numberOfItems: int = 20):
        # TODO: CHECK DATA INTEGRITY!
        # methodsResultNewDict: dict[str, pd.Series] = self._penaltyTool.runPenalization(
        #        userID, methodsResultDict, self._history)

        itemsWithResposibilityOfRecommenders: List[int, Series[int, str]] = super().runWithResponsibility(
            methodsResultDict, modelDF, userID, numberOfItems=numberOfItems)

        return itemsWithResposibilityOfRecommenders
