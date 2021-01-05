#!/usr/bin/python3

from evaluationTool.aEvalTool import AEvalTool #class

import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import PolynomialFeatures

from typing import List

from pandas.core.frame import DataFrame  # class

class EvalToolContext(AEvalTool):

    ARG_SELECTOR: str = "selector"
    ARG_ITEMS: str = "items"
    ARG_USERS: str = "users"
    ARG_DATASET: str = "dataset"
    ARG_USER_ID: str = "userID"
    ARG_RELEVANCE = "relevance"
    ARG_HISTORY = "history"

    def __init__(self, argsDict:dict):
        if type(argsDict) is not dict:
            raise ValueError("Argument argsDict isn't type dict.")

        self.maxVotesConst:float = 0.99
        self.minVotesConst:float = 0.01
        self.dataset_name: str = argsDict[self.ARG_DATASET]
        self.items: DataFrame = self._preprocessItems(argsDict[self.ARG_ITEMS])
        self.users: DataFrame = self._preprocessUsers(argsDict[self.ARG_USERS])
        self.history = argsDict[self.ARG_HISTORY]
        self._contextDim: int = 0
        self._b: dict = {}
        self._A: dict = {}
        self._inverseA: dict = {}
        self._context = None
        self._INVERSE_CALCULATION_THRESHOLD: int = 100
        self._inverseCounter: int = 0

    def _preprocessUsers(self, users: DataFrame):
        if self.dataset_name == "ml":
            return self._onehotUsersOccupationML(users)
        else:
            raise ValueError("Dataset " + self.dataset_name + " is not supported!")

    def _onehotUsersOccupationML(self, users: DataFrame):
        one_hot_encoding = pd.get_dummies(users['occupation'])
        users.drop(['occupation'], axis=1, inplace=True)
        return pd.concat([users, one_hot_encoding], axis=1)

    def _preprocessItems(self, items: DataFrame):
        if self.dataset_name == "ml":
            return self._onehotItemsGenresML(items)
        else:
            raise ValueError("Dataset " + self.dataset_name + " is not supported!")

    def _onehotItemsGenresML(self, items: DataFrame):
        one_hot_encoding = items["Genres"].str.get_dummies(sep='|')
        one_hot_encoding.drop(one_hot_encoding.columns[0], axis=1, inplace=True)
        tmp = items.drop(['Genres'], axis=1, inplace=False)
        return pd.concat([tmp, one_hot_encoding], axis=1)

    def click(self, rItemIDsWithResponsibility:List, clickedItemID:int, portfolioModel:DataFrame, evaluationDict:dict):
        if type(rItemIDsWithResponsibility) is not list:
            raise ValueError("Argument rItemIDsWithResponsibility isn't type list.")
        if type(clickedItemID) is not int and type(clickedItemID) is not np.int64:
            raise ValueError("Argument clickedItemID isn't type int.")
        if type(portfolioModel) is not DataFrame:
            raise ValueError("Argument pModelDF isn't type DataFrame.")
        if list(portfolioModel.columns) != ['votes']:
            raise ValueError("Argument pModelDF doen't contain rights columns.")
        if type(evaluationDict) is not dict:
            raise ValueError("Argument evaluationDict isn't type dict.")

        # get userID from dict
        userID = evaluationDict[self.ARG_USER_ID]

        # compute context for selected user
        self._context = self._calculateContext(userID)

        # check for each recommender method that it has A, b, inverseA
        for recommender, row in portfolioModel.iterrows():
            if recommender not in self._A:
                self._b[recommender] = np.zeros(self._contextDim)
                self._A[recommender] = np.identity(self._contextDim)
                self._inverseA[recommender] = np.identity(self._contextDim)

        # get relevances
        methodsResultDict = evaluationDict[self.ARG_RELEVANCE]

        # update b's
        for recommender, relevances in methodsResultDict.items():
            if clickedItemID in relevances:
                # TODO: Maybe sum of rewards should be 1? (now it is below 1)
                reward = relevances.loc[clickedItemID]
                self._b[recommender] = self._b[recommender] + (reward * self._context)

        # update recommender's votes
        updatedVotes = dict()
        totalOriginalVotes = 0
        totalUpdatedVotes = 0
        for recommender, votes in portfolioModel.iterrows():
            # Calculate change rate
            ridgeRegression = self._inverseA[recommender].dot(self._b[recommender])
            UCB = self._context.T.dot(self._inverseA[recommender]).dot(self._context)
            change_rate = (ridgeRegression.T.dot(self._context) + math.sqrt(UCB))

            # update votes
            updatedVotes[recommender] = change_rate
            totalOriginalVotes += portfolioModel.at[recommender, 'votes']
            totalUpdatedVotes += updatedVotes[recommender]

        # normalize updated votes and save it to modelDF
        rate = (totalOriginalVotes / totalUpdatedVotes)
        for recommender, votes in portfolioModel.iterrows():
            portfolioModel.at[recommender, 'votes'] = round(updatedVotes[recommender] * rate)

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
        CLICKED_INDEX = 5
        previousClickedItemsOfUser = list(filter(lambda x: x[CLICKED_INDEX], self.history.getPreviousRecomOfUser(userID)))
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

    def displayed(self, rItemIDsWithResponsibility:List, portfolioModel:DataFrame, evaluationDict:dict):
        if type(rItemIDsWithResponsibility) is not list:
            raise ValueError("Argument rItemIDsWithResponsibility isn't type list.")
        if type(portfolioModel) is not DataFrame:
            raise ValueError("Argument pModelDF isn't type DataFrame.")
        if list(portfolioModel.columns) != ['votes']:
            raise ValueError("Argument pModelDF doen't contain rights columns.")
        if type(evaluationDict) is not dict:
            raise ValueError("Argument evaluationDict isn't type dict.")

        userID = evaluationDict[self.ARG_USER_ID]

        # recompute context - previous user doesn't have to be the same as current
        # TODO: Performace improvement: check if user changed -> do not recompute context if not?
        self._context = self._calculateContext(userID)

        # get relevances
        methodsResultDict = evaluationDict[self.ARG_RELEVANCE]

        for recommender, value in self._A.items():
            # get relevance of items, which were recommended by recommender and are in itemsWithResposibilityOfRecommenders
            relevanceSum = 0

            for recommendedItemID, votes in rItemIDsWithResponsibility:
                if recommendedItemID in methodsResultDict[recommender].index:
                    relevanceSum += methodsResultDict[recommender][recommendedItemID]
            self._A[recommender] += self._context.dot(self._context.T) * relevanceSum

        # recompute inverse A's if threshold is hit
        if self._inverseCounter > self._INVERSE_CALCULATION_THRESHOLD:
            for recommender, value in self._inverseA.items():
                self._inverseA[recommender] = np.linalg.inv(self._A[recommender])
            self._inverseCounter = 0
        self._inverseCounter += 1

        # update recommender's votes
        updatedVotes = dict()
        totalOriginalVotes = 0
        totalUpdatedVotes = 0
        for recommender, votes in portfolioModel.iterrows():
            # Calculate change rate
            ridgeRegression = self._inverseA[recommender].dot(self._b[recommender])
            UCB = self._context.T.dot(self._inverseA[recommender]).dot(self._context)
            change_rate = (ridgeRegression.T.dot(self._context) + math.sqrt(UCB))

            # update votes
            updatedVotes[recommender] = change_rate
            totalOriginalVotes += portfolioModel.at[recommender, 'votes']
            totalUpdatedVotes += updatedVotes[recommender]

        # normalize updated votes and save it to modelDF
        rate = (totalOriginalVotes / totalUpdatedVotes)

        for recommender, votes in portfolioModel.iterrows():
            portfolioModel.at[recommender, 'votes'] = round(updatedVotes[recommender] * rate)
