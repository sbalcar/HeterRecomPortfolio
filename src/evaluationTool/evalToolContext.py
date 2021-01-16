#!/usr/bin/python3

from evaluationTool.aEvalTool import AEvalTool #class

import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import PolynomialFeatures

from typing import List
from typing import Dict #class

from pandas.core.frame import DataFrame  # class

class EvalToolContext(AEvalTool):

    ARG_SELECTOR: str = "selector"
    ARG_ITEMS: str = "items"
    ARG_USERS: str = "users"
    ARG_DATASET: str = "dataset"
    ARG_USER_ID: str = "userID"
    ARG_RELEVANCE = "relevance"
    ARG_HISTORY = "history"
    ARG_SENIORITY = "seniority"
    ARG_PAGE_TYPE = "page_type"
    ARG_ITEMS_SHOWN = "items_shown"
    ARG_ITEM_ID = "itemID"
    ARG_EVENTS = "events"

    def __init__(self, argumentsDict:Dict[str,object]):
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argsDict isn't type dict.")

        self.maxVotesConst:float = 0.99
        self.minVotesConst:float = 0.01
        self.dataset_name: str = argumentsDict[self.ARG_DATASET]
        self.items: DataFrame = self._preprocessItems(argumentsDict[self.ARG_ITEMS])
        self.users: DataFrame = self._preprocessUsers(argumentsDict)
        self.history = argumentsDict[self.ARG_HISTORY]
        self._contextDim: int = 0
        self._b: dict = {}
        self._A: dict = {}
        self._inverseA: dict = {}
        self._context = None
        self._INVERSE_CALCULATION_THRESHOLD: int = 2
        self._inverseCounter: int = 0

    def _preprocessUsers(self, argsDict: dict):
        if self.dataset_name == "ml":
            return self._onehotUsersOccupationML(argsDict[self.ARG_USERS])
        elif self.dataset_name == "st":
            return self._preprocessEventsST(argsDict[self.ARG_EVENTS])
        else:
            raise ValueError("Dataset " + self.dataset_name + " is not supported!")

    def _preprocessEventsST(self, events: DataFrame):
        self.users = {}
        for index, row in events.iterrows():
            if row.loc['pageType'] == "zobrazit":
                if row.loc['userID'] not in self.users:
                    self.users[row.loc['userID']] = [row.loc['objectID']]
                else:
                    if row.loc['objectID'] not in self.users[row.loc['userID']]:
                        self.users[row.loc['userID']].append(row.loc['objectID'])
        return self.users

    def _onehotUsersOccupationML(self, users: DataFrame):
        one_hot_encoding = pd.get_dummies(users['occupation'])
        users.drop(['occupation'], axis=1, inplace=True)
        return pd.concat([users, one_hot_encoding], axis=1)

    def _preprocessItems(self, items: DataFrame):
        if self.dataset_name == "ml":
            return self._onehotItemsGenresML(items)
        elif self.dataset_name == "st":
            return self._preprocessItemsST(items)
        else:
            raise ValueError("Dataset " + self.dataset_name + " is not supported!")

    def _preprocessItemsST(self, items:DataFrame):
        #print(items['prumerna_cena_noc'].head(10))
        listMonths = ["month_" + str(i) for i in range(1,13)]
        listMonths.extend(['delka', 'id_serial'])
        print(listMonths)
        self.items = items[listMonths]
        items.loc[items['prumerna_cena_noc']<=0,'prumerna_cena_noc'] = 1
        #print(items[['delka', 'id_serial','prumerna_cena_noc']].describe())
        log_price = items[['prumerna_cena_noc']].apply(lambda x: math.log(x[0]), axis=1)
        self.items = self.items.join(pd.DataFrame(data=log_price, columns=['prumerna_cena_noc']))
        #print(self.items.head())
        # TODO: weird format of 'zeme' column->think about spliting the values
        #  (f.e. if value is "Anglie:Sport" -> in OHE into 2 columns "Anglie", "Sport",
        #  not into one column "Anglie:Sport")
        # onehot country
        oneHot = pd.get_dummies(items['zeme'])
        self.items = self.items.join(oneHot)

        # onehot accomodation
        oneHot = pd.get_dummies(items['ubytovani'], prefix=['ubytovani'])
        self.items = self.items.join(oneHot)
        #print(self.items.head())
        # onehot transport
        oneHot = pd.get_dummies(items['doprava'], prefix=['doprava'])
        self.items = self.items.join(oneHot)

        # onehot food
        oneHot = pd.get_dummies(items['strava'], prefix=['strava'])
        self.items = self.items.join(oneHot)

        #onehot id_type
        oneHot = pd.get_dummies(items['id_typ'], prefix=['typ'])
        self.items = self.items.join(oneHot)
        #print(self.items.head())
        # add months
        
        """
        for i in range(1, 13):
            self.items.insert(0, "month_" + str(i), 0)

        # populate months
        dfMonths = items[['od', 'do']]
        for index, row in dfMonths.iterrows():
            i = int(row["od"].split('-')[1])
            j = int(row["do"].split('-')[1])
            while True:
                self.items.loc[index, "month_" + str(i)] = 1

                if (i == j) or (j <= 0) or (j > 12):
                    break
                else:
                    i = (i % 12) + 1
        """
        print(self.items.shape)
        self.items.set_index('id_serial', inplace=True)
        return self.items

    def _onehotItemsGenresML(self, items: DataFrame):
        one_hot_encoding = items["Genres"].str.get_dummies(sep='|')
        one_hot_encoding.drop(one_hot_encoding.columns[0], axis=1, inplace=True)
        tmp = items.drop(['Genres'], axis=1, inplace=False)
        return pd.concat([tmp, one_hot_encoding], axis=1)

    def click(self, rItemIDsWithResponsibility:List, clickedItemID:int, portfolioModel:DataFrame, argumentsDict:Dict[str,object]):
        if type(rItemIDsWithResponsibility) is not list:
            raise ValueError("Argument rItemIDsWithResponsibility isn't type list.")
        if type(clickedItemID) is not int and type(clickedItemID) is not np.int64:
            raise ValueError("Argument clickedItemID isn't type int.")
        if type(portfolioModel) is not DataFrame:
            raise ValueError("Argument pModelDF isn't type DataFrame.")
        if list(portfolioModel.columns) != ['votes']:
            raise ValueError("Argument pModelDF doen't contain rights columns.")
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")

        # get userID from dict
        userID = argumentsDict[self.ARG_USER_ID]

        # compute context for selected user
        self._context = self.calculateContext(userID, argumentsDict)

        # check for each recommender method that it has A, b, inverseA
        for recommender, row in portfolioModel.iterrows():
            if recommender not in self._A:
                self._b[recommender] = np.zeros(self._contextDim)
                self._A[recommender] = np.identity(self._contextDim)
                self._inverseA[recommender] = np.identity(self._contextDim)

        # update b's
        for recommendedItemID, votes in rItemIDsWithResponsibility:
            for recommender, _ in self._b.items():
                # TODO: Maybe sum of rewards should be 1? (now it is below 1)
                reward = votes[recommender]
                self._b[recommender] = self._b[recommender] + (reward * self._context)

    def calculateContext(self, userID, argumentsDict:Dict[str,object]):
        if self.dataset_name == "ml":
            return self._calculateContextML(userID)
        elif self.dataset_name == "st":
            return self._calculateContextST(userID, argumentsDict)
        else:
            raise ValueError("Dataset " + self.dataset_name + " is not supported!")

    def _calculateContextST(self, userID, argumentsDict):
        result = np.zeros(5)
        if argumentsDict[self.ARG_PAGE_TYPE] == 'zobrazit':
            itemID = argumentsDict[self.ARG_ITEM_ID]

            # check if item is in items
            if itemID in self.items.index:
                item = self.items.loc[itemID]

                # if this is new user
                if userID not in self.users:
                    self.users[userID] = [itemID]
                else:
                    if itemID not in self.users[userID]:
                        self.users[userID].append(itemID)

            # if item not present in items -> init empty list
            else:
                item = np.array([0] * self.items.shape[1], dtype=float)

            result = np.append(result, item)
            result[3] = 1

        else:
            if argumentsDict[self.ARG_PAGE_TYPE] == 'index':
                result[4] = 1
            else:
                result[5] = 1
            AggregationOfItems = np.array([0] * self.items.shape[1], dtype=float)

            # if we even know the user
            if userID in self.users:
                for objectID in self.users[userID]:
                    # check if itemID is in items
                    if objectID in self.items.index:
                        AggregationOfItems += self.items.loc[objectID]
                AggregationOfItems.to_numpy()

            # do nothing if we don't know user (leave context empty)

            # append aggregation to results
            result = np.append(result, AggregationOfItems)

        result[0] = math.log(argumentsDict[self.ARG_SENIORITY])
        result[1] = argumentsDict[self.ARG_ITEMS_SHOWN]

        poly = PolynomialFeatures(2)
        result = poly.fit_transform(result.reshape(-1, 1))
        result = result.flatten()

        self._contextDim = len(result)

        return result

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
        itemsIDs = [i[2] for i in last20MoviesList]
        items = self.items.loc[self.items['movieId'].isin(itemsIDs)]
        itemsGenres = items.drop(items.columns[[0,1]], axis=1).sum()
        result = np.append(result, [float(i) for i in itemsGenres])

        # create polynomial features from [seniority]*[genres]*[userInfo]
        # append age and onehot occupation
        tmp = user.T.drop(labels=['userId', 'gender', 'zipCode', 'age']).to_numpy().flatten()
        result = np.concatenate([result, [float(i) for i in tmp]])

        # add user gender to the context (one-hot encoding)
        result = np.append(result, 1.0 if user['gender'].item() == 'F' else 0.0)
        result = np.append(result, 1.0 if user['gender'].item() != 'F' else 0.0)

        poly = PolynomialFeatures(2)
        result = poly.fit_transform(result.reshape(-1, 1))
        result = result.flatten()

        # adjust context dimension attribute
        self._contextDim = len(result)

        return result

    def displayed(self, rItemIDsWithResponsibility:List, portfolioModel:DataFrame, argumentsDict:Dict[str,object]):
        if type(rItemIDsWithResponsibility) is not list:
            raise ValueError("Argument rItemIDsWithResponsibility isn't type list.")
        if type(portfolioModel) is not DataFrame:
            raise ValueError("Argument pModelDF isn't type DataFrame.")
        if list(portfolioModel.columns) != ['votes']:
            raise ValueError("Argument pModelDF doen't contain rights columns.")
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")

        userID = argumentsDict[self.ARG_USER_ID]

        # recompute context - previous user doesn't have to be the same as current
        # TODO: Performace improvement: check if user changed -> do not recompute context if not?
        self._context = self.calculateContext(userID, argumentsDict)

        # check for each recommender method that it has A, b, inverseA
        for recommender, row in portfolioModel.iterrows():
            if recommender not in self._A:
                self._b[recommender] = np.zeros(self._contextDim)
                self._A[recommender] = np.identity(self._contextDim)
                self._inverseA[recommender] = np.identity(self._contextDim)
                
        for recommender, value in self._A.items():
            # get relevance of items, which were recommended by recommender and are in itemsWithResposibilityOfRecommenders
            relevanceSum = 0

            for recommendedItemID, votes in rItemIDsWithResponsibility:
                relevanceSum += votes[recommender]
            self._A[recommender] += np.outer(self._context.T, self._context) * relevanceSum

        # recompute inverse A's if threshold is hit
        if self._inverseCounter > self._INVERSE_CALCULATION_THRESHOLD:
            print('=============================RECALCULATE INVERSE MATRIX!=================')
            for recommender, value in self._inverseA.items():
                self._inverseA[recommender] = np.linalg.inv(self._A[recommender])
            self._inverseCounter = 0
            if self._INVERSE_CALCULATION_THRESHOLD < 100:
                self._INVERSE_CALCULATION_THRESHOLD *= 2
        self._inverseCounter += 1