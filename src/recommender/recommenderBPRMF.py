#!/usr/bin/python3

import os
from tqdm import tqdm
import scipy.sparse as sp

import pickle
import pandas as pd
import numpy as np

from typing import List # class
from typing import Dict  # class

from pandas.core.frame import DataFrame  # class
from pandas.core.series import Series  # class

from recommender.aRecommender import ARecommender #class

from history.aHistory import AHistory #class
from history.historyHierDF import HistoryHierDF  # class

from datasets.aDataset import ADataset #class
from datasets.datasetML import DatasetML #class
from datasets.datasetRetailRocket import DatasetRetailRocket #class
from datasets.datasetST import DatasetST #class

from typing import Dict
from sklearn.preprocessing import normalize


class RecommenderBPRMF(ARecommender):

    ARG_EPOCHS:str = "epochs"
    ARG_FACTORS:str = "factors"
    ARG_LEARNINGRATE:str = "learning_rate"
    ARG_UREGULARIZATION:str = "user_regularization"
    ARG_BREGULARIZATION:str = "bias_regularization"
    ARG_PIREGULARIZATION:str = "positive_item_regularization"
    ARG_NIREGULARIZATION:str = "negative_item_regularization"


    def __init__(self, batchID:str, argumentsDict:Dict[str,object]):
        if type(batchID) is not str:
            raise ValueError("Argument batchID is not type strt.")
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict is not type dict.")

        self._batchID:str = batchID

        self._epochs = argumentsDict[self.ARG_EPOCHS]
        self._factors = argumentsDict[self.ARG_FACTORS]
        self._learning_rate = argumentsDict[self.ARG_LEARNINGRATE]
        self._user_regularization = argumentsDict[self.ARG_UREGULARIZATION]
        self._bias_regularization = argumentsDict[self.ARG_BREGULARIZATION]
        self._positive_item_regularization = argumentsDict[self.ARG_PIREGULARIZATION]
        self._negative_item_regularization = argumentsDict[self.ARG_NIREGULARIZATION]
        self._seed = 27
        self._verbose = True
        self._batch_size = 1


    def train(self, history:AHistory, dataset:ADataset):
        if not isinstance(history, AHistory):
            raise ValueError("Argument history isn't type AHistory.")
        if not isinstance(dataset, ADataset):
            raise ValueError("Argument dataset isn't type ADataset.")

        self._data = Data(dataset)

        self._model = ModelMF(self._factors,
                              self._data,
                              self._learning_rate,
                              self._user_regularization,
                              self._bias_regularization,
                              self._positive_item_regularization,
                              self._negative_item_regularization,
                              self._seed)
        self._sampler = Sampler(self._data.i_train_dict)

        print(f"Transactions: {self._data.transactions}")

        for it in self.__iterate(self._epochs):
            print(f"\n********** Iteration: {it + 1}")
            loss = 0
            steps = 0
            with tqdm(total=int(self._data.transactions // self._batch_size), disable=not self._verbose) as t:
                for batch in self._sampler.step(self._data.transactions, self._batch_size):
                    steps += 1
                    self._model.train_step(batch)
                    t.update()


    def __iterate(self, epochs):
        for iteration in range(epochs):
            yield iteration

    def update(self, ratingsUpdateDF:DataFrame, argumentsDict:Dict[str, object]):
        if type(ratingsUpdateDF) is not DataFrame:
            raise ValueError("Argument ratingsUpdateDF isn't type DataFrame.")
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")
        pass

    def recommend(self, userId:int, numberOfItems:int, argumentsDict:Dict[str, object]):
        if type(userId) is not int and type(userId) is not np.int64:
            raise ValueError("Argument userId " + str(userId) + " isn't type int.")
        if type(numberOfItems) is not int:
            raise ValueError("Argument numberOfItems isn't type int.")
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")

        if not userId in self._data.users:
            print("unknown user: " + str(userId))
            return Series([], index=[])

        mask = None
        rec:List[tuple] = self._model.get_user_predictions(userId, mask, numberOfItems)

        recItemIDs:List[int] = [i for i, _ in rec]
        recScores:List[float] = [r for _, r in rec]

        finalScores = normalize(np.expand_dims(recScores, axis=0))[0, :]

        return Series(finalScores, index=recItemIDs)





class Sampler:
    def __init__(self, indexed_ratings):
        np.random.seed(42)
        self._indexed_ratings = indexed_ratings
        self._users = list(self._indexed_ratings.keys())
        self._nusers = len(self._users)
        self._items = list({k for a in self._indexed_ratings.values() for k in a.keys()})
        self._nitems = len(self._items)
        self._ui_dict = {u: list(set(indexed_ratings[u])) for u in indexed_ratings}
        self._lui_dict = {u: len(v) for u, v in self._ui_dict.items()}

    def step(self, events: int, batch_size: int):
        r_int = np.random.randint
        n_users = self._nusers
        n_items = self._nitems
        ui_dict = self._ui_dict
        lui_dict = self._lui_dict

        def sample():
            u = r_int(n_users)
            ui = ui_dict[u]
            lui = lui_dict[u]
            if lui == n_items:
                sample()
            i = ui[r_int(lui)]

            j = r_int(n_items)
            while j in ui:
                j = r_int(n_items)
            return u, i, j

        for batch_start in range(0, events, batch_size):
            bui, bii, bij = map(np.array, zip(*[sample() for _ in range(batch_start, min(batch_start + batch_size, events))]))
            yield bui[:, None], bii[:, None], bij[:, None]



class Data(object):

    def __init__(self, dataset):
        if not isinstance(dataset, ADataset):
            raise ValueError("Argument dataset isn't type ADataset.")

        self.train_dict = self.__dataframe_to_dict(dataset)

        self.users:List[int] = list(self.train_dict.keys())
        self.items = list({k for a in self.train_dict.values() for k in a.keys()})
        self.num_users = len(self.users)
        self.num_items = len(self.items)
        self.transactions = sum(len(v) for v in self.train_dict.values())

        self.private_users = {p: u for p, u in enumerate(self.users)}
        self.public_users = {v: k for k, v in self.private_users.items()}
        self.private_items = {p: i for p, i in enumerate(self.items)}
        self.public_items = {v: k for k, v in self.private_items.items()}

        self.i_train_dict = {self.public_users[user]: {self.public_items[i]: v for i, v in items.items()}
                                for user, items in self.train_dict.items()}


    def __dataframe_to_dict(self, dataset:ADataset):

        #ratingsDF:DataFrame = dataset.ratingsDF

        if type(dataset) is DatasetML:
            from datasets.ml.ratings import Ratings  # class
            COL_USERID:str = Ratings.COL_USERID
            COL_ITEMID:str = Ratings.COL_MOVIEID
            COL_RATING:str = Ratings.COL_RATING
            ratingsDF:DataFrame = dataset.ratingsDF
            # ratingsSum:Dataframe<(userId:int, movieId:int, ratings:int, timestamp:int)>
            ratingsDF = ratingsDF.loc[ratingsDF[Ratings.COL_RATING] >= 4]

        elif type(dataset) is DatasetRetailRocket:
            from datasets.retailrocket.events import Events  # class
            COL_USERID:str = Events.COL_VISITOR_ID
            COL_ITEMID:str = Events.COL_ITEM_ID
            ratingsDF:DataFrame = dataset.eventsDF


        elif type(dataset) is DatasetST:
            from datasets.slantour.events import Events  # class
            COL_USERID:str = Events.COL_USER_ID
            COL_ITEMID:str = Events.COL_OBJECT_ID
            ratingsDF:DataFrame = dataset.eventsDF


        users = list(ratingsDF[COL_USERID].unique())

        "Conversion to Dictionary"
        ratings = {}
        for u in users:
            sel_ = ratingsDF[ratingsDF[COL_USERID] == u]
            if type(dataset) in [DatasetRetailRocket, DatasetST]:
                ratings[u] = dict(zip(sel_[COL_ITEMID], [1.0]*len(sel_)))
            else:
                ratings[u] = dict(zip(sel_[COL_ITEMID], sel_[COL_RATING]))

        return ratings



class ModelMF(object):
    def __init__(self, F,
                 data,
                 lr,
                 user_regularization,
                 bias_regularization,
                 positive_item_regularization,
                 negative_item_regularization,
                 random_seed,
                 *args):
        np.random.seed(random_seed)
        self._factors = F
        self._users = data.users
        self._items = data.items
        self._private_users = data.private_users
        self._public_users = data.public_users
        self._private_items = data.private_items
        self._public_items = data.public_items
        self._learning_rate = lr
        self._user_regularization = user_regularization
        self._bias_regularization = bias_regularization
        self._positive_item_regularization = positive_item_regularization
        self._negative_item_regularization = negative_item_regularization

        self.initialize(*args)

    def initialize(self, loc: float = 0, scale: float = 0.1):

        self._global_bias = 0

        "same parameters as np.randn"
        self._user_bias = np.zeros(len(self._users))
        self._item_bias = np.zeros(len(self._items))
        self._user_factors = \
            np.random.normal(loc=loc, scale=scale, size=(len(self._users), self._factors))
        self._item_factors = \
            np.random.normal(loc=loc, scale=scale, size=(len(self._items), self._factors))

    def predict(self, user, item):
        return self._global_bias + self._item_bias[self._public_items[item]] \
               + self._user_factors[self._public_users[user]] @ self._item_factors[self._public_items[item]]

    def indexed_predict(self, user, item):
        return self._global_bias + self._item_bias[item] \
               + self._user_factors[user] @ self._item_factors[item]

    def get_user_predictions(self, user_id, mask, top_k=10):
        #print(self._public_users)
        user_id = self._public_users.get(user_id)
        #print("user_id: " + str(user_id))
        #print("self._item_bias.shape: " + str(self._item_bias.shape))
        #print("self._user_factors.shape: " + str(self._user_factors.shape))
        #print("self._user_factors[user_id].shape: " + str(self._user_factors[user_id].shape))
        #print("self._item_factors.T.shape: " + str(self._item_factors.T.shape))
        b = self._item_bias + self._user_factors[user_id] @ self._item_factors.T

        #a = mask[user_id]
        #b[~a] = -np.inf
        indices, values = zip(*[(self._private_items.get(u_list[0]), u_list[1])
                              for u_list in enumerate(b.data)])

        indices = np.array(indices)
        values = np.array(values)
        local_k = min(top_k, len(values))
        partially_ordered_preds_indices = np.argpartition(values, -local_k)[-local_k:]
        real_values = values[partially_ordered_preds_indices]
        real_indices = indices[partially_ordered_preds_indices]
        local_top_k = real_values.argsort()[::-1]
        return [(real_indices[item], real_values[item]) for item in local_top_k]

    def train_step(self, batch, **kwargs):
        for u, i, j in zip(*batch):
            self.update_factors(u[0], i[0], j[0])

    def update_factors(self, ui: int, ii: int, ji: int):
        user_factors = self._user_factors[ui]
        item_factors_i = self._item_factors[ii]
        item_factors_j = self._item_factors[ji]
        item_bias_i = self._item_bias[ii]
        item_bias_j = self._item_bias[ji]

        z = 1/(1 + np.exp(self.indexed_predict(ui, ii)-self.indexed_predict(ui, ji)))
        # update bias i
        d_bi = (z - self._bias_regularization*item_bias_i)
        self._item_bias[ii] = item_bias_i + (self._learning_rate * d_bi)

        # update bias j
        d_bj = (-z - self._bias_regularization*item_bias_j)
        self._item_bias[ji] = item_bias_j + (self._learning_rate * d_bj)

        # update user factors
        d_u = ((item_factors_i - item_factors_j)*z - self._user_regularization*user_factors)
        self._user_factors[ui] = user_factors + (self._learning_rate * d_u)

        # update item i factors
        d_i = (user_factors*z - self._positive_item_regularization*item_factors_i)
        self._item_factors[ii] = item_factors_i + (self._learning_rate * d_i)

        # update item j factors
        d_j = (-user_factors*z - self._negative_item_regularization*item_factors_j)
        self._item_factors[ji] = item_factors_j + (self._learning_rate * d_j)
