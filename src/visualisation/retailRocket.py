#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np

from typing import List

import os

from pandas.core.frame import DataFrame #class
from pandas.core.frame import Series #class

import pandas as pd

from evaluationTool.evalToolContext import EvalToolContext #class

from datasets.ml.users import Users #class
from datasets.ml.items import Items #class
from datasets.aDataset import ADataset #class
from datasets.datasetML import DatasetML #class
from datasets.datasetST import DatasetST #class

from datasets.retailrocket.events import Events #class

from recommender.aRecommender import ARecommender #class
from recommender.recommenderItemBasedKNN import RecommenderItemBasedKNN #class
from recommender.recommenderTheMostPopular import RecommenderTheMostPopular #class

from history.aHistory import AHistory #class
from history.historyDF import HistoryDF #class

from datasets.ml.users import Users #class
from datasets.ml.items import Items #class
from datasets.ml.ratings import Ratings #class

from aggregation.aggrContextFuzzyDHondt import AggrContextFuzzyDHondt #class
from aggregation.aggrFuzzyDHondt import AggrFuzzyDHondt
from aggregation.operators.theMostVotedItemSelector import TheMostVotedItemSelector #class


def test01():

    DEBUG:bool = False

    # First get Dataset Data
    eventsDF:DataFrame = Events.readFromFile()
    eventsDF = eventsDF.sort_values(Events.COL_TIME_STAMP)


    userIDs:List[int] = list(eventsDF[Events.COL_VISITOR_ID].unique())
    #eventsIDs:List[int] = list(eventsDF[Events.COL_EVENT].unique())

    print("Number of all events:                             " + str(len(eventsDF)))
    print("Number of all usersIDs:                           " + str(len(userIDs)))
    print("Number of all addtocart:                            " + str(len(eventsDF.loc[eventsDF[Events.COL_EVENT] == "addtocart"])))
    print("Number of all transaction:                          " + str(len(eventsDF.loc[eventsDF[Events.COL_EVENT] == "transaction"])))
    print("---------------------------------------------------------")

    # kolik vsech uziatelu si vlozilo do kosiku item vice nez jednou
    eventsACartDF:DataFrame = eventsDF.loc[eventsDF[Events.COL_EVENT] == "addtocart"]
    eventsACart2DF = eventsACartDF.groupby(
        [Events.COL_VISITOR_ID, Events.COL_ITEM_ID], as_index=False)[Events.COL_TIME_STAMP].count()
    eventsACart2DF = eventsACart2DF.loc[eventsACart2DF[Events.COL_TIME_STAMP] > 1]

    # kolik vsech uzivatelu si pridalo 1-N polozek
    usersAndAddedItemsCountDF:DataFrame = eventsACartDF[[Events.COL_VISITOR_ID, Events.COL_ITEM_ID]].drop_duplicates().groupby(
        [Events.COL_VISITOR_ID], as_index=False)[Events.COL_ITEM_ID].count()

    plt.hist(list(usersAndAddedItemsCountDF[Events.COL_ITEM_ID]))
    plt.yscale('log')
    if DEBUG:
        plt.show()


    userIDs:List[int] = list(eventsDF[Events.COL_VISITOR_ID].unique())
    print("Number of all users:                              " + str(len(userIDs)))

    print("Number of all users 1< #itemI added:                 " + str(len(eventsACart2DF)))
    print("---------------------------------------------------------")


    # filtrace eventu - vymazeme uzivatele, kteri maji mene nez K zaznamu
    userIdAndTimestampDF:DataFrame[int, int] = eventsDF.groupby(
        [Events.COL_VISITOR_ID], as_index=False)[Events.COL_TIME_STAMP].count()

    userIdAndTimestampSelDF:DataFrame[int, int] = userIdAndTimestampDF.loc[
        userIdAndTimestampDF[Events.COL_TIME_STAMP] > 5]
    #print(userIdAndTimestampSelDF)
    userIDsSel:List[int] = list(userIdAndTimestampSelDF[Events.COL_VISITOR_ID].unique())

    eventsSelDF:DataFrame = eventsDF.loc[eventsDF[Events.COL_VISITOR_ID].isin(userIDsSel)]

    print("Number of selected events:                         " + str(len(eventsSelDF)))
    print("Number of selected usersIDs:                         " + str(len(userIDsSel)))
    print("Number of all addtocart:                            " + str(len(eventsSelDF.loc[eventsSelDF[Events.COL_EVENT] == "addtocart"])))
    print("Number of selected transaction:                      " + str(len(eventsSelDF.loc[eventsSelDF[Events.COL_EVENT] == "transaction"])))
    print("---------------------------------------------------------")



    # kolik vybranych uzivatelu si vlozilo do kosiku item aspon jednou
    eventsSelACartDF:DataFrame = eventsSelDF.loc[eventsSelDF[Events.COL_EVENT] == "addtocart"]
    userIDSelsACart:List[int] = list(eventsSelACartDF[Events.COL_VISITOR_ID].unique())

    eventsSelACart1DF = eventsSelACartDF.groupby(
        [Events.COL_VISITOR_ID, Events.COL_ITEM_ID], as_index=False)[Events.COL_TIME_STAMP].count()
    eventsSelACart1DF = eventsSelACart1DF.loc[eventsSelACart1DF[Events.COL_TIME_STAMP] > 0]

    print("Number of selected users:                            " + str(len(userIDsSel)))
    print("Number of selected users who added something to cart: " + str(len(userIDSelsACart)))

    print("---------------------------------------------------------")



    # kolik vybranych uzivatelu koupilo vice jak dva ne nutne ruzne itemy
    eventsSelACartDF:DataFrame = eventsSelDF.loc[eventsSelDF[Events.COL_EVENT] == "addtocart"]
    eventsSelACartDF_ = eventsSelACartDF.groupby(
        [Events.COL_VISITOR_ID], as_index=False)[Events.COL_TIME_STAMP].count()

    eventsSelACart2DF = eventsSelACartDF_.loc[eventsSelACartDF_[Events.COL_TIME_STAMP] > 1]
    userSelACart2IDs:List[int] = list(eventsSelACart2DF[Events.COL_VISITOR_ID])

    print("Number of selected users 1< #addItem:                 " + str(len(userSelACart2IDs)))

    eventsSelACart1DF = eventsSelACartDF_.loc[eventsSelACartDF_[Events.COL_TIME_STAMP] == 1]
    userSelACart1IDs:List[int] = list(eventsSelACart1DF[Events.COL_VISITOR_ID])

    print("Number of selected users 1= #addItem:                 " + str(len(userSelACart1IDs)))



    # kolik vybranych uzivatelu si pridalo jednu jedinou polozku libovolnekrat
    usersSelAndAddedItemsCountDF:DataFrame = eventsSelACartDF[[Events.COL_VISITOR_ID, Events.COL_ITEM_ID]].drop_duplicates().groupby(
        [Events.COL_VISITOR_ID], as_index=False)[Events.COL_ITEM_ID].count()

    plt.hist(list(usersSelAndAddedItemsCountDF[Events.COL_ITEM_ID]), bins=700)
    if DEBUG:
        plt.show()

    a = len(usersSelAndAddedItemsCountDF.loc[usersSelAndAddedItemsCountDF[Events.COL_ITEM_ID] == 1])
    print("Number of selected users who added only 1 unique itemI (possibly many times):     " + str(a))



    print("---------------------------------------------------------")

    # kolik procent uzivatelu si koupilo jen jednu vec vicekrat
    b = (a - len(userSelACart1IDs)) / len(userSelACart2IDs) * 100
    print(str(b) + " %")

    # zajimalo by me kolikrat si uzivatele kupovali stejnou vec
    eventsTransDF:DataFrame = eventsDF.loc[eventsDF[Events.COL_EVENT] == "transaction"]
    events2DF = eventsTransDF.groupby(
        [Events.COL_VISITOR_ID, Events.COL_ITEM_ID], as_index=False)[Events.COL_TIME_STAMP].count()
    events2DF = events2DF.loc[events2DF[Events.COL_TIME_STAMP] > 4]
    #print(events2DF.head(10))

    events152963DF:DataFrame = eventsDF.loc[eventsDF[Events.COL_VISITOR_ID] == 152963]
    events152963DF = events152963DF.loc[events152963DF[Events.COL_ITEM_ID] == 119736]
    #print(events152963DF.head(1000000).to_string())

    events530559DF:DataFrame = eventsDF.loc[eventsDF[Events.COL_VISITOR_ID] == 530559]
    events530559DF = events530559DF.loc[events530559DF[Events.COL_ITEM_ID] == 119736]
    #print(events530559DF.head(1000000).to_string())

    events76757DF:DataFrame = eventsDF.loc[eventsDF[Events.COL_VISITOR_ID] == 76757]
    events76757DF:DataFrame = events76757DF.loc[events76757DF[Events.COL_EVENT] == "transaction"]
    #print(events76757DF.head(1000000).to_string())





if __name__ == "__main__":
    os.chdir("..")

    test01()
    #test02()