#!/usr/bin/python3

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

def test03():
    # First get Dataset Data
    dataset:ADataset = DatasetST.readDatasets()
    events = dataset.eventsDF
    serials = dataset.serialsDF

    # I created some dummy data, but each key,value pair should be result list from a recommender
    #   (=what recommender recommended)
    methodsResultDict: dict = {
        "metoda1": pd.Series([0.2, 0.1, 0.3, 0.3, 0.1], [32, 2, 8, 1, 4], name="rating"),
        "metoda2": pd.Series([0.1, 0.1, 0.2, 0.3, 0.3], [1, 5, 32, 6, 7], name="rating")
    }

    # init votes for each recommender
    portfolioModelData = [['metoda1', 0.6], ['metoda2', 0.4]]
    portfolioModelDF:DataFrame = pd.DataFrame(portfolioModelData, columns=["methodID","votes"])
    portfolioModelDF.set_index("methodID", inplace=True)


    userID = 1
    itemID = 2
    historyDF: AHistory = HistoryDF("test01")

    # WHAT EVALUATIOR NEEDS into dictionary!
    evaluationDict: dict = {EvalToolContext.ARG_USER_ID: userID,    # USERID
                            EvalToolContext.ARG_RELEVANCE: methodsResultDict,   # EACH RECOMMENDER VOTES
                            EvalToolContext.ARG_ITEM_ID: itemID,    # ITEMID (not mandatory if EvalToolContext.ARG_PAGE_TYPE: "zobrazit")
                            EvalToolContext.ARG_SENIORITY: 5,   # SENIORITY OF USER
                            EvalToolContext.ARG_PAGE_TYPE: "zobrazit",  #   TYPE OF PAGE ("zobrazit", "index" or "katalog)
                            EvalToolContext.ARG_ITEMS_SHOWN: 10 # HOW MANY ITEMS ARE SHOWN TO USER
                            }
    # Init eTool
    eToolContext = EvalToolContext(
        {
         EvalToolContext.ARG_ITEMS: serials,    # ITEMS
         EvalToolContext.ARG_EVENTS: events,    # EVENTS (FOR CALCULATING HISTORY OF USER)
         EvalToolContext.ARG_DATASET: "st",     # WHAT DATASET ARE WE IN
         EvalToolContext.ARG_HISTORY: historyDF} # empty instance of AHistory is OK for ST dataset
    )
    aggr: AggrContextFuzzyDHondt = AggrContextFuzzyDHondt(historyDF, {  # empty instance of AHistory is OK for ST dataset
        AggrContextFuzzyDHondt.ARG_EVAL_TOOL: eToolContext, # eTool
        AggrContextFuzzyDHondt.ARG_SELECTOR: TheMostVotedItemSelector({}) # ? FuzzyDHondt needs this, not contextAggr
    })

    # Get data from aggregator
    rItemsWithResponsibility = aggr.runWithResponsibility(methodsResultDict, portfolioModelDF, userID, numberOfItems=5, argumentsDict=evaluationDict)
    # call click & displayed methods
    l1 = eToolContext.displayed(rItemsWithResponsibility, portfolioModelDF, evaluationDict)
    # rItemsWithResponsibility[0][0] is clicked item
    l1 = eToolContext.click(rItemsWithResponsibility,rItemsWithResponsibility[0][0],portfolioModelDF,evaluationDict)

    # ...
    # ...
    # ...
    # user is now on "index" page type, so we have to change page type in evaluationDict (!)
    evaluationDict[EvalToolContext.ARG_PAGE_TYPE] = "index"

    # same as before
    # Get data from aggregator
    rItemsWithResponsibility = aggr.runWithResponsibility(methodsResultDict, portfolioModelDF, userID, numberOfItems=5,
                                                          argumentsDict=evaluationDict)
    # call click & displayed methods
    l1 = eToolContext.displayed(rItemsWithResponsibility, portfolioModelDF, evaluationDict)
    # rItemsWithResponsibility[0][0] is clicked item
    l1 = eToolContext.click(rItemsWithResponsibility, rItemsWithResponsibility[0][0], portfolioModelDF, evaluationDict)


def test01():
    print("Test 01")

    #print("Running Two paralel History Databases:")

    # method results, items=[1,2,4,5,6,7,8,12,32,64,77]
    methodsResultDict:dict = {
        "metoda1": pd.Series([0.2, 0.1, 0.3, 0.3, 0.1], [32, 2, 8, 1, 4], name="rating"),
        "metoda2": pd.Series([0.1, 0.1, 0.2, 0.3, 0.3], [1, 5, 32, 6, 7], name="rating"),
        "metoda3": pd.Series([0.3, 0.1, 0.2, 0.3, 0.1], [7, 2, 77, 64, 12], name="rating")
    }

    rItemIDsWithResponsibility:List = [(7, {'metoda1': 0, 'metoda2': 24.0, 'metoda3': 18.0}), (1, {'metoda1': 30.0, 'metoda2': 8.0, 'metoda3': 0}), (32, {'metoda1': 20.0, 'metoda2': 16.0, 'metoda3': 0}), (8, {'metoda1': 30.0, 'metoda2': 0, 'metoda3': 0}), (6, {'metoda1': 0, 'metoda2': 24.0, 'metoda3': 0}), (64, {'metoda1': 0, 'metoda2': 0, 'metoda3': 18.0}), (2, {'metoda1': 10.0, 'metoda2': 0, 'metoda3': 6.0}), (77, {'metoda1': 0, 'metoda2': 0, 'metoda3': 12.0}), (4, {'metoda1': 10.0, 'metoda2': 0, 'metoda3': 0}), (5, {'metoda1': 0, 'metoda2': 8.0, 'metoda3': 0}), (12, {'metoda1': 0, 'metoda2': 0, 'metoda3': 6.0})]

    # methods parametes
    portfolioModelData = [['metoda1',100], ['metoda2',80], ['metoda3',60]]
    portfolioModelDF:DataFrame = pd.DataFrame(portfolioModelData, columns=["methodID","votes"])
    portfolioModelDF.set_index("methodID", inplace=True)


    itemsDF: DataFrame = Items.readFromFileMl1m()
    usersDF: DataFrame = Users.readFromFileMl1m()

    print("Definition:")
    print(portfolioModelDF)
    print(sumMethods(portfolioModelDF))
    print()

    userID = 1
    itemID = 2
    historyDF: AHistory = HistoryDF("test01")
    historyDF.insertRecommendation(userID, itemID, 1, True, 10)
    historyDF.insertRecommendation(userID, itemID, 1, True, 20)
    historyDF.insertRecommendation(userID, itemID, 1, True, 30)
    historyDF.insertRecommendation(userID, itemID, 1, False, 40)


    evaluationDict:dict = {EvalToolContext.ARG_USER_ID: userID,
                           EvalToolContext.ARG_RELEVANCE: methodsResultDict}
    evalToolDHondt = EvalToolContext(
        {EvalToolContext.ARG_USERS: usersDF,
         EvalToolContext.ARG_ITEMS: itemsDF,
         EvalToolContext.ARG_DATASET: "ml",
         EvalToolContext.ARG_HISTORY: historyDF}
    )

    print("Clicked:")
    evalToolDHondt.click(rItemIDsWithResponsibility, 7, portfolioModelDF, evaluationDict)
    evalToolDHondt.click(rItemIDsWithResponsibility, 1, portfolioModelDF, evaluationDict)
    evalToolDHondt.click(rItemIDsWithResponsibility, 7, portfolioModelDF, evaluationDict)
    print(portfolioModelDF)
    print(sumMethods(portfolioModelDF))
    print()

    print("Displayed - start:")
    for i in range(100):
        evalToolDHondt.displayed(rItemIDsWithResponsibility, portfolioModelDF, evaluationDict)
        print(portfolioModelDF)
        print(sumMethods(portfolioModelDF))
        print()
    print(portfolioModelDF)
    print(sumMethods(portfolioModelDF))
    print("Displayed - end:")
    print()

    print("Clicked:")
    evalToolDHondt.click(rItemIDsWithResponsibility, 4, portfolioModelDF, evaluationDict)
    print(portfolioModelDF)
    print(sumMethods(portfolioModelDF))
    print()

def sumMethods(methods):
    sum = 0
    for index, m in methods.iterrows():
        sum += m[0]
    return sum

def test02(repetitions = 1):
    N = 100

    # get dataset
    itemsDF: DataFrame = Items.readFromFileMl1m()
    usersDF: DataFrame = Users.readFromFileMl1m()
    ratingsDF: DataFrame = Ratings.readFromFileMl1m()

    ratingsDFTrain = ratingsDF[:50000]
    ratingsDFUpdate: DataFrame = ratingsDF.iloc[50001:50100]

    trainDataset: ADataset = DatasetML("ml",ratingsDFTrain, usersDF, itemsDF)

    historyDF: AHistory = HistoryDF("test01")

    # train KNN
    rec1: ARecommender = RecommenderItemBasedKNN("run", {})
    rec1.train(HistoryDF("test01"), trainDataset)

    # train Most Popular
    rec2: ARecommender = RecommenderTheMostPopular("run", {})
    rec2.train(historyDF, trainDataset)

    # methods parametes
    methodsParamsData: List[tuple] = [['ItembasedKNN', 0.4], ['MostPopular', 0.6]]
    methodsParamsDF: DataFrame = pd.DataFrame(methodsParamsData, columns=["methodID", "votes"])
    methodsParamsDF.set_index("methodID", inplace=True)

    userID = 352
    ratingsDFuserID = ratingsDF[ratingsDF['userId'] == userID]
    itemID = ratingsDFuserID.iloc[0]['movieId']

    historyDF: AHistory = HistoryDF("test01")
    historyDF.insertRecommendation(userID, itemID, 1, True, 10)

    r1: Series = rec1.recommend(userID, N, {})
    r2: Series = rec2.recommend(userID, N, {})

    methodsResultDict: dict = {
        "ItembasedKNN": r1,
        "MostPopular": r2
    }
    evaluationDict: dict = {EvalToolContext.ARG_USER_ID: userID,
                            EvalToolContext.ARG_RELEVANCE: methodsResultDict}
    evalToolDHondt = EvalToolContext(
        {EvalToolContext.ARG_USERS: usersDF,
         EvalToolContext.ARG_ITEMS: itemsDF,
         EvalToolContext.ARG_DATASET: "ml",
         EvalToolContext.ARG_HISTORY: historyDF}
    )

    aggr: AggrContextFuzzyDHondt = AggrContextFuzzyDHondt(historyDF, {
        AggrContextFuzzyDHondt.ARG_EVAL_TOOL: evalToolDHondt,
        AggrContextFuzzyDHondt.ARG_SELECTOR: TheMostVotedItemSelector({})
    })
    aggrInit: AggrFuzzyDHondt = AggrFuzzyDHondt(historyDF, {
        AggrFuzzyDHondt.ARG_SELECTOR: TheMostVotedItemSelector({})
    })
    l1 = aggrInit.runWithResponsibility(methodsResultDict, methodsParamsDF, userID, N)
    import random

    evalToolDHondt.displayed(l1, methodsParamsDF, evaluationDict)
    evalToolDHondt.click(l1, random.choice(l1)[0], methodsParamsDF, evaluationDict)
    timestamp = 10
    counter = 0
    r1c = 0
    r2c = 0
    for _ in range(repetitions):
        for index, row in ratingsDFuserID.iterrows():
            r1: Series = rec1.recommend(userID, N, {})
            r2: Series = rec2.recommend(userID, N, {})
            methodsResultDict: dict = {
                "ItembasedKNN": r1,
                "MostPopular": r2
            }
            evalDict = {"a" : 1}
            historyDF.insertRecommendation(userID, row['movieId'], 1, True, timestamp)
            timestamp += 1
            l1 = aggr.runWithResponsibility(methodsResultDict, methodsParamsDF, userID, argumentsDict=evalDict, numberOfItems=N)
            import random
            randomItem = random.choice(l1)[0]
            if randomItem in r1.index:
                r1c += 1
            if randomItem in r2.index:
                r2c += 1
            evaluationDict: dict = {EvalToolContext.ARG_USER_ID: userID,
                                    EvalToolContext.ARG_RELEVANCE: methodsResultDict}
            print("votes Items: ", r1c)
            print("votes mostPopular ", r2c)
            evalToolDHondt.displayed(l1, methodsParamsDF, evaluationDict)
            evalToolDHondt.click(l1, randomItem, methodsParamsDF, evaluationDict)
            rec1.update(ratingsDFuserID.loc[[index]], {})
            # rec2.update(ratingsDFuserID.loc[index]) Not implemented
            #print("Counter = ", counter, "; All = ", len(ratingsDFuserID.iloc[800:]), "; Index: ", index)
            print(methodsParamsDF)
            counter += 1




if __name__ == "__main__":
    os.chdir("..")
    #test01()
    #test02(repetitions=1000)
    test03()