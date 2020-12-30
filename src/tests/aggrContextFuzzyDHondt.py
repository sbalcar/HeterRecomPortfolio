#!/usr/bin/python3

import os

from typing import List

from pandas.core.series import Series #class
from pandas.core.frame import DataFrame #class
import pandas as pd

from aggregation.aggrContextFuzzyDHondt import AggrContextFuzzyDHondt #class

from history.aHistory import AHistory #class
from history.historyDF import HistoryDF #class

from datasets.ml.users import Users #class
from datasets.ml.items import Items #class
from datasets.ml.ratings import Ratings #class

from recommender.aRecommender import ARecommender #class
from recommender.recommenderItemBasedKNN import RecommenderItemBasedKNN #class
from recommender.recommenderTheMostPopular import RecommenderTheMostPopular #class

from datasets.aDataset import ADataset #class
from datasets.datasetML import DatasetML #class

from configuration.configuration import Configuration #class


from aggregation.negImplFeedback.penalUsingReduceRelevance import penaltyLinear #function

from aggregation.negImplFeedback.penalUsingReduceRelevance import PenalUsingReduceRelevance #class
from aggregation.negImplFeedback.aPenalization import APenalization #class

from aggregation.operators.theMostVotedItemSelector import TheMostVotedItemSelector #class

def test01():
    print("Test 01")

    # number of recommended items
    N = 120

    methodsResultDict:dict[str,pd.Series] = {
          "metoda1":pd.Series([0.2,0.1,0.3,0.3,0.1],[32,2,8,1,4],name="rating"),
          "metoda2":pd.Series([0.1,0.1,0.2,0.3,0.3],[1,5,32,6,7],name="rating"),
          "metoda3":pd.Series([0.3,0.1,0.2,0.3,0.1],[7,2,77,64,12],name="rating")
          }

    itemsDF: DataFrame = Items.readFromFileMl1m()
    usersDF: DataFrame = Users.readFromFileMl1m()
    ratingsDF: DataFrame = Ratings.readFromFileMl1m()

    # methods parametes
    methodsParamsData:List[tuple] = [['metoda1',100], ['metoda2',80], ['metoda3',60]]
    methodsParamsDF:DataFrame = pd.DataFrame(methodsParamsData, columns=["methodID", "votes"])
    methodsParamsDF.set_index("methodID", inplace=True)

    userID:int = 1

    historyDF:AHistory = HistoryDF("test01")

    # TODO: What is ARG_SELECTOR?
    aggr:AggrContextFuzzyDHondt = AggrContextFuzzyDHondt(historyDF, {AggrContextFuzzyDHondt.ARG_SELECTOR:TheMostVotedItemSelector({}),
                                                                     AggrContextFuzzyDHondt.ARG_USERS:usersDF,
                                                                     AggrContextFuzzyDHondt.ARG_ITEMS:itemsDF,
                                                                     AggrContextFuzzyDHondt.ARG_DATASET:"ml"})

    itemIDs = aggr.runWithResponsibility(methodsResultDict, methodsParamsDF, userID, N)

    print("recommended items:")
    for itemID, votes in itemIDs:
        film_info: DataFrame = itemsDF[itemsDF['movieId'] == itemID]
        print('\t', film_info['movieTitle'].to_string(header=False),
                  film_info['Genres'].to_string(header=False))
    print()
    print("===========================END OF RECOMMENDATION LIST===========================")
    print("Ratings:")
    resultList = None
    for index, row in ratingsDF.iloc[-100:].iterrows():
        aggr.update(ratingsDF.iloc[index:index+1])
        resultList = aggr.runWithResponsibility(methodsResultDict, methodsParamsDF, row['userId'],N)
        film_info: DataFrame = itemsDF[itemsDF['movieId'] == ratingsDF.iloc[index]['movieId']]
        print('\t', film_info['movieTitle'].to_string(header=False),
              film_info['Genres'].to_string(header=False))
    itemIDs = aggr.runWithResponsibility(methodsResultDict, methodsParamsDF, ratingsDF.iloc[-1:]['userId'].item(), N)

    print("recommended items:")
    for itemID, votes in itemIDs:
        film_info: DataFrame = itemsDF[itemsDF['movieId'] == itemID]
        print('\t', film_info['movieTitle'].to_string(header=False),
              film_info['Genres'].to_string(header=False))

    print()
    print("===========================END OF TEST01===========================")
    print()
    print()


def test02():
    print("Test 02")

    # number of recommended items
    N = 100

    # get dataset
    itemsDF: DataFrame = Items.readFromFileMl1m()
    usersDF: DataFrame = Users.readFromFileMl1m()
    ratingsDF: DataFrame = Ratings.readFromFileMl1m()

    ratingsDFTrain = ratingsDF[:50000]
    ratingsDFUpdate: DataFrame = ratingsDF.iloc[50001:50100]

    trainDataset:ADataset = DatasetML(ratingsDFTrain, usersDF, itemsDF)

    historyDF: AHistory = HistoryDF("test01")

    # train KNN
    rec1: ARecommender = RecommenderItemBasedKNN("run", {})
    rec1.train(HistoryDF("test01"), trainDataset)

    #train Most Popular
    rec2: ARecommender = RecommenderTheMostPopular("run", {})
    rec2.train(historyDF, trainDataset)

    # methods parametes
    methodsParamsData: List[tuple] = [['ItembasedKNN', 100], ['MostPopular', 80]]
    methodsParamsDF: DataFrame = pd.DataFrame(methodsParamsData, columns=["methodID", "votes"])
    methodsParamsDF.set_index("methodID", inplace=True)

    userID: int = 1

    # TODO: What is ARG_SELECTOR?
    aggr: AggrContextFuzzyDHondt = AggrContextFuzzyDHondt(historyDF, {
        AggrContextFuzzyDHondt.ARG_SELECTOR: TheMostVotedItemSelector({}),
        AggrContextFuzzyDHondt.ARG_USERS: usersDF,
        AggrContextFuzzyDHondt.ARG_ITEMS: itemsDF,
        AggrContextFuzzyDHondt.ARG_DATASET: "ml"})

    r1: Series = rec1.recommend(userID, N, {})

    r2: Series = rec2.recommend(userID, N, {})

    r1.name = 'rating'
    r2.name = 'rating'
    methodsResultDict: dict[str, pd.Series] = {
        "ItembasedKNN": r1,
        "MostPopular": r2
    }

    itemIDs = aggr.runWithResponsibility(methodsResultDict, methodsParamsDF, userID, N)

    print("recommended items:")
    for itemID, votes in itemIDs:
        film_info: DataFrame = itemsDF[itemsDF['movieId'] == itemID]
        print('\t', film_info['movieTitle'].to_string(header=False),
              film_info['Genres'].to_string(header=False))
    print()
    print("===========================END OF RECOMMENDATION LIST===========================")
    print("Ratings:")

    # ratings from ratingsDFUpdate are mostly from the same user, so we actually simulate how
    # aggregator behaves on multiple feedbacks given by single user
    userID = 1
    randomItem = 1
    # TODO: what about the rewards?

    for index,row in ratingsDFUpdate.iterrows():
        updateData = pd.DataFrame([[userID, randomItem, 4, 4]],columns=['userId','movieId','rating','timestamp'])
        aggr.update(updateData)
        rec1.update(updateData)
        rec2.update(updateData)
        r1: Series = rec1.recommend(userID, N, {})
        r2: Series = rec2.recommend(userID, N, {})
        r1.name = 'rating'
        r2.name = 'rating'
        methodsResultDict: dict[str, pd.Series] = {
            "ItembasedKNN": r1,
            "MostPopular": r2
        }
        x = aggr.runWithResponsibility(methodsResultDict, methodsParamsDF, userID, N)
        import random
        randomItem = random.choice(x)[0]
        film_info: DataFrame = itemsDF[itemsDF['movieId'] == ratingsDF.iloc[index]['movieId']]
        print('\t', film_info['movieTitle'].to_string(header=False),
              film_info['Genres'].to_string(header=False))

    itemIDs = aggr.runWithResponsibility(methodsResultDict, methodsParamsDF, ratingsDF.iloc[-1:]['userId'].item(), N)

    print("recommended items:")
    for itemID, votes in itemIDs:
        film_info: DataFrame = itemsDF[itemsDF['movieId'] == itemID]
        print('\t', film_info['movieTitle'].to_string(header=False),
              film_info['Genres'].to_string(header=False))

    print()
    print("===========================END OF TEST02===========================")

def test03():
    print("Test 03")

    # number of recommended items
    N = 100

    # get dataset
    itemsDF: DataFrame = Items.readFromFileMl1m()
    usersDF: DataFrame = Users.readFromFileMl1m()
    ratingsDF: DataFrame = Ratings.readFromFileMl1m()

    ratingsDFTrain = ratingsDF[:50000]
    ratingsDFUpdate: DataFrame = ratingsDF.iloc[50001:50100]

    trainDataset: ADataset = DatasetML(ratingsDFTrain, usersDF, itemsDF)

    historyDF: AHistory = HistoryDF("test01")

    # train KNN
    rec1: ARecommender = RecommenderItemBasedKNN("run", {})
    rec1.train(HistoryDF("test01"), trainDataset)

    # train Most Popular
    rec2: ARecommender = RecommenderTheMostPopular("run", {})
    rec2.train(historyDF, trainDataset)

    # methods parametes
    methodsParamsData: List[tuple] = [['ItembasedKNN', 100], ['MostPopular', 80]]
    methodsParamsDF: DataFrame = pd.DataFrame(methodsParamsData, columns=["methodID", "votes"])
    methodsParamsDF.set_index("methodID", inplace=True)

    userID: int = 1

    # TODO: What is ARG_SELECTOR?
    aggr: AggrContextFuzzyDHondt = AggrContextFuzzyDHondt(historyDF, {
        AggrContextFuzzyDHondt.ARG_SELECTOR: TheMostVotedItemSelector({}),
        AggrContextFuzzyDHondt.ARG_USERS: usersDF,
        AggrContextFuzzyDHondt.ARG_ITEMS: itemsDF,
        AggrContextFuzzyDHondt.ARG_DATASET: "ml"})

    r1: Series = rec1.recommend(userID, N, {})

    r2: Series = rec2.recommend(userID, N, {})

    r1.name = 'rating'
    r2.name = 'rating'
    methodsResultDict: dict[str, pd.Series] = {
        "ItembasedKNN": r1,
        "MostPopular": r2
    }

    itemIDs = aggr.runWithResponsibility(methodsResultDict, methodsParamsDF, userID, N)

    print("recommended items:")
    for itemID, votes in itemIDs:
        film_info: DataFrame = itemsDF[itemsDF['movieId'] == itemID]
        print('\t', film_info['movieTitle'].to_string(header=False),
              film_info['Genres'].to_string(header=False))
    print()
    print("===========================END OF RECOMMENDATION LIST===========================")
    print("Ratings:")

    # ratings from ratingsDFUpdate are mostly from the same user, so we actually simulate how
    # aggregator behaves on multiple feedbacks given by single user

    for index,row in ratingsDFUpdate.iterrows():
        aggr.update(ratingsDF.iloc[index:index + 1])
        rec1.update(ratingsDF.iloc[index:index + 1])
        rec2.update(ratingsDF.iloc[index:index + 1])
        r1: Series = rec1.recommend(row['userId'], N, {})
        r2: Series = rec2.recommend(row['userId'], N, {})
        r1.name = 'rating'
        r2.name = 'rating'
        methodsResultDict: dict[str, pd.Series] = {
            "ItembasedKNN": r1,
            "MostPopular": r2
        }
        x = aggr.runWithResponsibility(methodsResultDict, methodsParamsDF, row['userId'], N)
        film_info: DataFrame = itemsDF[itemsDF['movieId'] == ratingsDF.iloc[index]['movieId']]
        print('\t', film_info['movieTitle'].to_string(header=False),
              film_info['Genres'].to_string(header=False))
    itemIDs = aggr.runWithResponsibility(methodsResultDict, methodsParamsDF, ratingsDF.iloc[-1:]['userId'].item(), N)

    print("recommended items:")
    for itemID, votes in itemIDs:
        film_info: DataFrame = itemsDF[itemsDF['movieId'] == itemID]
        print('\t', film_info['movieTitle'].to_string(header=False),
              film_info['Genres'].to_string(header=False))

    print()
    print("===========================END OF TEST03===========================")


if __name__ == "__main__":
    os.chdir("..")
    #test01()
    test02()
    #test03()