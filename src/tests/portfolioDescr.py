#!/usr/bin/python3

import os

#from pandas.core.series import Series #class

#from aggregation.negImplFeedback.penalUsingFiltering import PenalUsingFiltering #class
#from aggregation.negImplFeedback.aPenalization import APenalization #class
from pandas import DataFrame

from history.aHistory import AHistory #class
from history.historyDF import HistoryDF #class

from datasets.aDataset import ADataset #class
from datasets.datasetST import DatasetST #class
from datasets.slantour.events import Events #class

from recommender.aRecommender import ARecommender #class
from recommender.recommenderTheMostPopular import RecommenderTheMostPopular #class
from recommender.recommenderCosineCB import RecommenderCosineCB #class
from recommender.recommenderW2V import RecommenderW2V #class
from recommender.recommenderItemBasedKNN import RecommenderItemBasedKNN #class
from recommender.recommenderBPRMF import RecommenderBPRMF #class

from portfolio.aPortfolio import APortfolio #class

from portfolioDescription.portfolio1MethDescription import Portfolio1MethDescription #class

from recommenderDescription.recommenderDescription import RecommenderDescription #class



def test01():
    print("Test 01")

    rDescr:RecommenderDescription = RecommenderDescription(RecommenderTheMostPopular, {})

    recommenderID:str = "TheMostPopular"
    pDescr:Portfolio1MethDescription = Portfolio1MethDescription(recommenderID.title(), recommenderID, rDescr)

    dataset:ADataset = DatasetST.readDatasets()

    history:AHistory = HistoryDF("test")
    p:APortfolio = pDescr.exportPortfolio("jobID", history)

    portFolioModel:DataFrame = DataFrame()

    p.train(history, dataset)

    df:DataFrame = DataFrame([[1, 555]], columns=[Events.COL_USER_ID, Events.COL_OBJECT_ID])

    p.update(df)

    userID:int = 1
    r = p.recommend(userID, portFolioModel, {APortfolio.ARG_NUMBER_OF_AGGR_ITEMS:20})
    print(r)



if __name__ == "__main__":
    os.chdir("..")

    test01()