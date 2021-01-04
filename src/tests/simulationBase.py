#!/usr/bin/python3

import os
from typing import List
from typing import Dict

from datasets.aDataset import ADataset #class
from datasets.datasetML import DatasetML #class
from datasets.datasetRetailrocket import DatasetRetailRocket #class

from datasets.ml.ratings import Ratings #class
from datasets.ml.behaviours import Behaviours #class
from datasets.retailrocket.behavioursRR import BehavioursRR #class
from datasets.slantour.behavioursST import BehavioursST #class


from datasets.datasetST import DatasetST #class

from pandas.core.frame import DataFrame #class

from simulator.simulator import Simulator #class

from simulation.tools.modelOfIndexes import ModelOfIndexes #class
from simulation.simulationML import SimulationML #class
from simulation.simulationRR import SimulationRR #class
from simulation.simulationST import SimulationST #class

from pandas.core.frame import DataFrame #class

from portfolio.aPortfolio import APortfolio #class
from portfolioDescription.portfolio1MethDescription import Portfolio1MethDescription #class

from input.inputRecomDefinition import InputRecomDefinition #class

from portfolioDescription.aPortfolioDescription import APortfolioDescription #class

from evaluationTool.evalToolSingleMethod import EToolSingleMethod #class

from recommenderDescription.recommenderDescription import RecommenderDescription #class

from input.inputSimulatorDefinition import InputSimulatorDefinition #class

from history.aHistory import AHistory #class
from history.historyHierDF import HistoryHierDF #class
from history.historyDF import HistoryDF #class

from recommender.aRecommender import ARecommender #class
from recommender.recommenderTheMostPopular import RecommenderTheMostPopular #class

import pandas as pd
from input.aBatch import BatchParameters #class
from input.aBatchML import ABatchML #class

import pandas as pd
from numpy.random import randint


def test01():
    print("Test 01")

    rDescr:RecommenderDescription = RecommenderDescription(RecommenderTheMostPopular, {})

    recommenderID:str = "TheMostPopular"
    pDescr:Portfolio1MethDescription = Portfolio1MethDescription(recommenderID.title(), recommenderID, rDescr)

    dataset:ADataset = DatasetML.readDatasets()

    history:AHistory = HistoryDF("test")
    p:APortfolio = pDescr.exportPortfolio("jobID", history)
    p.train(history, dataset)

#    r, rwr = p.recommend(1, DataFrame(), {APortfolio.ARG_NUMBER_OF_AGGR_ITEMS:20})
#    rItemID1 = r[0]
#    rItemID2 = r[1]
#    rItemID3 = r[2]
#
#    print(r)
#    print("rItemID1: " + str(rItemID1))
#    print("rItemID2: " + str(rItemID2))
#    print("rItemID3: " + str(rItemID3))

    testRatingsDF:DataFrame = DataFrame(columns=[Ratings.COL_USERID, Ratings.COL_MOVIEID, Ratings.COL_RATING, Ratings.COL_TIMESTAMP])
    timeStampI:int = 1000

    userID1:int = 1
    userID2:int = 2
    userID3:int = 3
    rItemID1:int = 9001
    rItemID2:int = 9002
    rItemID3:int = 9003
    # training part of dataset
    for i in [i + 0 for i in range(5*8)]:
        timeStampI = timeStampI +1
        testRatingsDF.loc[i] = [userID1] + list([9000, 5, timeStampI])
    timeStampI = timeStampI +1
    testRatingsDF.loc[len(testRatingsDF)] = [userID2] + list([rItemID1, 5, timeStampI])
    timeStampI = timeStampI +1
    testRatingsDF.loc[len(testRatingsDF)] = [userID2] + list([rItemID2, 5, timeStampI])
    timeStampI = timeStampI +1
    testRatingsDF.loc[len(testRatingsDF)] = [userID3] + list([rItemID3, 5, timeStampI])
    timeStampI = timeStampI +1
    testRatingsDF.loc[len(testRatingsDF)] = [userID2] + list([rItemID2, 5, timeStampI])
    timeStampI = timeStampI +1
    testRatingsDF.loc[len(testRatingsDF)] = [userID2] + list([rItemID2, 5, timeStampI])

    # testing part of dataset
    userID11:int = 11
    timeStampI = timeStampI +1
    testRatingsDF.loc[len(testRatingsDF)] = [userID11] + list([rItemID1, 5, timeStampI])
    timeStampI = timeStampI +1
    testRatingsDF.loc[len(testRatingsDF)] = [userID11] + list([rItemID2, 5, timeStampI])
    timeStampI = timeStampI +1
    testRatingsDF.loc[len(testRatingsDF)] = [userID11] + list([rItemID3, 5, timeStampI])
    timeStampI = timeStampI +1
    testRatingsDF.loc[len(testRatingsDF)] = [userID11] + list([rItemID2, 5, timeStampI])
    timeStampI = timeStampI +1
    testRatingsDF.loc[len(testRatingsDF)] = [userID11] + list([rItemID2, 5, timeStampI])

    print("len(testRatingsDF): " + str(len(testRatingsDF)))
    print(testRatingsDF.head(20))
    print(testRatingsDF.tail(20))

    datasetMy:ADataset = DatasetML("", testRatingsDF, dataset.usersDF, dataset.itemsDF)

    behavioursDF:DataFrame = DataFrame(columns=[Behaviours.COL_REPETITION, Behaviours.COL_BEHAVIOUR])
    for ratingIndexI in range(len(testRatingsDF)):
        for repetitionI in range(5):
            behavioursDF.loc[ratingIndexI*5 + repetitionI] = list([repetitionI, [True]*20])
    print(behavioursDF.head(20))


    argsSimulationDict:Dict[str,str] = {SimulationML.ARG_WINDOW_SIZE: 5,
                                SimulationML.ARG_RECOM_REPETITION_COUNT: 1,
                                SimulationML.ARG_NUMBER_OF_RECOMM_ITEMS: 100,
                                SimulationML.ARG_NUMBER_OF_AGGR_ITEMS: InputSimulatorDefinition.numberOfAggrItems,
                                SimulationML.ARG_DIV_DATASET_PERC_SIZE: 90,
                                SimulationML.ARG_HISTORY_LENGTH: 10}

    # simulation of portfolio
    simulator:Simulator = Simulator("test", SimulationML, argsSimulationDict, datasetMy, behavioursDF)
    simulator.simulate([pDescr], [DataFrame()], [EToolSingleMethod({})], HistoryHierDF)


if __name__ == "__main__":
    os.chdir("..")

    test01()