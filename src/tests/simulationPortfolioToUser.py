#!/usr/bin/python3

import os
from typing import List

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

from portfolioDescription.portfolio1MethDescription import Portfolio1MethDescription #class

from input.inputRecomDefinition import InputRecomDefinition #class

from portfolioDescription.aPortfolioDescription import APortfolioDescription #class

from evaluationTool.evalToolSingleMethod import EToolSingleMethod #class

from recommenderDescription.recommenderDescription import RecommenderDescription #class

from input.inputSimulatorDefinition import InputSimulatorDefinition #class

from history.historyHierDF import HistoryHierDF #class

import pandas as pd
from input.aBatch import BatchParameters #class
from input.aBatchML import ABatchML #class


def test01():

    df:DataFrame = DataFrame({'$a':[2783,2783,2783,3970, 3970], '$b':[1909,1396,2901,3408,3407], '$c':[2,4,5,4,1]})
    df.columns = [Ratings.COL_USERID, Ratings.COL_MOVIEID, Ratings.COL_RATING]
    print(df)
    print("--------------")

    m:ModelOfIndexes = ModelOfIndexes(df, Ratings)

    a1:int = m.getNextIndex(2783, 1909)
    print(a1)

    a2:int = m.getNextIndex(2783, 1396)
    print(a2)

    a3:int = m.getNextIndex(2783, 2901)
    print(a3)
    print("--------------")

    a4:int = m.getNextIndex(3970, 3408)
    print(a4)



def test02():
    ratingsDF:DataFrame = DataFrame({'$a':[2783,2783,2783,3970,3970], '$b':[1909,1396,2901,3408,3407], '$c':[2,4,5,4,1],
                                     '$d':[100,101,102,103,104]})
    ratingsDF.columns = [Ratings.COL_USERID, Ratings.COL_MOVIEID, Ratings.COL_RATING, Ratings.COL_TIMESTAMP]
    print(ratingsDF)
    print("")

    behaviourDF:DataFrame = DataFrame({'$a':[2783,2783,2783,2783,2783,2783, 3970,3970,3970,3970], '$b':[1909,1909,1396,1396,2901,2901,2901,2901,3407,3407], '$c':[0,1,0,1,0,1,0,1,0,1],
                                       '$d':["b0000000000", "b0000000000", "b0000000000", "b0000000000", "b0000000000",
                                             "b0000000000", "b0000000000", "b0000000000", "b0000000000", "b0000000000"]})
    behaviourDF.columns = [Behaviours.COL_USERID, Behaviours.COL_MOVIEID, Behaviours.COL_REPETITION, Behaviours.COL_BEHAVIOUR]
    print(behaviourDF)
    print("")

    dataset:ADataset = DatasetML(ratingsDF, DataFrame(), DataFrame())
    divisionDatasetPercentualSize:int = 50
    testDatasetPercentualSize:int = 50
    recomRepetitionCount:int = 2

    trainDataset:ADataset
    testRatingsDF:DataFrame
    testRepeatedBehaviourDict:dict
    trainDataset, testRatingsDF, testRepeatedBehaviourDict = SimulationML.divideDataset(dataset, behaviourDF,
                      divisionDatasetPercentualSize, testDatasetPercentualSize, recomRepetitionCount)

    print("---------------------")
    print("trainRatingsDF:")
    print(trainDataset.ratingsDF)
    print("")

    print("testRatingsDF:")
    print(testRatingsDF)
    print("")

    print("")
    print(testRepeatedBehaviourDict[0])
    print("")

    print("")
    print(testRepeatedBehaviourDict[1])
    print("")



def test03():

    rDescr:RecommenderDescription = InputRecomDefinition.exportRDescTheMostPopular()

    pDescr:APortfolioDescription = Portfolio1MethDescription(InputRecomDefinition.THE_MOST_POPULAR.title(),
                                    InputRecomDefinition.THE_MOST_POPULAR, rDescr)

    batchID:str = "ml1mDiv90Ulinear0109R1"
    dataset:DatasetRetailRocket = DatasetRetailRocket.readDatasets()
    behaviourFile:str = BehavioursRR.getFile(Behaviours.BHVR_LINEAR0109)
    behavioursDF:DataFrame = Behaviours.readFromFileMl1m(behaviourFile)

    argsSimulationDict: dict = {SimulationML.ARG_WINDOW_SIZE: 5,
                                SimulationML.ARG_RECOM_REPETITION_COUNT: 1,
                                SimulationML.ARG_NUMBER_OF_RECOMM_ITEMS: 100,
                                SimulationML.ARG_NUMBER_OF_AGGR_ITEMS: InputSimulatorDefinition.numberOfAggrItems,
                                SimulationML.ARG_DIV_DATASET_PERC_SIZE: 90,
                                SimulationML.ARG_HISTORY_LENGTH: 10}

    # remove old results
    path:str = ".." + os.sep + "results" + os.sep + batchID
    try:
        os.remove(path + os.sep + "computation-theMostPopular.txt")
        os.remove(path + os.sep + "historyOfRecommendation-theMostPopular.txt")
        os.remove(path + os.sep + "portfModelTimeEvolution-theMostPopular.txt")
    except:
        print("An exception occurred")

    # simulation of portfolio
    simulator:Simulator = Simulator(batchID, SimulationML, argsSimulationDict, dataset, behavioursDF)
    simulator.simulate([pDescr], [DataFrame()], [EToolSingleMethod({})], HistoryHierDF)




def test04():

    rDescr:RecommenderDescription = InputRecomDefinition.exportRDescTheMostPopular()

    pDescr:APortfolioDescription = Portfolio1MethDescription(InputRecomDefinition.THE_MOST_POPULAR.title(),
                                    InputRecomDefinition.THE_MOST_POPULAR, rDescr)

    batchID:str = "retailrocketDiv90Ulinear0109R1"
    dataset:DatasetRetailRocket = DatasetRetailRocket.readDatasets()
    behaviourFile:str = BehavioursRR.getFile(BehavioursRR.BHVR_LINEAR0109)
    behavioursDF:DataFrame = BehavioursRR.readFromFileRR(behaviourFile)

    argsSimulationDict:dict = {SimulationML.ARG_WINDOW_SIZE: 5,
                                SimulationML.ARG_RECOM_REPETITION_COUNT: 1,
                                SimulationML.ARG_NUMBER_OF_RECOMM_ITEMS: 100,
                                SimulationML.ARG_NUMBER_OF_AGGR_ITEMS: InputSimulatorDefinition.numberOfAggrItems,
                                SimulationML.ARG_DIV_DATASET_PERC_SIZE: 90,
                                SimulationML.ARG_HISTORY_LENGTH: 10}

    # remove old results
    path:str = ".." + os.sep + "results" + os.sep + batchID
    try:
        os.remove(path + os.sep + "computation-theMostPopular.txt")
        os.remove(path + os.sep + "historyOfRecommendation-theMostPopular.txt")
        os.remove(path + os.sep + "portfModelTimeEvolution-theMostPopular.txt")
    except:
        print("An exception occurred")

    # simulation of portfolio
    simulator:Simulator = Simulator(batchID, SimulationRR, argsSimulationDict, dataset, behavioursDF)
    simulator.simulate([pDescr], [DataFrame()], [EToolSingleMethod({})], HistoryHierDF)



def test05():

    rDescr:RecommenderDescription = InputRecomDefinition.exportRDescTheMostPopular()

    pDescr:APortfolioDescription = Portfolio1MethDescription(InputRecomDefinition.THE_MOST_POPULAR.title(),
                                    InputRecomDefinition.THE_MOST_POPULAR, rDescr)

    batchID:str = "slantourDiv90Ulinear0109R1"
    dataset:DatasetST = DatasetST.readDatasets()
    behaviourFile:str = BehavioursST.getFile(BehavioursST.BHVR_LINEAR0109)
    behavioursDF:DataFrame = BehavioursST.readFromFileST(behaviourFile)

    argsSimulationDict:dict = {SimulationST.ARG_WINDOW_SIZE: 5,
                                SimulationST.ARG_RECOM_REPETITION_COUNT: 1,
                                SimulationST.ARG_NUMBER_OF_RECOMM_ITEMS: 100,
                                SimulationST.ARG_NUMBER_OF_AGGR_ITEMS: InputSimulatorDefinition.numberOfAggrItems,
                                SimulationST.ARG_DIV_DATASET_PERC_SIZE: 90,
                                SimulationST.ARG_HISTORY_LENGTH: 10}

    # remove old results
    path:str = ".." + os.sep + "results" + os.sep + batchID
    try:
        os.remove(path + os.sep + "computation-theMostPopular.txt")
        os.remove(path + os.sep + "historyOfRecommendation-theMostPopular.txt")
        os.remove(path + os.sep + "portfModelTimeEvolution-theMostPopular.txt")
    except:
        print("An exception occurred")

    # simulation of portfolio
    simulator:Simulator = Simulator(batchID, SimulationST, argsSimulationDict, dataset, behavioursDF)
    simulator.simulate([pDescr], [DataFrame()], [EToolSingleMethod({})], HistoryHierDF)



if __name__ == "__main__":
    os.chdir("..")
    #test01()
    #test02()
    #test03()
    #test04()
    test05()