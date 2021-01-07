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



def test00():
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


argsSimulationDict:dict = {SimulationST.ARG_WINDOW_SIZE: 5,
                            SimulationST.ARG_RECOM_REPETITION_COUNT: 1,
                            SimulationST.ARG_NUMBER_OF_RECOMM_ITEMS: 100,
                            SimulationST.ARG_NUMBER_OF_AGGR_ITEMS: InputSimulatorDefinition.numberOfAggrItems,
                            SimulationST.ARG_DIV_DATASET_PERC_SIZE: 90,
                            SimulationST.ARG_HISTORY_LENGTH: 10}

def test01():

    print("Simulation: ML TheMostPopular")

    rDescr:RecommenderDescription = InputRecomDefinition.exportRDescTheMostPopular()

    pDescr:APortfolioDescription = Portfolio1MethDescription(InputRecomDefinition.THE_MOST_POPULAR.title(),
                                    InputRecomDefinition.THE_MOST_POPULAR, rDescr)

    batchID:str = "ml1mDiv90Ulinear0109R1"
    dataset:DatasetML = DatasetML.readDatasets()
    behaviourFile:str = Behaviours.getFile(Behaviours.BHVR_LINEAR0109)
    behavioursDF:DataFrame = Behaviours.readFromFileMl1m(behaviourFile)

    # remove old results
    #path:str = ".." + os.sep + "results" + os.sep + batchID
    #try:
    #    os.remove(path + os.sep + "computation-theMostPopular.txt")
    #    os.remove(path + os.sep + "historyOfRecommendation-theMostPopular.txt")
    #    os.remove(path + os.sep + "portfModelTimeEvolution-theMostPopular.txt")
    #except:
    #    print("An exception occurred")

    # simulation of portfolio
    simulator:Simulator = Simulator(batchID, SimulationML, argsSimulationDict, dataset, behavioursDF)
    simulator.simulate([pDescr], [DataFrame()], [EToolSingleMethod({})], HistoryHierDF)


def test02():

    print("Simulation: ML W2V")

    rDescr:RecommenderDescription = InputRecomDefinition.exportRDescW2vPositiveMax()

    pDescr:APortfolioDescription = Portfolio1MethDescription("W2vPositiveMax",
                                    "w2vPositiveMax", rDescr)

    batchID:str = "ml1mDiv90Ulinear0109R1"
    dataset:DatasetML = DatasetML.readDatasets()
    behaviourFile:str = Behaviours.getFile(Behaviours.BHVR_LINEAR0109)
    behavioursDF:DataFrame = Behaviours.readFromFileMl1m(behaviourFile)

    # simulation of portfolio
    simulator:Simulator = Simulator(batchID, SimulationML, argsSimulationDict, dataset, behavioursDF)
    simulator.simulate([pDescr], [DataFrame()], [EToolSingleMethod({})], HistoryHierDF)


def test03():

    print("Simulation: ML KNN")

    rDescr:RecommenderDescription = InputRecomDefinition.exportRDescKNN()

    pDescr:APortfolioDescription = Portfolio1MethDescription(InputRecomDefinition.KNN.title(),
                                    InputRecomDefinition.KNN, rDescr)

    batchID:str = "ml1mDiv90Ulinear0109R1"
    dataset:DatasetML = DatasetML.readDatasets()
    behaviourFile:str = Behaviours.getFile(Behaviours.BHVR_LINEAR0109)
    behavioursDF:DataFrame = Behaviours.readFromFileMl1m(behaviourFile)

    # simulation of portfolio
    simulator:Simulator = Simulator(batchID, SimulationML, argsSimulationDict, dataset, behavioursDF)
    simulator.simulate([pDescr], [DataFrame()], [EToolSingleMethod({})], HistoryHierDF)


def test04():

    print("Simulation: ML CB")

    #rDescr:RecommenderDescription = InputRecomDefinition.exportRDescCBmean()
    rDescr:RecommenderDescription = InputRecomDefinition.exportRDescCBwindow3()


    #pDescr:APortfolioDescription = Portfolio1MethDescription(InputRecomDefinition.COS_CB_MEAN.title(),
    #                                InputRecomDefinition.COS_CB_MEAN, rDescr)
    pDescr:APortfolioDescription = Portfolio1MethDescription(InputRecomDefinition.COS_CB_WINDOW3.title(),
                                    InputRecomDefinition.COS_CB_WINDOW3, rDescr)


    batchID:str = "ml1mDiv90Ulinear0109R1"
    dataset:DatasetML = DatasetML.readDatasets()
    behaviourFile:str = Behaviours.getFile(Behaviours.BHVR_LINEAR0109)
    behavioursDF:DataFrame = Behaviours.readFromFileMl1m(behaviourFile)

    # simulation of portfolio
    simulator:Simulator = Simulator(batchID, SimulationML, argsSimulationDict, dataset, behavioursDF)
    simulator.simulate([pDescr], [DataFrame()], [EToolSingleMethod({})], HistoryHierDF)


def test05():

    print("Simulation: ML MF")

    rDescr:RecommenderDescription = InputRecomDefinition.exportRDescBPRMF()

    pDescr:APortfolioDescription = Portfolio1MethDescription(InputRecomDefinition.BPRMF.title(),
                                    InputRecomDefinition.BPRMF, rDescr)


    batchID:str = "ml1mDiv90Ulinear0109R1"
    dataset:DatasetML = DatasetML.readDatasets()
    behaviourFile:str = Behaviours.getFile(Behaviours.BHVR_LINEAR0109)
    behavioursDF:DataFrame = Behaviours.readFromFileMl1m(behaviourFile)

    # simulation of portfolio
    simulator:Simulator = Simulator(batchID, SimulationML, argsSimulationDict, dataset, behavioursDF)
    simulator.simulate([pDescr], [DataFrame()], [EToolSingleMethod({})], HistoryHierDF)


def test11():

    print("Simulation: RR TheMostPopular")

    rDescr:RecommenderDescription = InputRecomDefinition.exportRDescTheMostPopular()

    pDescr:APortfolioDescription = Portfolio1MethDescription(InputRecomDefinition.THE_MOST_POPULAR.title(),
                                    InputRecomDefinition.THE_MOST_POPULAR, rDescr)

    batchID:str = "retailrocketDiv90Ulinear0109R1"
    dataset:DatasetRetailRocket = DatasetRetailRocket.readDatasets()
    behaviourFile:str = BehavioursRR.getFile(BehavioursRR.BHVR_LINEAR0109)
    behavioursDF:DataFrame = BehavioursRR.readFromFileRR(behaviourFile)

    # simulation of portfolio
    simulator:Simulator = Simulator(batchID, SimulationRR, argsSimulationDict, dataset, behavioursDF)
    simulator.simulate([pDescr], [DataFrame()], [EToolSingleMethod({})], HistoryHierDF)




def test21():

    print("Simulation: ST TheMostPopular")

    rDescr:RecommenderDescription = InputRecomDefinition.exportRDescTheMostPopular()

    pDescr:APortfolioDescription = Portfolio1MethDescription(InputRecomDefinition.THE_MOST_POPULAR.title(),
                                    InputRecomDefinition.THE_MOST_POPULAR, rDescr)

    batchID:str = "slantourDiv90Ulinear0109R1"
    dataset:DatasetST = DatasetST.readDatasets()
    behaviourFile:str = BehavioursST.getFile(BehavioursST.BHVR_LINEAR0109)
    behavioursDF:DataFrame = BehavioursST.readFromFileST(behaviourFile)

    # simulation of portfolio
    simulator:Simulator = Simulator(batchID, SimulationST, argsSimulationDict, dataset, behavioursDF)
    simulator.simulate([pDescr], [DataFrame()], [EToolSingleMethod({})], HistoryHierDF)


def test22():

    print("Simulation: ST W2V")

    rDescr:RecommenderDescription = InputRecomDefinition.exportRDescW2vPosnegMean()

    pDescr:APortfolioDescription = Portfolio1MethDescription(InputRecomDefinition.W2V_POSNEG_MEAN.title(),
                                    InputRecomDefinition.W2V_POSNEG_MEAN, rDescr)

    batchID:str = "slantourDiv90Ulinear0109R1"
    dataset:DatasetST = DatasetST.readDatasets()
    behaviourFile:str = BehavioursST.getFile(BehavioursST.BHVR_LINEAR0109)
    behavioursDF:DataFrame = BehavioursST.readFromFileST(behaviourFile)

    # simulation of portfolio
    simulator:Simulator = Simulator(batchID, SimulationST, argsSimulationDict, dataset, behavioursDF)
    simulator.simulate([pDescr], [DataFrame()], [EToolSingleMethod({})], HistoryHierDF)


def test23():

    print("Simulation: ST KNN")

    rDescr:RecommenderDescription = InputRecomDefinition.exportRDescKNN()

    pDescr:APortfolioDescription = Portfolio1MethDescription(InputRecomDefinition.KNN.title(),
                                    InputRecomDefinition.KNN, rDescr)

    batchID:str = "slantourDiv90Ulinear0109R1"
    dataset:DatasetST = DatasetST.readDatasets()
    behaviourFile:str = BehavioursST.getFile(BehavioursST.BHVR_LINEAR0109)
    behavioursDF:DataFrame = BehavioursST.readFromFileST(behaviourFile)

    # simulation of portfolio
    simulator:Simulator = Simulator(batchID, SimulationST, argsSimulationDict, dataset, behavioursDF)
    simulator.simulate([pDescr], [DataFrame()], [EToolSingleMethod({})], HistoryHierDF)


def test24():

    print("Simulation: ST CB")

    rDescr:RecommenderDescription = InputRecomDefinition.exportRDescCBmean()

    pDescr:APortfolioDescription = Portfolio1MethDescription(InputRecomDefinition.COS_CB_MEAN.title(),
                                    InputRecomDefinition.COS_CB_MEAN, rDescr)

    batchID:str = "slantourDiv90Ulinear0109R1"
    dataset:DatasetST = DatasetST.readDatasets()
    behaviourFile:str = BehavioursST.getFile(BehavioursST.BHVR_LINEAR0109)
    behavioursDF:DataFrame = BehavioursST.readFromFileST(behaviourFile)

    # simulation of portfolio
    simulator:Simulator = Simulator(batchID, SimulationST, argsSimulationDict, dataset, behavioursDF)
    simulator.simulate([pDescr], [DataFrame()], [EToolSingleMethod({})], HistoryHierDF)


def test25():

    print("Simulation: ST MF")

    rDescr:RecommenderDescription = InputRecomDefinition.exportRDescBPRMF()

    pDescr:APortfolioDescription = Portfolio1MethDescription(InputRecomDefinition.BPRMF.title(),
                                    InputRecomDefinition.BPRMF, rDescr)

    batchID:str = "slantourDiv90Ulinear0109R1"
    dataset:DatasetST = DatasetST.readDatasets()
    behaviourFile:str = BehavioursST.getFile(BehavioursST.BHVR_LINEAR0109)
    behavioursDF:DataFrame = BehavioursST.readFromFileST(behaviourFile)

    # simulation of portfolio
    simulator:Simulator = Simulator(batchID, SimulationST, argsSimulationDict, dataset, behavioursDF)
    simulator.simulate([pDescr], [DataFrame()], [EToolSingleMethod({})], HistoryHierDF)



if __name__ == "__main__":
    os.chdir("..")

    #test00()

    # Simulation ML
    #test01()  # TheMostPopular
    #test02()  # W2V
    test03()  # KNNN
    #test04()  # CB
    #test05()  # MF

    # Simulation RR
    #test11()  # TheMostPopular

    # Simulation ST
    #test21()  # TheMostPopular
    #test22()  # W2V
    #test23()  # KNN
    #test24()  # CB
    #test25()  # MF
