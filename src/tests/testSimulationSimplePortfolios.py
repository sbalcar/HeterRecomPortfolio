#!/usr/bin/python3

import os
from typing import List
from typing import Dict

from datasets.aDataset import ADataset #class
from datasets.datasetML import DatasetML #class
from datasets.datasetRetailrocket import DatasetRetailRocket #class
from datasets.datasetST import DatasetST #class

from datasets.ml.ratings import Ratings #class
from datasets.ml.behavioursML import BehavioursML #class
from datasets.retailrocket.behavioursRR import BehavioursRR #class
from datasets.slantour.behavioursST import BehavioursST #class

from pandas.core.frame import DataFrame #class

from simulator.simulator import Simulator #class

from simulation.tools.modelOfIndexes import ModelOfIndexes #class
from simulation.simulationML import SimulationML #class
from simulation.simulationRR import SimulationRR #class
from simulation.simulationST import SimulationST #class

from pandas.core.frame import DataFrame #class

from portfolioDescription.portfolio1MethDescription import Portfolio1MethDescription #class

from input.inputRecomMLDefinition import InputRecomMLDefinition #class
from input.inputRecomSTDefinition import InputRecomSTDefinition #class

from portfolioDescription.aPortfolioDescription import APortfolioDescription #class

from evaluationTool.evalToolDoNothing import EToolDoNothing #class

from recommenderDescription.recommenderDescription import RecommenderDescription #class

from input.inputSimulatorDefinition import InputSimulatorDefinition #class

from history.historyHierDF import HistoryHierDF #class

import pandas as pd
from input.inputABatchDefinition import InputABatchDefinition
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
    behaviourDF.columns = [BehavioursML.COL_USERID, BehavioursML.COL_MOVIEID, BehavioursML.COL_REPETITION, BehavioursML.COL_BEHAVIOUR]
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


argsSimulationDict:Dict[str,object] = {SimulationST.ARG_WINDOW_SIZE: 5,
                            SimulationST.ARG_RECOM_REPETITION_COUNT: 1,
                            SimulationST.ARG_NUMBER_OF_RECOMM_ITEMS: 100,
                            SimulationST.ARG_NUMBER_OF_AGGR_ITEMS: InputSimulatorDefinition.numberOfAggrItems,
                            SimulationST.ARG_DIV_DATASET_PERC_SIZE: 90,
                            SimulationST.ARG_HISTORY_LENGTH: 10}

def test01():

    print("Simulation: ML TheMostPopular")

    rDescr:RecommenderDescription = InputRecomMLDefinition.exportRDescTheMostPopular()

    pDescr:APortfolioDescription = Portfolio1MethDescription(InputRecomMLDefinition.THE_MOST_POPULAR.title(),
                                                             InputRecomMLDefinition.THE_MOST_POPULAR, rDescr)

    batchID:str = "ml1mDiv90Ulinear0109R1"
    dataset:DatasetML = DatasetML.readDatasets()
    behaviourFile:str = BehavioursML.getFile(BehavioursML.BHVR_LINEAR0109)
    behavioursDF:DataFrame = BehavioursML.readFromFileMl1m(behaviourFile)

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
    simulator.simulate([pDescr], [DataFrame()], [EToolDoNothing({})], [HistoryHierDF(pDescr.getPortfolioID())])


def test02():

    print("Simulation: ML W2V")

    rDescr:RecommenderDescription = InputRecomMLDefinition.exportRDescW2Vtpositivei50000ws1vs32upsweightedMeanups3()

    pDescr:APortfolioDescription = Portfolio1MethDescription("W2vPositiveMax",
                                    "w2vPositiveMax", rDescr)

    batchID:str = "ml1mDiv90Ulinear0109R1"
    dataset:DatasetML = DatasetML.readDatasets()
    behaviourFile:str = BehavioursML.getFile(BehavioursML.BHVR_LINEAR0109)
    behavioursDF:DataFrame = BehavioursML.readFromFileMl1m(behaviourFile)

    # simulation of portfolio
    simulator:Simulator = Simulator(batchID, SimulationML, argsSimulationDict, dataset, behavioursDF)
    simulator.simulate([pDescr], [DataFrame()], [EToolDoNothing({})], [HistoryHierDF(pDescr.getPortfolioID())])


def test03():

    print("Simulation: ML KNN")

    rDescr:RecommenderDescription = InputRecomMLDefinition.exportRDescKNN()

    pDescr:APortfolioDescription = Portfolio1MethDescription(InputRecomMLDefinition.KNN.title(),
                                                             InputRecomMLDefinition.KNN, rDescr)

    batchID:str = "ml1mDiv90Ulinear0109R1"
    dataset:DatasetML = DatasetML.readDatasets()
    behaviourFile:str = BehavioursML.getFile(BehavioursML.BHVR_LINEAR0109)
    behavioursDF:DataFrame = BehavioursML.readFromFileMl1m(behaviourFile)

    # simulation of portfolio
    simulator:Simulator = Simulator(batchID, SimulationML, argsSimulationDict, dataset, behavioursDF)
    simulator.simulate([pDescr], [DataFrame()], [EToolDoNothing({})], [HistoryHierDF(pDescr.getPortfolioID())])


def test04():

    print("Simulation: ML CB")

    #rDescr:RecommenderDescription = InputRecomMLDefinition.exportRDescCBmean()
    rDescr:RecommenderDescription = InputRecomMLDefinition.exportRDescCosineCBcbdOHEupsmaxups1()


    #pDescr:APortfolioDescription = Portfolio1MethDescription(InputRecomMLDefinition.COS_CB_MEAN.title(),
    #                                InputRecomMLDefinition.COS_CB_MEAN, rDescr)
    pDescr:APortfolioDescription = Portfolio1MethDescription(InputRecomMLDefinition.COS_CB_WINDOW3.title(),
                                                             InputRecomMLDefinition.COS_CB_WINDOW3, rDescr)


    batchID:str = "ml1mDiv90Ulinear0109R1"
    dataset:DatasetML = DatasetML.readDatasets()
    behaviourFile:str = BehavioursML.getFile(BehavioursML.BHVR_LINEAR0109)
    behavioursDF:DataFrame = BehavioursML.readFromFileMl1m(behaviourFile)

    # simulation of portfolio
    simulator:Simulator = Simulator(batchID, SimulationML, argsSimulationDict, dataset, behavioursDF)
    simulator.simulate([pDescr], [DataFrame()], [EToolDoNothing({})], [HistoryHierDF(pDescr.getPortfolioID())])


def test05():

    print("Simulation: ML MF")

    rDescr:RecommenderDescription = InputRecomMLDefinition.exportRDescBPRMFf100i10lr0003r01()

    pDescr:APortfolioDescription = Portfolio1MethDescription(InputRecomMLDefinition.BPRMFf100i10lr0003r01.title(),
                                                             InputRecomMLDefinition.BPRMFf100i10lr0003r01, rDescr)


    batchID:str = "ml1mDiv90Ulinear0109R1"
    dataset:DatasetML = DatasetML.readDatasets()
    behaviourFile:str = BehavioursML.getFile(BehavioursML.BHVR_LINEAR0109)
    behavioursDF:DataFrame = BehavioursML.readFromFileMl1m(behaviourFile)

    # simulation of portfolio
    simulator:Simulator = Simulator(batchID, SimulationML, argsSimulationDict, dataset, behavioursDF)
    simulator.simulate([pDescr], [DataFrame()], [EToolDoNothing({})], [HistoryHierDF(pDescr.getPortfolioID())])

def test06():

    print("Simulation: ML VMCMF")

    rDescr:RecommenderDescription = InputRecomSTDefinition.exportRDescVMContextKNN()

    pDescr:APortfolioDescription = Portfolio1MethDescription(InputRecomSTDefinition.VMC_KNN.title(),
                                                             InputRecomSTDefinition.VMC_KNN, rDescr)

    batchID:str = "mlDiv90Ulinear0109R1"
    dataset:DatasetML = DatasetML.readDatasets()
    behaviourFile:str = BehavioursML.getFile(BehavioursML.BHVR_LINEAR0109)
    behavioursDF:DataFrame = BehavioursML.readFromFileMl1m(behaviourFile)

    # simulation of portfolio
    simulator:Simulator = Simulator(batchID, SimulationML, argsSimulationDict, dataset, behavioursDF)
    simulator.simulate([pDescr], [DataFrame()], [EToolDoNothing({})], [HistoryHierDF(pDescr.getPortfolioID())])



def test11():

    print("Simulation: RR TheMostPopular")

    rDescr:RecommenderDescription = InputRecomMLDefinition.exportRDescTheMostPopular()

    pDescr:APortfolioDescription = Portfolio1MethDescription(InputRecomMLDefinition.THE_MOST_POPULAR.title(),
                                                             InputRecomMLDefinition.THE_MOST_POPULAR, rDescr)

    batchID:str = "retailrocketDiv90Ulinear0109R1"
    dataset:DatasetRetailRocket = DatasetRetailRocket.readDatasets()
    behaviourFile:str = BehavioursRR.getFile(BehavioursRR.BHVR_LINEAR0109)
    behavioursDF:DataFrame = BehavioursRR.readFromFileRR(behaviourFile)

    # simulation of portfolio
    simulator:Simulator = Simulator(batchID, SimulationRR, argsSimulationDict, dataset, behavioursDF)
    simulator.simulate([pDescr], [DataFrame()], [EToolDoNothing({})], [HistoryHierDF(pDescr.getPortfolioID())])




def test21():

    print("Simulation: ST TheMostPopular")

    rDescr:RecommenderDescription = InputRecomMLDefinition.exportRDescTheMostPopular()

    pDescr:APortfolioDescription = Portfolio1MethDescription(InputRecomMLDefinition.THE_MOST_POPULAR.title(),
                                                             InputRecomMLDefinition.THE_MOST_POPULAR, rDescr)

    batchID:str = "slantourDiv90Ulinear0109R1"
    dataset:DatasetST = DatasetST.readDatasets()
    behaviourFile:str = BehavioursST.getFile(BehavioursST.BHVR_LINEAR0109)
    behavioursDF:DataFrame = BehavioursST.readFromFileST(behaviourFile)

    # simulation of portfolio
    simulator:Simulator = Simulator(batchID, SimulationST, argsSimulationDict, dataset, behavioursDF)
    simulator.simulate([pDescr], [DataFrame()], [EToolDoNothing({})], [HistoryHierDF(pDescr.getPortfolioID())])


def test22():

    print("Simulation: ST W2V")

    rDescr:RecommenderDescription = InputRecomMLDefinition.exportRDescW2vPosnegMean()

    pDescr:APortfolioDescription = Portfolio1MethDescription(InputRecomMLDefinition.W2V_POSNEG_MEAN.title(),
                                                             InputRecomMLDefinition.W2V_POSNEG_MEAN, rDescr)

    batchID:str = "slantourDiv90Ulinear0109R1"
    dataset:DatasetST = DatasetST.readDatasets()
    behaviourFile:str = BehavioursST.getFile(BehavioursST.BHVR_LINEAR0109)
    behavioursDF:DataFrame = BehavioursST.readFromFileST(behaviourFile)

    # simulation of portfolio
    simulator:Simulator = Simulator(batchID, SimulationST, argsSimulationDict, dataset, behavioursDF)
    simulator.simulate([pDescr], [DataFrame()], [EToolDoNothing({})], [HistoryHierDF(pDescr.getPortfolioID())])


def test23():

    print("Simulation: ST KNN")

    rDescr:RecommenderDescription = InputRecomMLDefinition.exportRDescKNN()

    pDescr:APortfolioDescription = Portfolio1MethDescription(InputRecomMLDefinition.KNN.title(),
                                                             InputRecomMLDefinition.KNN, rDescr)

    batchID:str = "slantourDiv90Ulinear0109R1"
    dataset:DatasetST = DatasetST.readDatasets()
    behaviourFile:str = BehavioursST.getFile(BehavioursST.BHVR_LINEAR0109)
    behavioursDF:DataFrame = BehavioursST.readFromFileST(behaviourFile)

    # simulation of portfolio
    simulator:Simulator = Simulator(batchID, SimulationST, argsSimulationDict, dataset, behavioursDF)
    simulator.simulate([pDescr], [DataFrame()], [EToolDoNothing({})], [HistoryHierDF(pDescr.getPortfolioID())])


def test24():

    print("Simulation: ST CB")

    rDescr:RecommenderDescription = InputRecomMLDefinition.exportRDescCosineCBcbdOHEupsweightedMeanups3()

    pDescr:APortfolioDescription = Portfolio1MethDescription(InputRecomMLDefinition.COS_CB_MEAN.title(),
                                                             InputRecomMLDefinition.COS_CB_MEAN, rDescr)

    batchID:str = "slantourDiv90Ulinear0109R1"
    dataset:DatasetST = DatasetST.readDatasets()
    behaviourFile:str = BehavioursST.getFile(BehavioursST.BHVR_LINEAR0109)
    behavioursDF:DataFrame = BehavioursST.readFromFileST(behaviourFile)

    # simulation of portfolio
    simulator:Simulator = Simulator(batchID, SimulationST, argsSimulationDict, dataset, behavioursDF)
    simulator.simulate([pDescr], [DataFrame()], [EToolDoNothing({})], [HistoryHierDF(pDescr.getPortfolioID())])


def test25():

    print("Simulation: ST MF")

    rDescr:RecommenderDescription = InputRecomMLDefinition.exportRDescBPRMFf100i10lr0003r01()

    pDescr:APortfolioDescription = Portfolio1MethDescription(InputRecomMLDefinition.BPRMFf100i10lr0003r01.title(),
                                                             InputRecomMLDefinition.BPRMFf100i10lr0003r01, rDescr)

    batchID:str = "slantourDiv90Ulinear0109R1"
    dataset:DatasetST = DatasetST.readDatasets()
    behaviourFile:str = BehavioursST.getFile(BehavioursST.BHVR_LINEAR0109)
    behavioursDF:DataFrame = BehavioursST.readFromFileST(behaviourFile)

    # simulation of portfolio
    simulator:Simulator = Simulator(batchID, SimulationST, argsSimulationDict, dataset, behavioursDF)
    simulator.simulate([pDescr], [DataFrame()], [EToolDoNothing({})], [HistoryHierDF(pDescr.getPortfolioID())])


def test26():

    print("Simulation: ST VMCMF")

    rDescr:RecommenderDescription = InputRecomSTDefinition.exportRDescVMContextKNN()

    pDescr:APortfolioDescription = Portfolio1MethDescription(InputRecomSTDefinition.VMC_KNN.title(),
                                                             InputRecomSTDefinition.VMC_KNN, rDescr)

    batchID:str = "slantourDiv90Ulinear0109R1"
    dataset:DatasetST = DatasetST.readDatasets()
    behaviourFile:str = BehavioursST.getFile(BehavioursST.BHVR_LINEAR0109)
    behavioursDF:DataFrame = BehavioursST.readFromFileST(behaviourFile)

    # simulation of portfolio
    simulator:Simulator = Simulator(batchID, SimulationST, argsSimulationDict, dataset, behavioursDF)
    simulator.simulate([pDescr], [DataFrame()], [EToolDoNothing({})], [HistoryHierDF(pDescr.getPortfolioID())])



if __name__ == "__main__":
    os.chdir("..")

    #test00()

    # Simulation ML
    #test01()  # TheMostPopular
    #test02()  # W2V
    #test03()  # KNNN
    #test04()  # CB
    #test05()  # MF
    test06()

    # Simulation RR
    #test11()  # TheMostPopular

    # Simulation ST
    #test21()  # TheMostPopular
    #test22()  # W2V
    #test23()  # KNN
    #test24()  # CB
    #test25()  # MF
    #test26()  # VMContextKNN
