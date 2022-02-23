#!/usr/bin/python3

import os
from typing import List #class
from typing import Dict #class

from datasets.aDataset import ADataset #class
from datasets.datasetML import DatasetML #class
from datasets.datasetRetailRocket import DatasetRetailRocket #class
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

from portfolioDescription.aPortfolioDescription import APortfolioDescription #class
from portfolioDescription.portfolio1AggrDescription import Portfolio1AggrDescription #class

#from evaluationTool.evalToolSingleMethod import EToolSingleMethod #class
from evaluationTool.evalToolBanditTS import EvalToolBanditTS #class
from evaluationTool.evalToolDHondt import EvalToolDHondt #class

from evaluationTool.aEvalTool import AEvalTool #class


from recommenderDescription.recommenderDescription import RecommenderDescription #class

from batchDefinition.inputSimulatorDefinition import InputSimulatorDefinition #class

from batchDefinition.inputRecomMLDefinition import InputRecomMLDefinition #class
from batchDefinition.inputRecomSTDefinition import InputRecomSTDefinition #class
from batchDefinition.inputRecomRRDefinition import InputRecomRRDefinition #class

from batchDefinition.inputAggrDefinition import InputAggrDefinition  # class

from batchDefinition.inputRecomMLDefinition import InputRecomMLDefinition #class
from batchDefinition.inputRecomSTDefinition import InputRecomSTDefinition #class

#from batchDefinition.batchesML1m.batchDefMLBanditTS import BatchDefMLBanditTS #class

from batchDefinition.inputABatchDefinition import InputABatchDefinition
from batchDefinition.aBatchDefinitionST import ABatchDefinitionST #class

from aggregation.aggrFuzzyDHondt import AggrFuzzyDHondt #class
from aggregation.operators.aDHondtSelector import ADHondtSelector #class
from batchDefinition.inputSimulatorDefinition import InputSimulatorDefinition #class

from simulator.simulator import Simulator #class

from history.historyHierDF import HistoryHierDF #class


from aggregation.aggrBanditTS import AggrBanditTS #class
from aggregation.operators.rouletteWheelSelector import RouletteWheelSelector #class
from aggregation.operators.theMostVotedItemSelector import TheMostVotedItemSelector #class

#import pandas as pd
#from history.historyDF import HistoryDF #class

from userBehaviourDescription.userBehaviourDescription import UserBehaviourDescription #class
from userBehaviourDescription.userBehaviourDescription import observationalStaticProbabilityFnc #function
from userBehaviourDescription.userBehaviourDescription import observationalLinearProbabilityFnc #function

from portfolioModel.pModelBandit import PModelBandit #class
from portfolioModel.pModelDHondtBanditsVotes import PModelDHondtBanditsVotes #class
from portfolioModel.pModelDHondt import PModelDHondt #class


argsSimulationDict:Dict[str,object] = {SimulationST.ARG_WINDOW_SIZE: 5,
                            SimulationST.ARG_RECOM_REPETITION_COUNT: 1,
                            SimulationST.ARG_NUMBER_OF_RECOMM_ITEMS: 100,
                            SimulationST.ARG_NUMBER_OF_AGGR_ITEMS: InputSimulatorDefinition.numberOfAggrItems,
                            SimulationST.ARG_DIV_DATASET_PERC_SIZE: 90,
                            SimulationST.ARG_HISTORY_LENGTH: 10}


def test01():

    print("Simulation: ML FuzzyDHondt")

    jobID:str = "Roulette1"

    selector = RouletteWheelSelector({RouletteWheelSelector.ARG_EXPONENT:1})

    rIDs, rDescs = InputRecomSTDefinition.exportPairOfRecomIdsAndRecomDescrs()

    pDescr:Portfolio1AggrDescription = Portfolio1AggrDescription(
        "FuzzyDHondt" + jobID, rIDs, rDescs, InputAggrDefinition.exportADescDHondt(selector))


    batchID:str = "ml1mDiv90Ulinear0109R1"
    dataset:DatasetML = DatasetML.readDatasets()
    behaviourFile:str = BehavioursML.getFile(BehavioursML.BHVR_LINEAR0109)
    behavioursDF:DataFrame = BehavioursML.readFromFileMl1m(behaviourFile)

    model:DataFrame = PModelDHondt(pDescr.getRecommendersIDs())

    lrClick:float = 0.03
    lrView:float = lrClick / 500
    eTool:AEvalTool = EvalToolDHondt({EvalToolDHondt.ARG_LEARNING_RATE_CLICKS: lrClick,
                                      EvalToolDHondt.ARG_LEARNING_RATE_VIEWS: lrView})

    # simulation of portfolio
    simulator:Simulator = Simulator(batchID, SimulationML, argsSimulationDict, dataset, behavioursDF)
    simulator.simulate([pDescr], [model], [eTool], [HistoryHierDF(pDescr.getPortfolioID())])


def test21():

    print("Simulation: ST FuzzyDHondt")

    jobID:str = "Roulette1"

    selector = RouletteWheelSelector({RouletteWheelSelector.ARG_EXPONENT:1})

    rIDs, rDescs = InputRecomSTDefinition.exportPairOfRecomIdsAndRecomDescrs()

    pDescr:Portfolio1AggrDescription = Portfolio1AggrDescription(
        "FuzzyDHondt" + jobID, rIDs, rDescs, InputAggrDefinition.exportADescDHondt(selector))


    batchID:str = "stDiv90Ulinear0109R1"
    dataset:DatasetST = DatasetST.readDatasets()
    behaviourFile:str = BehavioursST.getFile(BehavioursST.BHVR_LINEAR0109)
    behavioursDF:DataFrame = BehavioursST.readFromFileST(behaviourFile)

    model:DataFrame = PModelDHondt(pDescr.getRecommendersIDs())
    print(model)

    lrClick:float = 0.1
    lrView:float = lrClick / 300
    evalTool:AEvalTool = EvalToolDHondt({EvalToolDHondt.ARG_LEARNING_RATE_CLICKS: lrClick,
                                         EvalToolDHondt.ARG_LEARNING_RATE_VIEWS: lrView})

    # simulation of portfolio
    simulator:Simulator = Simulator(batchID, SimulationST, argsSimulationDict, dataset, behavioursDF)
    simulator.simulate([pDescr], [model], [evalTool], [HistoryHierDF(pDescr.getPortfolioID())])


def test31():

    print("Simulation: RR FuzzyDHondt")

    lrClick:float = 0.03
    #lrView:float = lrClick / 300
    lrViewDivisor:float = 250

    jobID:str = "Fixed" + "Clk" + str(lrClick).replace(".", "") + "ViewDivisor" + str(lrViewDivisor).replace(".", "")

    selector:ADHondtSelector = TheMostVotedItemSelector({})

    rIDs, rDescs = InputRecomRRDefinition.exportPairOfRecomIdsAndRecomDescrs()

    pDescr:Portfolio1AggrDescription = Portfolio1AggrDescription(
        "FDHondt" + jobID, rIDs, rDescs, InputAggrDefinition.exportADescDHondt(selector))


    batchID:str = "rrDiv90Ulinear0109R1"
    dataset:DatasetRetailRocket = DatasetRetailRocket.readDatasetsWithFilter(minEventCount=50)
    behaviourFile:str = BehavioursRR.getFile(BehavioursRR.BHVR_LINEAR0109)
    behavioursDF:DataFrame = BehavioursRR.readFromFileRR(behaviourFile)

    model:DataFrame = PModelDHondt(pDescr.getRecommendersIDs())
    print(model)

    evalTool:AEvalTool = EvalToolDHondt({EvalToolDHondt.ARG_LEARNING_RATE_CLICKS: lrClick,
                                         EvalToolDHondt.ARG_LEARNING_RATE_VIEWS: lrClick / lrViewDivisor})

    # simulation of portfolio
    simulator:Simulator = Simulator(batchID, SimulationRR, argsSimulationDict, dataset, behavioursDF)
    simulator.simulate([pDescr], [model], [evalTool], [HistoryHierDF(pDescr.getPortfolioID())])


if __name__ == "__main__":
    os.chdir("..")

    # Simulation ML
    #test01()  # BanditsTS

    # Simulation ST
    #test21()  # BanditsTS

    # Simulation RR
    test31()