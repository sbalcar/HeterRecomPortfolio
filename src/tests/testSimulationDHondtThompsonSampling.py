#!/usr/bin/python3

import os
from typing import List #class
from typing import Dict #class

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

from portfolioDescription.aPortfolioDescription import APortfolioDescription #class
from portfolioDescription.portfolio1AggrDescription import Portfolio1AggrDescription #class

#from evaluationTool.evalToolSingleMethod import EToolSingleMethod #class
from evaluationTool.evalToolBanditTS import EvalToolBanditTS #class
from evaluationTool.evalToolDHondt import EvalToolDHondt #class
from evaluationTool.evalToolDHondtBanditVotes import EvalToolDHondtBanditVotes #class

from evaluationTool.aEvalTool import AEvalTool #class


from recommenderDescription.recommenderDescription import RecommenderDescription #class

from input.inputSimulatorDefinition import InputSimulatorDefinition #class

from input.inputRecomMLDefinition import InputRecomMLDefinition #class
from input.inputRecomSTDefinition import InputRecomSTDefinition #class

from input.inputAggrDefinition import InputAggrDefinition  # class
from input.modelDefinition import ModelDefinition

from input.inputRecomMLDefinition import InputRecomMLDefinition #class
from input.inputRecomSTDefinition import InputRecomSTDefinition #class

from input.batchesML1m.batchMLBanditTS import BatchMLBanditTS #class

from input.aBatch import BatchParameters #class
from input.aBatchST import ABatchST #class

from aggregation.aggrFuzzyDHondt import AggrFuzzyDHondt #class
from aggregation.operators.aDHondtSelector import ADHondtSelector #class
from input.inputSimulatorDefinition import InputSimulatorDefinition #class

from simulator.simulator import Simulator #class

from history.historyHierDF import HistoryHierDF #class


from aggregation.aggrBanditTS import AggrBanditTS #class
from aggregation.operators.rouletteWheelSelector import RouletteWheelSelector #class

#import pandas as pd
#from history.historyDF import HistoryDF #class

from userBehaviourDescription.userBehaviourDescription import UserBehaviourDescription #class
from userBehaviourDescription.userBehaviourDescription import observationalStaticProbabilityFnc #function
from userBehaviourDescription.userBehaviourDescription import observationalLinearProbabilityFnc #function



argsSimulationDict:Dict[str,object] = {SimulationST.ARG_WINDOW_SIZE: 5,
                            SimulationST.ARG_RECOM_REPETITION_COUNT: 1,
                            SimulationST.ARG_NUMBER_OF_RECOMM_ITEMS: 100,
                            SimulationST.ARG_NUMBER_OF_AGGR_ITEMS: InputSimulatorDefinition.numberOfAggrItems,
                            SimulationST.ARG_DIV_DATASET_PERC_SIZE: 90,
                            SimulationST.ARG_HISTORY_LENGTH: 10}


def test01():

    print("Simulation: ML DHontThompsonSampling")

    jobID:str = ""

    selector = RouletteWheelSelector({RouletteWheelSelector.ARG_EXPONENT:1})

    rIDs, rDescs = InputRecomSTDefinition.exportPairOfRecomIdsAndRecomDescrs()

    pDescr:Portfolio1AggrDescription = Portfolio1AggrDescription(
        "DHontThompsonSampling" + jobID, rIDs, rDescs, InputAggrDefinition.exportADescDHontThompsonSampling(selector))


    batchID:str = "ml1mDiv90Ulinear0109R1"
    dataset:DatasetML = DatasetML.readDatasets()
    behaviourFile:str = BehavioursML.getFile(BehavioursML.BHVR_LINEAR0109)
    behavioursDF:DataFrame = BehavioursML.readFromFileMl1m(behaviourFile)

    model:DataFrame = ModelDefinition.createDHondtBanditsVotesModel(pDescr.getRecommendersIDs())

    eTool:AEvalTool = EvalToolDHondtBanditVotes({})

    # simulation of portfolio
    simulator:Simulator = Simulator(batchID, SimulationML, argsSimulationDict, dataset, behavioursDF)
    simulator.simulate([pDescr], [model], [eTool], HistoryHierDF)


def test21():

    print("Simulation: ST DHontThompsonSampling")

    jobID:str = ""

    selector = RouletteWheelSelector({RouletteWheelSelector.ARG_EXPONENT:1})

    rIDs, rDescs = InputRecomSTDefinition.exportPairOfRecomIdsAndRecomDescrs()

    pDescr:Portfolio1AggrDescription = Portfolio1AggrDescription(
        "DHontThompsonSampling" + jobID, rIDs, rDescs, InputAggrDefinition.exportADescDHontThompsonSampling(selector))


    batchID:str = "stDiv90Ulinear0109R1"
    dataset:DatasetST = DatasetST.readDatasets()
    behaviourFile:str = BehavioursST.getFile(BehavioursST.BHVR_LINEAR0109)
    behavioursDF:DataFrame = BehavioursST.readFromFileST(behaviourFile)

    model:DataFrame = ModelDefinition.createDHondtBanditsVotesModel(pDescr.getRecommendersIDs())
    print(model)

    evalTool:AEvalTool = EvalToolDHondtBanditVotes({})

    # simulation of portfolio
    simulator:Simulator = Simulator(batchID, SimulationST, argsSimulationDict, dataset, behavioursDF)
    simulator.simulate([pDescr], [model], [evalTool], HistoryHierDF)


if __name__ == "__main__":
    os.chdir("..")

    # Simulation ML
#    test01()  # BanditsTS

    # Simulation ST
    test21()  # BanditsTS