#!/usr/bin/python3

import os
from typing import List #class
from typing import Dict #class

from datasets.aDataset import ADataset #class
from datasets.datasetML import DatasetML #class
from datasets.datasetRetailRocket import DatasetRetailRocket #class
from datasets.datasetST import DatasetST #class

from datasets.ml.ratings import Ratings #class
from datasets.ml.items import Items #class
from datasets.ml.users import Users #class
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
from evaluationTool.evalToolContext import EvalToolContext #class

from recommenderDescription.recommenderDescription import RecommenderDescription #class

from batchDefinition.inputSimulatorDefinition import InputSimulatorDefinition #class

from batchDefinition.inputRecomMLDefinition import InputRecomMLDefinition #class
from batchDefinition.inputRecomSTDefinition import InputRecomSTDefinition #class

from batchDefinition.inputAggrDefinition import InputAggrDefinition  # class
from batchDefinition.modelDefinition import ModelDefinition

from batchDefinition.inputRecomMLDefinition import InputRecomMLDefinition #class
from batchDefinition.inputRecomSTDefinition import InputRecomSTDefinition #class

#from batchDefinition.batchesML1m.batchDefMLBanditTS import BatchDefMLBanditTS #class

from batchDefinition.inputABatchDefinition import InputABatchDefinition #class
from batchDefinition.aBatchDefinitionST import ABatchDefinitionST #class

from aggregation.aggrFuzzyDHondt import AggrFuzzyDHondt #class
from aggregation.operators.aDHondtSelector import ADHondtSelector #class
from batchDefinition.inputSimulatorDefinition import InputSimulatorDefinition #class

from simulator.simulator import Simulator #class

from history.aHistory import AHistory #class
from history.historyHierDF import HistoryHierDF #class

from aggregation.aggrBanditTS import AggrBanditTS #class
from aggregation.operators.rouletteWheelSelector import RouletteWheelSelector #class

#import pandas as pd
#from history.historyDF import HistoryDF #class

from userBehaviourDescription.userBehaviourDescription import UserBehaviourDescription #class
from userBehaviourDescription.userBehaviourDescription import observationalStaticProbabilityFnc #function
from userBehaviourDescription.userBehaviourDescription import observationalLinearProbabilityFnc #function

from aggregation.negImplFeedback.aPenalization import APenalization #class
from batchDefinition.inputAggrDefinition import PenalizationToolDefinition #class

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

    print("Simulation: ML ContextFuzzyDHondtDirectOptimize")

    jobID:str = "Roulette1"

    selector:ADHondtSelector = RouletteWheelSelector({RouletteWheelSelector.ARG_EXPONENT:1})


    itemsDF:DataFrame = Items.readFromFileMl1m()
    usersDF:DataFrame = Users.readFromFileMl1m()

    historyDF:AHistory = HistoryHierDF("test01")

    eTool:AEvalTool = EvalToolContext({
            EvalToolContext.ARG_USERS: usersDF,
            EvalToolContext.ARG_ITEMS: itemsDF,
            EvalToolContext.ARG_DATASET: "ml",
            EvalToolContext.ARG_HISTORY: historyDF})


    rIDs, rDescs = InputRecomSTDefinition.exportPairOfRecomIdsAndRecomDescrs()

    pDescr:Portfolio1AggrDescription = Portfolio1AggrDescription(
        "ContextFuzzyDHondtDirectOptimize" + jobID, rIDs, rDescs, InputAggrDefinition.exportADescContextFuzzyDHondtDirectOptimize(selector, eTool))


    batchID:str = "ml1mDiv90Ulinear0109R1"
    dataset:DatasetML = DatasetML.readDatasets()
    behaviourFile:str = BehavioursML.getFile(BehavioursML.BHVR_LINEAR0109)
    behavioursDF:DataFrame = BehavioursML.readFromFileMl1m(behaviourFile)

    model:DataFrame = PModelDHondt(pDescr.getRecommendersIDs())

    # simulation of portfolio
    simulator:Simulator = Simulator(batchID, SimulationML, argsSimulationDict, dataset, behavioursDF)
    simulator.simulate([pDescr], [model], [eTool], [historyDF])


if __name__ == "__main__":
    os.chdir("..")

    # Simulation ML
    test01()  # ContextFuzzyDHondtDirectOptimize

    # Simulation ST
#    test21()  # ContextFuzzyDHondtDirectOptimize