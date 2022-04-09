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

from portfolioDescription.portfolio1AggrDescription import Portfolio1AggrDescription #class

from evaluationTool.aEvalTool import AEvalTool #class
from evaluationTool.evalToolContext import EvalToolContext #class

from aggregationDescription.aggregationDescription import AggregationDescription #class

from batchDefinition.inputAggrDefinition import InputAggrDefinition  # class

from batchDefinition.inputRecomSTDefinition import InputRecomSTDefinition #class

from aggregation.operators.aDHondtSelector import ADHondtSelector #class
from aggregation.operators.rouletteWheelSelector import RouletteWheelSelector #class
from aggregation.operators.theMostVotedItemSelector import TheMostVotedItemSelector #class

from aggregation.negImplFeedback.aPenalization import APenalization #class

from batchDefinition.inputABatchDefinition import InputABatchDefinition
from batchDefinition.aBatchDefinitionST import ABatchDefinitionST #class
from batchDefinition.ml1m.batchDefMLFuzzyDHondtThompsonSamplingINF import BatchDefMLFuzzyDHondtThompsonSamplingINF #class

from batchDefinition.inputSimulatorDefinition import InputSimulatorDefinition #class

from simulator.simulator import Simulator #class

from history.historyHierDF import HistoryHierDF #class
from batchDefinition.slanTour.batchDefSTContextDHondtINF import BatchDefSTContextDHondtINF #class

from history.aHistory import AHistory #class

from portfolioModel.pModelBandit import PModelBandit #class
from portfolioModel.pModelDHondtBanditsVotes import PModelDHondtBanditsVotes #class
from portfolioModel.pModelDHondt import PModelDHondt #class


class BatchDefSTContextFuzzyDHondtDirectOptimizeINF(ABatchDefinitionST):

    def getBatchName(self):
        return "ContextFDHondtDirectOptimizeINF"

    def getParameters(self):
        return BatchDefSTContextDHondtINF().getParameters()


    def run(self, batchID:str, jobID:str):

        divisionDatasetPercentualSize:int
        uBehaviour:str
        repetition:int
        divisionDatasetPercentualSize, uBehaviour, repetition = InputABatchDefinition().getBatchParameters(self.datasetID)[batchID]

        selector:ADHondtSelector
        nImplFeedback:APenalization
        selector, nImplFeedback = self.getParameters()[jobID]

        dataset:ADataset = DatasetST.readDatasets()
        events = dataset.eventsDF
        serials = dataset.serialsDF

        historyDF:AHistory = HistoryHierDF("test01")

        # Init evalTool
        evalTool:AEvalTool = EvalToolContext({
            EvalToolContext.ARG_ITEMS: serials,  # ITEMS
            EvalToolContext.ARG_EVENTS: events,  # EVENTS (FOR CALCULATING HISTORY OF USER)
            EvalToolContext.ARG_DATASET: "st",  # WHAT DATASET ARE WE IN
            EvalToolContext.ARG_HISTORY: historyDF})  # empty instance of AHistory is OK for ST dataset

        rIDs, rDescs = InputRecomSTDefinition.exportPairOfRecomIdsAndRecomDescrs()

        aDescDHont:AggregationDescription = InputAggrDefinition.exportADescContextFuzzyDHondtDirectOptimizeINF(selector, nImplFeedback, evalTool)

        pDescr:Portfolio1AggrDescription = Portfolio1AggrDescription(
            self.getBatchName() + jobID, rIDs, rDescs, aDescDHont)

        model:DataFrame = PModelDHondt(pDescr.getRecommendersIDs())

        simulator:Simulator = InputSimulatorDefinition().exportSimulatorSlantour(
                batchID, divisionDatasetPercentualSize, uBehaviour, repetition)
        simulator.simulate([pDescr], [model], [evalTool], [HistoryHierDF(pDescr.getPortfolioID())])




if __name__ == "__main__":
    os.chdir("..")
    os.chdir("..")
    print(os.getcwd())

    BatchDefSTContextFuzzyDHondtDirectOptimizeINF.generateAllBatches(InputABatchDefinition())
    #BatchSTContextFuzzyDHondtDirectOptimizeINF().run("stDiv90Ulinear0109R1", "Fixed")
