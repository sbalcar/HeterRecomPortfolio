#!/usr/bin/python3

import os

from typing import List

from pandas.core.frame import DataFrame #class

from portfolioDescription.portfolio1AggrDescription import Portfolio1AggrDescription #class

from evaluationTool.aEvalTool import AEvalTool #class
from evaluationTool.evalToolDHondt import EvalToolDHondt #class
from evaluationTool.evalToolContext import EvalToolContext #class

from aggregationDescription.aggregationDescription import AggregationDescription #class

from input.inputAggrDefinition import InputAggrDefinition  #class
from input.modelDefinition import ModelDefinition

from input.inputRecomSTDefinition import InputRecomSTDefinition #class

from input.batchesML1m.batchMLFuzzyDHondt import BatchMLFuzzyDHondt #class

from aggregation.negImplFeedback.aPenalization import APenalization #class

from aggregation.operators.aDHondtSelector import ADHondtSelector #class
from aggregation.negImplFeedback.penalUsingFiltering import PenalUsingFiltering #class
from aggregation.negImplFeedback.penalUsingProbability import PenalUsingProbability #class

from input.batchesML1m.batchMLFuzzyDHondt import BatchMLFuzzyDHondt #class

from input.aBatch import BatchParameters #class
from input.batchesML1m.batchMLBanditTS import BatchMLBanditTS #class
from input.aBatchST import ABatchST #class

from input.inputSimulatorDefinition import InputSimulatorDefinition #class

from input.inputAggrDefinition import PenalizationToolDefinition #class

from simulator.simulator import Simulator #class

from history.aHistory import AHistory #class
from history.historyDF import HistoryDF #class
from history.historyHierDF import HistoryHierDF #class

from datasets.aDataset import ADataset #class
from datasets.datasetST import DatasetST #class
from datasets.ml.users import Users #class
from datasets.ml.items import Items #class
from datasets.ml.ratings import Ratings #class


class BatchSTContextDHondt(ABatchST):

    @staticmethod
    def getParameters():
        return BatchMLBanditTS.getParameters()


    def run(self, batchID:str, jobID:str):

        divisionDatasetPercentualSize:int
        uBehaviour:str
        repetition:int
        divisionDatasetPercentualSize, uBehaviour, repetition = BatchParameters.getBatchParameters(self.datasetID)[batchID]

        selector:ADHondtSelector = self.getParameters()[jobID]

        portfolioID:str = "ContextDHondt" + jobID

        history:AHistory = HistoryHierDF(portfolioID)

        dataset:ADataset = DatasetST.readDatasets()
        events = dataset.eventsDF
        serials = dataset.serialsDF

        historyDF: AHistory = HistoryDF("test01")

        # Init evalTool
        evalTool:AEvalTool = EvalToolContext({
            EvalToolContext.ARG_ITEMS: serials,  # ITEMS
            EvalToolContext.ARG_EVENTS: events,  # EVENTS (FOR CALCULATING HISTORY OF USER)
            EvalToolContext.ARG_DATASET: "st",  # WHAT DATASET ARE WE IN
            EvalToolContext.ARG_HISTORY: historyDF})  # empty instance of AHistory is OK for ST dataset

        rIDs, rDescs = InputRecomSTDefinition.exportPairOfRecomIdsAndRecomDescrs()

        aDescContextDHont:AggregationDescription = InputAggrDefinition.exportADescDContextHondt(selector, evalTool)

        pDescr:Portfolio1AggrDescription = Portfolio1AggrDescription(
            portfolioID, rIDs, rDescs, aDescContextDHont)

        model:DataFrame = ModelDefinition.createDHontModel(pDescr.getRecommendersIDs())

        simulator:Simulator = InputSimulatorDefinition.exportSimulatorSlantour(
                batchID, divisionDatasetPercentualSize, uBehaviour, repetition)
        simulator.simulate([pDescr], [model], [evalTool], [history])




if __name__ == "__main__":
   os.chdir("..")
   os.chdir("..")
   print(os.getcwd())
   BatchSTContextDHondt.generateBatches()