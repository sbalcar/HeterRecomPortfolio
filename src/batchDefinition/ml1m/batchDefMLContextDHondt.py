#!/usr/bin/python3

import os

from typing import List

from pandas.core.frame import DataFrame #class

from portfolioDescription.portfolio1AggrDescription import Portfolio1AggrDescription #class

from evaluationTool.aEvalTool import AEvalTool #class
from evaluationTool.evalToolDHondt import EvalToolDHondt #class
from evaluationTool.evalToolContext import EvalToolContext #class

from aggregationDescription.aggregationDescription import AggregationDescription #class

from batchDefinition.inputAggrDefinition import InputAggrDefinition  #class
from batchDefinition.modelDefinition import ModelDefinition

from batchDefinition.inputRecomMLDefinition import InputRecomMLDefinition #class

from batchDefinition.ml1m.batchDefMLFuzzyDHondt import BatchDefMLFuzzyDHondt #class

from aggregation.negImplFeedback.aPenalization import APenalization #class

from aggregation.operators.aDHondtSelector import ADHondtSelector #class
from aggregation.negImplFeedback.penalUsingFiltering import PenalUsingFiltering #class
from aggregation.negImplFeedback.penalUsingProbability import PenalUsingProbability #class

from batchDefinition.ml1m.batchDefMLFuzzyDHondt import BatchDefMLFuzzyDHondt #class

from batchDefinition.inputABatchDefinition import InputABatchDefinition
from batchDefinition.ml1m.batchDefMLBanditTS import BatchDefMLBanditTS #class
from batchDefinition.aBatchDefinitionML import ABatchDefinitionML #class

from batchDefinition.inputSimulatorDefinition import InputSimulatorDefinition #class

from batchDefinition.inputAggrDefinition import PenalizationToolDefinition #class

from simulator.simulator import Simulator #class

from history.aHistory import AHistory #class
from history.historyDF import HistoryDF #class
from history.historyHierDF import HistoryHierDF #class

from datasets.ml.users import Users #class
from datasets.ml.items import Items #class
from datasets.ml.ratings import Ratings #class

from batchDefinition.ml1m.batchDefMLFuzzyDHondtDirectOptimize import BatchDefMLFuzzyDHondtDirectOptimize #class


class BatchDefMLContextDHondt(ABatchDefinitionML):

    selectorIDs = [BatchDefMLFuzzyDHondtDirectOptimize.SLCTR_FIXED]

    def getBatchName(self):
        return "ContextDHondt"

    def getParameters(self):
        batchDefMLBanditTS = BatchDefMLBanditTS()
        batchDefMLBanditTS.selectorIDs = self.selectorIDs
        return batchDefMLBanditTS.getParameters()

    def run(self, batchID:str, jobID:str):

        divisionDatasetPercentualSize:int
        uBehaviour:str
        repetition:int
        divisionDatasetPercentualSize, uBehaviour, repetition = InputABatchDefinition.getBatchParameters(self.datasetID)[batchID]

        selector:ADHondtSelector = self.getParameters()[jobID]

        portfolioID:str = self.getBatchName() + jobID

        history:AHistory = HistoryHierDF(portfolioID)

        itemsDF:DataFrame = Items.readFromFileMl1m()
        usersDF:DataFrame = Users.readFromFileMl1m()

        eTool:AEvalTool = EvalToolContext({
            EvalToolContext.ARG_USERS: usersDF,
            EvalToolContext.ARG_ITEMS: itemsDF,
            EvalToolContext.ARG_DATASET: "ml",
            EvalToolContext.ARG_HISTORY: history})

        rIDs, rDescs = InputRecomMLDefinition.exportPairOfRecomIdsAndRecomDescrs()

        aDescContextDHont:AggregationDescription = InputAggrDefinition.exportADescDContextHondt(selector, eTool)

        pDescr:Portfolio1AggrDescription = Portfolio1AggrDescription(
            portfolioID, rIDs, rDescs, aDescContextDHont)

        model:DataFrame = ModelDefinition.createDHontModel(pDescr.getRecommendersIDs())

        simulator:Simulator = InputSimulatorDefinition.exportSimulatorML1M(
                batchID, divisionDatasetPercentualSize, uBehaviour, repetition)
        simulator.simulate([pDescr], [model], [eTool], [history])




if __name__ == "__main__":
   os.chdir("..")
   os.chdir("..")
   print(os.getcwd())

   BatchDefMLContextDHondt.generateAllBatches()