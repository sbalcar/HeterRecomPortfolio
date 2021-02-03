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

from input.inputRecomMLDefinition import InputRecomMLDefinition #class

from input.batchesML1m.batchMLFuzzyDHondt import BatchMLFuzzyDHondt #class

from aggregation.negImplFeedback.aPenalization import APenalization #class

from aggregation.operators.aDHondtSelector import ADHondtSelector #class
from aggregation.negImplFeedback.penalUsingFiltering import PenalUsingFiltering #class
from aggregation.negImplFeedback.penalUsingProbability import PenalUsingProbability #class

from input.batchesML1m.batchMLFuzzyDHondt import BatchMLFuzzyDHondt #class

from input.inputABatchDefinition import InputABatchDefinition
from input.batchesML1m.batchMLBanditTS import BatchMLBanditTS #class
from input.aBatchML import ABatchML #class

from input.inputSimulatorDefinition import InputSimulatorDefinition #class

from input.inputAggrDefinition import PenalizationToolDefinition #class

from simulator.simulator import Simulator #class

from history.aHistory import AHistory #class
from history.historyDF import HistoryDF #class
from history.historyHierDF import HistoryHierDF #class

from datasets.ml.users import Users #class
from datasets.ml.items import Items #class
from datasets.ml.ratings import Ratings #class


class BatchMLContextDHondt(ABatchML):

    @staticmethod
    def getParameters():
        return BatchMLBanditTS.getParameters()


    def run(self, batchID:str, jobID:str):

        divisionDatasetPercentualSize:int
        uBehaviour:str
        repetition:int
        divisionDatasetPercentualSize, uBehaviour, repetition = InputABatchDefinition.getBatchParameters(self.datasetID)[batchID]

        selector:ADHondtSelector = self.getParameters()[jobID]

        portfolioID:str = "ContextDHondt" + jobID

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
   BatchMLContextDHondt.generateAllBatches()