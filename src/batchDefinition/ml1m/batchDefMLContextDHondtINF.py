#!/usr/bin/python3

import os

from typing import List #class
from typing import Dict #class

from pandas.core.frame import DataFrame #class

from portfolioDescription.portfolio1AggrDescription import Portfolio1AggrDescription #class

from evaluationTool.aEvalTool import AEvalTool #class
from evaluationTool.evalToolDHondt import EvalToolDHondt #class
from evaluationTool.evalToolContext import EvalToolContext #class

from aggregationDescription.aggregationDescription import AggregationDescription #class

from batchDefinition.inputAggrDefinition import InputAggrDefinition  #class

from batchDefinition.inputRecomMLDefinition import InputRecomMLDefinition #class

from batchDefinition.ml1m.batchDefMLFuzzyDHondt import BatchDefMLFuzzyDHondt #class

from aggregation.negImplFeedback.aPenalization import APenalization #class

from aggregation.operators.aDHondtSelector import ADHondtSelector #class
from aggregation.negImplFeedback.penalUsingFiltering import PenalUsingFiltering #class
from aggregation.negImplFeedback.penalUsingProbability import PenalUsingProbability #class

from batchDefinition.ml1m.batchDefMLFuzzyDHondt import BatchDefMLFuzzyDHondt #class
from batchDefinition.ml1m.batchDefMLFuzzyDHondtINF import BatchDefMLFuzzyDHondtINF #class

from batchDefinition.inputABatchDefinition import InputABatchDefinition
from batchDefinition.ml1m.batchDefMLBanditTS import BatchDefMLBanditTS #class
from batchDefinition.aBatchDefinitionML import ABatchDefinitionML #class

from batchDefinition.inputSimulatorDefinition import InputSimulatorDefinition #class

from batchDefinition.inputAggrDefinition import PenalizationToolDefinition #class

from simulator.simulator import Simulator #class

from history.aHistory import AHistory #class
#from history.historyDF import HistoryDF #class
from history.historyHierDF import HistoryHierDF #class

from datasets.aDataset import ADataset #class
from datasets.datasetML import DatasetML #class
from datasets.ml.users import Users #class
from datasets.ml.items import Items #class
from datasets.ml.ratings import Ratings #class

from portfolioModel.pModelBandit import PModelBandit #class
from portfolioModel.pModelDHondtBanditsVotes import PModelDHondtBanditsVotes #class
from portfolioModel.pModelDHondt import PModelDHondt #class


class BatchDefMLContextDHondtINF(ABatchDefinitionML):

    def getBatchName(self):
        return "ContextDHondtINF"
    
    def getParameters(self):
        negativeImplFeedbackDict:Dict[str,object] = BatchDefMLFuzzyDHondtINF.getNegativeImplFeedbackParameters()
        selectorDict:Dict[str,object] = BatchDefMLFuzzyDHondt().getSelectorParameters()

        aDict:Dict[str,object] = {}
        for selectorIdI in selectorDict.keys():
            for nImplFeedbackIdJ in negativeImplFeedbackDict.keys():

                selectorI = selectorDict[selectorIdI]
                negativeImplFeedbackJ = negativeImplFeedbackDict[nImplFeedbackIdJ]

                keyIJ:str = str(selectorIdI) + nImplFeedbackIdJ
                aDict[keyIJ] = (selectorI, negativeImplFeedbackJ)
        return aDict


    def run(self, batchID:str, jobID:str):

        divisionDatasetPercentualSize:int
        uBehaviour:str
        repetition:int
        divisionDatasetPercentualSize, uBehaviour, repetition = InputABatchDefinition().getBatchParameters(self.datasetID)[batchID]

        selector:ADHondtSelector
        negativeImplFeedback:APenalization
        selector, negativeImplFeedback = self.getParameters()[jobID]

        portfolioID:str = self.getBatchName() + jobID

        history:AHistory = HistoryHierDF(portfolioID)

        dataset:ADataset = DatasetML.readDatasets()
        usersDF = dataset.usersDF
        itemsDF = dataset.itemsDF

        # Init evalTool
        evalTool:AEvalTool = EvalToolContext({
            EvalToolContext.ARG_USERS: usersDF,
            EvalToolContext.ARG_ITEMS: itemsDF,
            EvalToolContext.ARG_DATASET: "ml",
            EvalToolContext.ARG_HISTORY: history})

        rIDs, rDescs = InputRecomMLDefinition.exportPairOfRecomIdsAndRecomDescrs()

        aDescContextDHont:AggregationDescription = InputAggrDefinition.exportADescDContextHondtINF(selector, negativeImplFeedback, evalTool)

        pDescr:Portfolio1AggrDescription = Portfolio1AggrDescription(
            portfolioID, rIDs, rDescs, aDescContextDHont)

        model:DataFrame = PModelDHondt(pDescr.getRecommendersIDs())

        simulator:Simulator = InputSimulatorDefinition().exportSimulatorML1M(
                batchID, divisionDatasetPercentualSize, uBehaviour, repetition)
        simulator.simulate([pDescr], [model], [evalTool], [history])




if __name__ == "__main__":
   os.chdir("..")
   os.chdir("..")
   print(os.getcwd())

   BatchDefMLContextDHondtINF.generateAllBatches(InputABatchDefinition())