#!/usr/bin/python3

import os

from typing import List #class
from typing import Dict #class

from pandas.core.frame import DataFrame #class

from portfolioDescription.portfolio1AggrDescription import Portfolio1AggrDescription #class

from evaluationTool.aEvalTool import AEvalTool #class
from evaluationTool.evalToolDHondt import EvalToolDHondt #class

from aggregationDescription.aggregationDescription import AggregationDescription #class

from batchDefinition.inputAggrDefinition import InputAggrDefinition  #class
from batchDefinition.modelDefinition import ModelDefinition

from batchDefinition.inputRecomSTDefinition import InputRecomSTDefinition #class

from batchDefinition.ml1m.batchDefMLFuzzyDHondt import BatchDefMLFuzzyDHondt #class
from batchDefinition.ml1m.batchDefMLFuzzyDHondtINF import BatchMLFuzzyDHondtINF #class
from batchDefinition.ml1m.batchDefMLFuzzyDHondtThompsonSamplingINF import BatchDefMLFuzzyDHondtThompsonSamplingINF #class
from aggregation.negImplFeedback.aPenalization import APenalization #class

from aggregation.operators.aDHondtSelector import ADHondtSelector #class

from batchDefinition.inputABatchDefinition import InputABatchDefinition
from batchDefinition.aBatchDefinitionST import ABatchDefinitionST #class

from evaluationTool.evalToolDHondtBanditVotes import EvalToolDHondtBanditVotes #class

from batchDefinition.inputSimulatorDefinition import InputSimulatorDefinition #class

from simulator.simulator import Simulator #class

from history.historyHierDF import HistoryHierDF #class

from aggregation.negImplFeedback.aPenalization import APenalization #class
from aggregation.negImplFeedback.penalUsingFiltering import PenalUsingFiltering #class
from aggregation.negImplFeedback.penalUsingReduceRelevance import PenalUsingReduceRelevance #class
from aggregation.negImplFeedback.penalUsingProbability import PenalUsingProbability #class
from aggregation.negImplFeedback.penalUsingReduceRelevance import penaltyStatic #function
from aggregation.negImplFeedback.penalUsingReduceRelevance import penaltyLinear #function



class BatchDefSTDFuzzyHondtThompsonSamplingINF(ABatchDefinitionST):

    @staticmethod
    def getParameters():
        return BatchDefMLFuzzyDHondtThompsonSamplingINF.getParameters()


    def run(self, batchID:str, jobID:str):

        divisionDatasetPercentualSize:int
        uBehaviour:str
        repetition:int
        divisionDatasetPercentualSize, uBehaviour, repetition = InputABatchDefinition.getBatchParameters(self.datasetID)[batchID]

        selector, nImplFeedback = self.getParameters()[jobID]

        eTool:AEvalTool = EvalToolDHondtBanditVotes({})

        rIDs, rDescs = InputRecomSTDefinition.exportPairOfRecomIdsAndRecomDescrs()

        aDescNegDHontThompsonSamplingI:AggregationDescription = InputAggrDefinition.exportADescDHondtThompsonSamplingINF(selector, nImplFeedback)

        pDescr:Portfolio1AggrDescription = Portfolio1AggrDescription(
            "DHondtThompsonSamplingINF" + jobID, rIDs, rDescs, aDescNegDHontThompsonSamplingI)

        model:DataFrame = ModelDefinition.createDHondtBanditsVotesModel(pDescr.getRecommendersIDs())

        simulator:Simulator = InputSimulatorDefinition.exportSimulatorSlantour(
                batchID, divisionDatasetPercentualSize, uBehaviour, repetition)
        simulator.simulate([pDescr], [model], [eTool], [HistoryHierDF(pDescr.getPortfolioID())])



if __name__ == "__main__":
    os.chdir("..")
    os.chdir("..")

    BatchDefSTDFuzzyHondtThompsonSamplingINF.generateAllBatches()
