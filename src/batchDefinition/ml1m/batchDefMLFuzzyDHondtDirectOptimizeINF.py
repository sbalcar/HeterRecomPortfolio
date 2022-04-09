#!/usr/bin/python3

import os

from typing import List

from pandas.core.frame import DataFrame #class

from portfolioDescription.portfolio1AggrDescription import Portfolio1AggrDescription #class

from evaluationTool.aEvalTool import AEvalTool #class
from evaluationTool.evalToolDHondt import EvalToolDHondt #class

from aggregationDescription.aggregationDescription import AggregationDescription #class

from batchDefinition.inputAggrDefinition import InputAggrDefinition  #class

from batchDefinition.inputRecomMLDefinition import InputRecomMLDefinition #class

from batchDefinition.ml1m.batchDefMLFuzzyDHondtDirectOptimize import BatchDefMLFuzzyDHondtDirectOptimize #class
from batchDefinition.ml1m.batchDefMLFuzzyDHondtThompsonSamplingINF import BatchDefMLFuzzyDHondtThompsonSamplingINF #class

from aggregation.negImplFeedback.aPenalization import APenalization #class

from aggregation.operators.aDHondtSelector import ADHondtSelector #class

from batchDefinition.inputABatchDefinition import InputABatchDefinition
from batchDefinition.aBatchDefinitionML import ABatchDefinitionML #class

from evaluationTool.evalToolDHondtBanditVotes import EvalToolDHondtBanditVotes #class

from batchDefinition.inputSimulatorDefinition import InputSimulatorDefinition #class

from simulator.simulator import Simulator #class

from history.historyHierDF import HistoryHierDF #class

from portfolioModel.pModelBandit import PModelBandit #class
from portfolioModel.pModelDHondtBanditsVotes import PModelDHondtBanditsVotes #class
from portfolioModel.pModelDHondt import PModelDHondt #class



class BatchDefMLFuzzyDHondtDirectOptimizeINF(ABatchDefinitionML):

    def getBatchName(self):
        return "FuzzyDHondtDirectOptimizeINF"

    def getParameters(self):
        return BatchDefMLFuzzyDHondtThompsonSamplingINF().getParameters()

    def run(self, batchID:str, jobID:str):

        divisionDatasetPercentualSize:int
        uBehaviour:str
        repetition:int
        divisionDatasetPercentualSize, uBehaviour, repetition = InputABatchDefinition().getBatchParameters(self.datasetID)[batchID]

        selector, nImplFeedback = self.getParameters()[jobID]

        eTool:AEvalTool = EvalToolDHondt({EvalToolDHondt.ARG_LEARNING_RATE_CLICKS: 0.02,
                                           EvalToolDHondt.ARG_LEARNING_RATE_VIEWS: 1000})

        rIDs, rDescs = InputRecomMLDefinition.exportPairOfRecomIdsAndRecomDescrs()

        aDescFuzzyHontDirectOptimizeINF:AggregationDescription = InputAggrDefinition.exportADescDFuzzyHondtDirectOptimizeINF(selector, nImplFeedback)

        pDescr:Portfolio1AggrDescription = Portfolio1AggrDescription(
            self.getBatchName() + jobID, rIDs, rDescs, aDescFuzzyHontDirectOptimizeINF)

        model:DataFrame = PModelDHondt(pDescr.getRecommendersIDs())

        simulator:Simulator = InputSimulatorDefinition().exportSimulatorML1M(
                batchID, divisionDatasetPercentualSize, uBehaviour, repetition)
        simulator.simulate([pDescr], [model], [eTool], [HistoryHierDF(pDescr.getPortfolioID())])
