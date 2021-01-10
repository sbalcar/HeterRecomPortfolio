#!/usr/bin/python3

import os

from typing import List

from pandas.core.frame import DataFrame #class

from portfolioDescription.portfolio1AggrDescription import Portfolio1AggrDescription #class

from evaluationTool.aEvalTool import AEvalTool #class
from evaluationTool.evalToolDHondt import EvalToolDHondt #class

from aggregationDescription.aggregationDescription import AggregationDescription #class

from input.inputAggrDefinition import InputAggrDefinition  #class
from input.modelDefinition import ModelDefinition

from input.inputRecomMLDefinition import InputRecomMLDefinition #class

from input.batchesML1m.batchMLFuzzyDHondtDirectOptimize import BatchMLFuzzyDHondtDirectOptimize #class
from input.batchesML1m.batchMLDHondtThompsonSamplingINF import BatchMLDHondtThompsonSamplingINF #class

from aggregation.negImplFeedback.aPenalization import APenalization #class

from aggregation.operators.aDHondtSelector import ADHondtSelector #class

from input.aBatch import BatchParameters #class
from input.aBatchML import ABatchML #class

from evaluationTool.evalToolDHondtBanditVotes import EvalToolDHondtBanditVotes #class

from input.inputSimulatorDefinition import InputSimulatorDefinition #class

from simulator.simulator import Simulator #class

from history.historyHierDF import HistoryHierDF #class



class BatchMLFuzzyDHondtDirectOptimizeINF(ABatchML):

    @staticmethod
    def getParameters():
        return BatchMLDHondtThompsonSamplingINF().getParameters()


    def run(self, batchID:str, jobID:str):

        divisionDatasetPercentualSize:int
        uBehaviour:str
        repetition:int
        divisionDatasetPercentualSize, uBehaviour, repetition = BatchParameters.getBatchParameters(self.datasetID)[batchID]

        selector, nImplFeedback = self.getParameters()[jobID]

        eTool:AEvalTool = EvalToolDHondt({EvalToolDHondt.ARG_LEARNING_RATE_CLICKS: 0.02,
                                           EvalToolDHondt.ARG_LEARNING_RATE_VIEWS: 1000})

        rIDs, rDescs = InputRecomMLDefinition.exportPairOfRecomIdsAndRecomDescrs()

        aDescFuzzyHontDirectOptimizeINF:AggregationDescription = InputAggrDefinition.exportADescDFuzzyHontDirectOptimizeINF(selector, nImplFeedback)

        pDescr:Portfolio1AggrDescription = Portfolio1AggrDescription(
            "FuzzyDHondtDirectOptimizeINF" + jobID, rIDs, rDescs, aDescFuzzyHontDirectOptimizeINF)

        model:DataFrame = ModelDefinition.createDHontModel(pDescr.getRecommendersIDs())

        simulator:Simulator = InputSimulatorDefinition.exportSimulatorML1M(
                batchID, divisionDatasetPercentualSize, uBehaviour, repetition)
        simulator.simulate([pDescr], [model], [eTool], HistoryHierDF)
