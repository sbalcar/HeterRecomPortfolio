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

from input.inputRecomSTDefinition import InputRecomSTDefinition #class

from input.batchesML1m.batchMLFuzzyDHondt import BatchMLFuzzyDHondt #class
from input.batchesML1m.batchMLFuzzyDHondtINF import BatchMLFuzzyDHondtINF #class

from aggregation.negImplFeedback.aPenalization import APenalization #class

from aggregation.operators.aDHondtSelector import ADHondtSelector #class

from input.aBatch import BatchParameters #class
from input.aBatchST import ABatchST #class

from evaluationTool.evalToolDHondtBanditVotes import EvalToolDHondtBanditVotes #class

from input.inputSimulatorDefinition import InputSimulatorDefinition #class

from simulator.simulator import Simulator #class

from history.historyHierDF import HistoryHierDF #class



class BatchSTDHondtThompsonSamplingINF(ABatchST):

    @staticmethod
    def getParameters():
        selectorIDs:List[str] = BatchMLFuzzyDHondt().getSelectorParameters().keys()
        negativeImplFeedback:List[str] = BatchMLFuzzyDHondtINF().getNegativeImplFeedbackParameters().keys()

        aDict:dict = {}
        for selectorIDH in selectorIDs:
            for nImplFeedbackI in negativeImplFeedback:
                keyIJ:str = str(selectorIDH) + nImplFeedbackI

                nImplFeedback:APenalization = BatchMLFuzzyDHondtINF().getNegativeImplFeedbackParameters()[nImplFeedbackI]
                selectorH:ADHondtSelector = BatchMLFuzzyDHondt().getSelectorParameters()[selectorIDH]

                aDict[keyIJ] = (selectorH, nImplFeedback)
        return aDict


    def run(self, batchID:str, jobID:str):

        divisionDatasetPercentualSize:int
        uBehaviour:str
        repetition:int
        divisionDatasetPercentualSize, uBehaviour, repetition = BatchParameters.getBatchParameters(self.datasetID)[batchID]

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