#!/usr/bin/python3

import os

from typing import List

from pandas.core.frame import DataFrame #class

from portfolioDescription.portfolio1AggrDescription import Portfolio1AggrDescription #class

from evaluationTool.aEvalTool import AEvalTool #class
from evaluationTool.evalToolDHondt import EvalToolDHondt #class

from aggregationDescription.aggregationDescription import AggregationDescription #class

from input.inputAggrDefinition import InputAggrDefinition, ModelDefinition  #class

from input.inputRecomDefinition import InputRecomDefinition #class

from input.batchesML1m.batchFuzzyDHondt import BatchFuzzyDHondt #class
from input.batchesML1m.batchFuzzyDHondtINF import BatchFuzzyDHondtINF #class

from aggregation.negImplFeedback.aPenalization import APenalization #class

from aggregation.operators.aDHondtSelector import ADHondtSelector #class

from input.aBatch import BatchParameters #class
from input.aBatchML import ABatchML #class

from evaluationTool.evalToolDHondtBanditVotes import EvalToolDHondtBanditVotes #class

from input.inputSimulatorDefinition import InputSimulatorDefinition #class

from simulator.simulator import Simulator #class

from history.historyHierDF import HistoryHierDF #class



class BatchDHondtThompsonSamplingINF(ABatchML):

    @staticmethod
    def getParameters():
        selectorIDs:List[str] = BatchFuzzyDHondt().getSelectorParameters().keys()
        negativeImplFeedback:List[str] = BatchFuzzyDHondtINF().getNegativeImplFeedbackParameters().keys()

        aDict:dict = {}
        for selectorIDH in selectorIDs:
            for nImplFeedbackI in negativeImplFeedback:
                keyIJ:str = str(selectorIDH) + nImplFeedbackI

                nImplFeedback:APenalization = BatchFuzzyDHondtINF().getNegativeImplFeedbackParameters()[nImplFeedbackI]
                selectorH:ADHondtSelector = BatchFuzzyDHondt().getSelectorParameters()[selectorIDH]

                aDict[keyIJ] = (selectorH, nImplFeedback)
        return aDict


    def run(self, batchID:str, jobID:str):

        divisionDatasetPercentualSize:int
        uBehaviour:str
        repetition:int
        divisionDatasetPercentualSize, uBehaviour, repetition = BatchParameters.getBatchParameters(self.datasetID)[batchID]

        selector, nImplFeedback = self.getParameters()[jobID]

        eTool:AEvalTool = EvalToolDHondtBanditVotes({})

        datasetID:str = "ml1m" + "Div" + str(divisionDatasetPercentualSize)

        rIDs, rDescs = InputRecomDefinition.exportPairOfRecomIdsAndRecomDescrsML()

        aDescNegDHontThompsonSamplingI:AggregationDescription = InputAggrDefinition.exportADescDHontThompsonSamplingINF(selector, nImplFeedback)

        pDescr:Portfolio1AggrDescription = Portfolio1AggrDescription(
            "DHondtThompsonSamplingINF" + jobID, rIDs, rDescs, aDescNegDHontThompsonSamplingI)

        model:DataFrame = ModelDefinition.createDHondtBanditsVotesModel(pDescr.getRecommendersIDs())

        simulator:Simulator = InputSimulatorDefinition.exportSimulatorML1M(
                batchID, divisionDatasetPercentualSize, uBehaviour, repetition)
        simulator.simulate([pDescr], [model], [eTool], HistoryHierDF)
