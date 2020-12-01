#!/usr/bin/python3

import os

from typing import List

from pandas.core.frame import DataFrame #class

from portfolioDescription.portfolio1AggrDescription import Portfolio1AggrDescription #class

from evaluationTool.aEvalTool import AEvalTool #class
from evaluationTool.evalToolDHondt import EvalToolDHondt #class

from aggregationDescription.aggregationDescription import AggregationDescription #class

from input.inputAggrDefinition import InputAggrDefinition, ModelDefinition  #class

from input.InputRecomDefinition import InputRecomDefinition #class

from input.batchesML1m.batchFuzzyDHondt import BatchFuzzyDHondt #class
from input.batchesML1m.batchFuzzyDHondtINF import BatchFuzzyDHondtINF #class

from aggregation.negImplFeedback.aPenalization import APenalization #class

from input.batchesML1m.aML1MConfig import AML1MConf #function

from aggregation.operators.aDHondtSelector import ADHondtSelector #class

from input.aBatch import ABatch #class

from evaluationTool.evalToolDHondtBanditVotes import EvalToolDHondtBanditVotes #class


class BatchDHondtThompsonSamplingINF(ABatch):

    def getParameters(self):
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

        from execute.generateBatches import BatchParameters #class
        divisionDatasetPercentualSize:int
        uBehaviour:str
        repetition:int
        divisionDatasetPercentualSize, uBehaviour, repetition = BatchParameters.getBatchParameters()[batchID]

        selector, nImplFeedback = self.getParameters()[jobID]

        eTool:AEvalTool = EvalToolDHondtBanditVotes({})

        aConf:AML1MConf = AML1MConf(batchID, divisionDatasetPercentualSize, uBehaviour, repetition)

        rIDs, rDescs = InputRecomDefinition.exportPairOfRecomIdsAndRecomDescrs(aConf.datasetID)

        aDescNegDHontThompsonSamplingI:AggregationDescription = InputAggrDefinition.exportADescNegDHontThompsonSampling(selector, nImplFeedback)

        pDescr:Portfolio1AggrDescription = Portfolio1AggrDescription(
            "NegDHondtThompsonSampling" + jobID, rIDs, rDescs, aDescNegDHontThompsonSamplingI)

        model:DataFrame = ModelDefinition.createDHondtBanditsVotesModel(pDescr.getRecommendersIDs())

        aConf.run(pDescr, model, eTool)
