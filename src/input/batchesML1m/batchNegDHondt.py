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

from input.batchesML1m.batchDHondt import BatchDHondt #class

from aggregation.toolsDHontNF.penalizationOfResultsByNegImpFeedback.aPenalization import APenalization #class

from input.batchesML1m.aML1MConfig import AML1MConf #function

from aggregation.operators.aDHondtSelector import ADHondtSelector #class
from aggregation.toolsDHontNF.penalizationOfResultsByNegImpFeedback.penalUsingFiltering import PenalUsingFiltering #class

from input.aBatch import ABatch #class


class BatchNegDHondt(ABatch):

    def getNegativeImplFeedbackParameters(self):

        pToolOLin0802HLin1002:APenalization = InputAggrDefinition.exportAPenaltyToolOLin0802HLin1002(AML1MConf.numberOfAggrItems)

        pToolOStat08HLin1002:APenalization = InputAggrDefinition.exportAPenaltyToolOStat08HLin1002(AML1MConf.numberOfAggrItems)

        pToolFilterBord3Lengt100:PenalUsingFiltering = InputAggrDefinition.exportAPenaltyToolFiltering()

        aDict:dict = {}
        aDict["OLin0802HLin1002"] = pToolOLin0802HLin1002
        aDict["OStat08HLin1002"] = pToolOStat08HLin1002
        aDict["FilterBord3Lengt100"] = pToolFilterBord3Lengt100
        return aDict


    def getParameters(self):
        selectorIDs:List[str] = BatchDHondt().getSelectorParameters().keys()
        negativeImplFeedback:List[str] = self.getNegativeImplFeedbackParameters().keys()
        #lrClicks:List[float] = [0.2, 0.1, 0.02, 0.005]
        lrClicks:List[float] = [0.1]
        #lrViewDivisors:List[float] = [200, 500, 1000]
        lrViewDivisors:List[float] = [200]

        aDict:dict = {}
        for selectorIDH in selectorIDs:
            for nImplFeedbackI in negativeImplFeedback:
                for lrClickJ in lrClicks:
                    for lrViewDivisorK in lrViewDivisors:
                        keyIJ:str = str(selectorIDH) + "Clk" + str(lrClickJ).replace(".", "") + "ViewDivisor" + str(lrViewDivisorK).replace(".", "") + nImplFeedbackI
                        lrViewIJK:float = lrClickJ / lrViewDivisorK
                        eTool:AEvalTool = EvalToolDHondt({EvalToolDHondt.ARG_LEARNING_RATE_CLICKS: lrClickJ,
                                                          EvalToolDHondt.ARG_LEARNING_RATE_VIEWS: lrViewIJK})
                        nImplFeedback:APenalization = self.getNegativeImplFeedbackParameters()[nImplFeedbackI]
                        selectorH:ADHondtSelector = BatchDHondt().getSelectorParameters()[selectorIDH]

                        aDict[keyIJ] = (selectorH, nImplFeedback, eTool)
        return aDict


    def run(self, batchID:str, jobID:str):

        from execute.generateBatches import BatchParameters #class
        divisionDatasetPercentualSize:int
        uBehaviour:str
        repetition:int
        divisionDatasetPercentualSize, uBehaviour, repetition = BatchParameters.getBatchParameters()[batchID]

        #eTool:AEvalTool
        selector, nImplFeedback, eTool = self.getParameters()[jobID]

        aConf:AML1MConf = AML1MConf(batchID, divisionDatasetPercentualSize, uBehaviour, repetition)

        rIDs, rDescs = InputRecomDefinition.exportPairOfRecomIdsAndRecomDescrs(aConf.datasetID)

        aDescNegDHont:AggregationDescription = InputAggrDefinition.exportADescNegDHont(selector, nImplFeedback)

        pDescr:Portfolio1AggrDescription = Portfolio1AggrDescription(
            "NegDHont" + jobID, rIDs, rDescs, aDescNegDHont)

        model:DataFrame = ModelDefinition.createDHontModel(pDescr.getRecommendersIDs())

        aConf.run(pDescr, model, eTool)



if __name__ == "__main__":
   os.chdir("..")
   os.chdir("..")
   print(os.getcwd())
   BatchNegDHondt.generateBatches()