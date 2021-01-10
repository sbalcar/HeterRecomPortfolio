#!/usr/bin/python3

import os

from typing import List

from pandas.core.frame import DataFrame #class

from portfolioDescription.portfolio1AggrDescription import Portfolio1AggrDescription #class

from evaluationTool.aEvalTool import AEvalTool #class
from evaluationTool.evalToolDHondt import EvalToolDHondt #class

from aggregationDescription.aggregationDescription import AggregationDescription #class

from input.inputAggrDefinition import InputAggrDefinition, ModelDefinition  #class

from input.inputRecomMLDefinition import InputRecomMLDefinition #class

from input.batchesML1m.batchMLFuzzyDHondt import BatchMLFuzzyDHondt #class

from aggregation.negImplFeedback.aPenalization import APenalization #class

from aggregation.operators.aDHondtSelector import ADHondtSelector #class
from aggregation.negImplFeedback.penalUsingFiltering import PenalUsingFiltering #class

from input.aBatch import BatchParameters #class
from input.aBatchML import ABatchML #class

from input.inputSimulatorDefinition import InputSimulatorDefinition #class

from simulator.simulator import Simulator #class

from history.historyHierDF import HistoryHierDF #class


class BatchFuzzyDHondtINF(ABatchML):

    @staticmethod
    def getNegativeImplFeedbackParameters():

        pToolOLin0802HLin1002:APenalization = InputAggrDefinition.exportAPenaltyToolOLin0802HLin1002(InputSimulatorDefinition.numberOfAggrItems)

        pToolOStat08HLin1002:APenalization = InputAggrDefinition.exportAPenaltyToolOStat08HLin1002(InputSimulatorDefinition.numberOfAggrItems)

        pToolFilterBord3Lengt100:PenalUsingFiltering = InputAggrDefinition.exportAPenaltyToolFiltering()

        aDict:dict = {}
        aDict["OLin0802HLin1002"] = pToolOLin0802HLin1002
        aDict["OStat08HLin1002"] = pToolOStat08HLin1002
        aDict["FilterBord3Lengt100"] = pToolFilterBord3Lengt100
        return aDict

    @staticmethod
    def getParameters():
        selectorIDs:List[str] = BatchMLFuzzyDHondt().getSelectorParameters().keys()
        negativeImplFeedback:List[str] = BatchFuzzyDHondtINF.getNegativeImplFeedbackParameters().keys()
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
                        nImplFeedback:APenalization = BatchFuzzyDHondtINF.getNegativeImplFeedbackParameters()[nImplFeedbackI]
                        selectorH:ADHondtSelector = BatchMLFuzzyDHondt().getSelectorParameters()[selectorIDH]

                        aDict[keyIJ] = (selectorH, nImplFeedback, eTool)
        return aDict


    def run(self, batchID:str, jobID:str):

        divisionDatasetPercentualSize:int
        uBehaviour:str
        repetition:int
        divisionDatasetPercentualSize, uBehaviour, repetition = BatchParameters.getBatchParameters(self.datasetID)[batchID]

        #eTool:AEvalTool
        selector, nImplFeedback, eTool = self.getParameters()[jobID]

        datasetID:str = "ml1m" + "Div" + str(divisionDatasetPercentualSize)

        rIDs, rDescs = InputRecomMLDefinition.exportPairOfRecomIdsAndRecomDescrs()

        aDescNegDHont:AggregationDescription = InputAggrDefinition.exportADescDHontINF(selector, nImplFeedback)

        pDescr:Portfolio1AggrDescription = Portfolio1AggrDescription(
            "FDHontINF" + jobID, rIDs, rDescs, aDescNegDHont)

        model:DataFrame = ModelDefinition.createDHontModel(pDescr.getRecommendersIDs())

        simulator:Simulator = InputSimulatorDefinition.exportSimulatorML1M(
                batchID, divisionDatasetPercentualSize, uBehaviour, repetition)
        simulator.simulate([pDescr], [model], [eTool], HistoryHierDF)




if __name__ == "__main__":
   os.chdir("..")
   os.chdir("..")
   print(os.getcwd())
   BatchFuzzyDHondtINF.generateBatches()