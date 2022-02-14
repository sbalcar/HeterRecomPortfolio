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

from batchDefinition.inputRecomMLDefinition import InputRecomMLDefinition #class

from batchDefinition.ml1m.batchDefMLFuzzyDHondt import BatchDefMLFuzzyDHondt #class

from aggregation.negImplFeedback.aPenalization import APenalization #class

from aggregation.operators.aDHondtSelector import ADHondtSelector #class
from aggregation.negImplFeedback.penalUsingFiltering import PenalUsingFiltering #class
from aggregation.negImplFeedback.penalUsingProbability import PenalUsingProbability #class

from batchDefinition.inputABatchDefinition import InputABatchDefinition
from batchDefinition.aBatchDefinitionML import ABatchDefinitionML #class

from batchDefinition.inputSimulatorDefinition import InputSimulatorDefinition #class

from batchDefinition.inputAggrDefinition import PenalizationToolDefinition #class

from simulator.simulator import Simulator #class

from history.historyHierDF import HistoryHierDF #class

from portfolioModel.pModelBandit import PModelBandit #class
from portfolioModel.pModelDHondtBanditsVotes import PModelDHondtBanditsVotes #class
from portfolioModel.pModelDHondt import PModelDHondt #class


class BatchDefMLFuzzyDHondtINF(ABatchDefinitionML):

    lrClicks:List[float] = [0.2, 0.1, 0.03, 0.005]
    lrViewDivisors:List[float] = [250, 500, 1000]

    def getBatchName(self):
        return "FDHondtINF"

    @staticmethod
    def getNegativeImplFeedbackParameters_():

        pToolOLin0802HLin1002:APenalization = PenalizationToolDefinition.exportPenaltyToolOLin0802HLin1002(InputSimulatorDefinition.numberOfAggrItems)

        pToolOStat08HLin1002:APenalization = PenalizationToolDefinition.exportPenaltyToolOStat08HLin1002(InputSimulatorDefinition.numberOfAggrItems)

        pProbToolOLin0802HLin1002:APenalization = PenalizationToolDefinition.exportProbPenaltyToolOStat08HLin1002(InputSimulatorDefinition.numberOfAggrItems)

        pProbToolOStat08HLin1002:APenalization = PenalizationToolDefinition.exportProbPenaltyToolOLin0802HLin1002(InputSimulatorDefinition.numberOfAggrItems)

        pToolFilterBord3Lengt100:APenalization = PenalizationToolDefinition.exportPenaltyToolFiltering()


        aDict:Dict[str,object] = {}
        aDict["OLin0802HLin1002"] = pToolOLin0802HLin1002
        aDict["OStat08HLin1002"] = pToolOStat08HLin1002
        aDict["ProbOLin0802HLin1002"] = pProbToolOLin0802HLin1002
        aDict["ProbOStat08HLin1002"] = pProbToolOStat08HLin1002

        aDict["TFilterBord3Lengt100"] = pToolFilterBord3Lengt100

        return aDict


    @classmethod
    def getNegativeImplFeedbackParameters(cls):
        from batchDefinition.ml1m.batchDefMLFuzzyDHondtThompsonSamplingINF import BatchDefMLFuzzyDHondtThompsonSamplingINF #class
        return BatchDefMLFuzzyDHondtThompsonSamplingINF.getPenalFncs()


    @classmethod
    def getParameters(cls):
        selectorIDs:List[str] = BatchDefMLFuzzyDHondt().getSelectorParameters().keys()
        negativeImplFeedback:List[str] = cls.getNegativeImplFeedbackParameters().keys()

        aDict:Dict[str,object] = {}
        for selectorIDH in selectorIDs:
            for nImplFeedbackI in negativeImplFeedback:
                for lrClickJ in cls.lrClicks:
                    for lrViewDivisorK in cls.lrViewDivisors:

                        keyIJ:str = str(selectorIDH) + "Clk" + str(lrClickJ).replace(".", "") + "ViewDivisor" + str(lrViewDivisorK).replace(".", "") + nImplFeedbackI
                        lrViewIJK:float = lrClickJ / lrViewDivisorK
                        eTool:AEvalTool = EvalToolDHondt({EvalToolDHondt.ARG_LEARNING_RATE_CLICKS: lrClickJ,
                                                          EvalToolDHondt.ARG_LEARNING_RATE_VIEWS: lrViewIJK})
                        nImplFeedback:APenalization = cls.getNegativeImplFeedbackParameters()[nImplFeedbackI]
                        selectorH:ADHondtSelector = BatchDefMLFuzzyDHondt().getSelectorParameters()[selectorIDH]

                        aDict[keyIJ] = (selectorH, nImplFeedback, eTool)
        return aDict


    def run(self, batchID:str, jobID:str):

        divisionDatasetPercentualSize:int
        uBehaviour:str
        repetition:int
        divisionDatasetPercentualSize, uBehaviour, repetition = InputABatchDefinition().getBatchParameters(self.datasetID)[batchID]

        #eTool:AEvalTool
        selector, nImplFeedback, eTool = self.getParameters()[jobID]

        rIDs, rDescs = InputRecomMLDefinition.exportPairOfRecomIdsAndRecomDescrs()

        aDescNegDHont:AggregationDescription = InputAggrDefinition.exportADescDHondtINF(selector, nImplFeedback)

        pDescr:Portfolio1AggrDescription = Portfolio1AggrDescription(
            self.getBatchName() + jobID, rIDs, rDescs, aDescNegDHont)

        model:DataFrame = PModelDHondt(pDescr.getRecommendersIDs())

        simulator:Simulator = InputSimulatorDefinition.exportSimulatorML1M(
                batchID, divisionDatasetPercentualSize, uBehaviour, repetition)
        simulator.simulate([pDescr], [model], [eTool], [HistoryHierDF(pDescr.getPortfolioID())])




if __name__ == "__main__":
   os.chdir("..")
   os.chdir("..")
   print(os.getcwd())

   BatchDefMLFuzzyDHondtINF().generateAllBatches(InputABatchDefinition())