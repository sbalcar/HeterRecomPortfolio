#!/usr/bin/python3

import os

from typing import List #class
from typing import Dict #class

from pandas.core.frame import DataFrame #class

from portfolioDescription.portfolio1AggrDescription import Portfolio1AggrDescription #class

from evaluationTool.aEvalTool import AEvalTool #class
from evaluationTool.evalToolDHondtBanditVotes import EvalToolDHondtBanditVotes #class

from aggregationDescription.aggregationDescription import AggregationDescription #class

from input.inputAggrDefinition import InputAggrDefinition  # class
from input.modelDefinition import ModelDefinition

from input.inputRecomMLDefinition import InputRecomMLDefinition #class

from aggregation.operators.aDHondtSelector import ADHondtSelector #class
from aggregation.operators.rouletteWheelSelector import RouletteWheelSelector #class
from aggregation.operators.theMostVotedItemSelector import TheMostVotedItemSelector #class

from aggregation.negImplFeedback.aPenalization import APenalization #class

from input.inputABatchDefinition import InputABatchDefinition
from input.aBatchML import ABatchML #class
from input.batchesML1m.batchMLFuzzyDHondtThompsonSamplingINF import BatchMLFuzzyDHondtThompsonSamplingINF #class

from input.inputSimulatorDefinition import InputSimulatorDefinition #class

from simulator.simulator import Simulator #class

from history.historyHierDF import HistoryHierDF #class
from input.batchesML1m.batchMLFuzzyDHondt import BatchMLFuzzyDHondt #class
from input.batchesML1m.batchMLFuzzyDHondtDirectOptimizeThompsonSampling import BatchMLFuzzyDHondtDirectOptimizeThompsonSampling #class


class BatchMLFuzzyDHondtDirectOptimizeThompsonSamplingINF(ABatchML):

    selectorIDs:List[str] = BatchMLFuzzyDHondt.selectorIDs
    negativeImplFeedback:List[str] = BatchMLFuzzyDHondtThompsonSamplingINF().getPenalFncs().keys()
    discFactors:List[str] = BatchMLFuzzyDHondtDirectOptimizeThompsonSampling.discFactors

    @classmethod
    def getParameters(cls):

        aDict:Dict[str,object] = {}
        for selectorIDH in cls.selectorIDs:
            for nImplFeedbackI in cls.negativeImplFeedback:
                keyIJ:str = str(selectorIDH) + nImplFeedbackI

                nImplFeedback:APenalization = BatchMLFuzzyDHondtThompsonSamplingINF().getPenalFncs()[nImplFeedbackI]
                selectorH:ADHondtSelector = BatchMLFuzzyDHondt().getSelectorParameters()[selectorIDH]

                aDict[keyIJ] = (selectorH, nImplFeedback)
        return aDict

    def run(self, batchID:str, jobID:str):

        divisionDatasetPercentualSize:int
        uBehaviour:str
        repetition:int
        divisionDatasetPercentualSize, uBehaviour, repetition = InputABatchDefinition.getBatchParameters(self.datasetID)[batchID]

        selector, nImplFeedback = self.getParameters()[jobID]

        eTool:AEvalTool = EvalToolDHondtBanditVotes({})

        rIDs, rDescs = InputRecomMLDefinition.exportPairOfRecomIdsAndRecomDescrs()

        aDescDHont:AggregationDescription = InputAggrDefinition.exportADescDHondtDirectOptimizeThompsonSamplingINF(selector, nImplFeedback)

        pDescr:Portfolio1AggrDescription = Portfolio1AggrDescription(
            "FDHondtDirectOptimizeThompsonSamplingINF" + jobID.replace("Uniform",""), rIDs, rDescs, aDescDHont)

        model:DataFrame = ModelDefinition.createDHondtBanditsVotesModel(pDescr.getRecommendersIDs())

        simulator:Simulator = InputSimulatorDefinition.exportSimulatorML1M(
                batchID, divisionDatasetPercentualSize, uBehaviour, repetition)
        simulator.simulate([pDescr], [model], [eTool], [HistoryHierDF(pDescr.getPortfolioID())])



if __name__ == "__main__":
    os.chdir("..")
    os.chdir("..")
    print(os.getcwd())

    BatchMLFuzzyDHondtDirectOptimizeThompsonSamplingINF.generateAllBatches()
