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

from input.inputRecomSTDefinition import InputRecomSTDefinition #class

from aggregation.operators.aDHondtSelector import ADHondtSelector #class
from aggregation.operators.rouletteWheelSelector import RouletteWheelSelector #class
from aggregation.operators.theMostVotedItemSelector import TheMostVotedItemSelector #class

from input.inputABatchDefinition import InputABatchDefinition
from input.aBatchST import ABatchST #class

from input.inputSimulatorDefinition import InputSimulatorDefinition #class

from simulator.simulator import Simulator #class

from history.historyHierDF import HistoryHierDF #class

from input.batchesML1m.batchMLFuzzyDHondtDirectOptimizeThompsonSamplingMMR import BatchMLFuzzyDHondtDirectOptimizeThompsonSamplingMMR #class


class BatchSTFuzzyDHondtDirectOptimizeThompsonSamplingMMR(ABatchST):

    SLCTR_ROULETTE1:str = BatchMLFuzzyDHondtDirectOptimizeThompsonSamplingMMR.SLCTR_ROULETTE1
    SLCTR_ROULETTE2:str = BatchMLFuzzyDHondtDirectOptimizeThompsonSamplingMMR.SLCTR_ROULETTE2
    SLCTR_FIXED:str = BatchMLFuzzyDHondtDirectOptimizeThompsonSamplingMMR.SLCTR_FIXED

    @staticmethod
    def getParameters():
        return BatchMLFuzzyDHondtDirectOptimizeThompsonSamplingMMR.getParameters()


    def run(self, batchID:str, jobID:str):

        divisionDatasetPercentualSize:int
        uBehaviour:str
        repetition:int
        divisionDatasetPercentualSize, uBehaviour, repetition = InputABatchDefinition.getBatchParameters(self.datasetID)[batchID]

        selector, discFactor = self.getParameters()[jobID]

        eTool:AEvalTool = EvalToolDHondtBanditVotes({})

        rIDs, rDescs = InputRecomSTDefinition.exportPairOfRecomIdsAndRecomDescrs()

        aDescDHont:AggregationDescription = InputAggrDefinition.exportADescDHondtDirectOptimizeThompsonSamplingMMR(selector, discFactor)

        pDescr:Portfolio1AggrDescription = Portfolio1AggrDescription(
            "FDHondtDirectOptimizeThompsonSamplingMMR" + jobID, rIDs, rDescs, aDescDHont)

        model:DataFrame = ModelDefinition.createDHondtBanditsVotesModel(pDescr.getRecommendersIDs())

        simulator:Simulator = InputSimulatorDefinition.exportSimulatorSlantour(
                batchID, divisionDatasetPercentualSize, uBehaviour, repetition)
        simulator.simulate([pDescr], [model], [eTool], [HistoryHierDF(pDescr.getPortfolioID())])



if __name__ == "__main__":
    os.chdir("..")
    os.chdir("..")
    print(os.getcwd())

    BatchSTFuzzyDHondtDirectOptimizeThompsonSamplingMMR.generateAllBatches()
