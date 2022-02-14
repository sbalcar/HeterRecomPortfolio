#!/usr/bin/python3

import os

from typing import List #class
from typing import Dict #class

from pandas.core.frame import DataFrame #class

from portfolioDescription.portfolio1AggrDescription import Portfolio1AggrDescription #class

from evaluationTool.aEvalTool import AEvalTool #class
from evaluationTool.evalToolDHondtBanditVotes import EvalToolDHondtBanditVotes #class

from aggregationDescription.aggregationDescription import AggregationDescription #class

from batchDefinition.inputAggrDefinition import InputAggrDefinition  # class

from batchDefinition.inputRecomSTDefinition import InputRecomSTDefinition #class

from aggregation.operators.aDHondtSelector import ADHondtSelector #class
from aggregation.operators.rouletteWheelSelector import RouletteWheelSelector #class
from aggregation.operators.theMostVotedItemSelector import TheMostVotedItemSelector #class

from batchDefinition.inputABatchDefinition import InputABatchDefinition
from batchDefinition.aBatchDefinitionST import ABatchDefinitionST #class

from batchDefinition.inputSimulatorDefinition import InputSimulatorDefinition #class

from simulator.simulator import Simulator #class

from history.historyHierDF import HistoryHierDF #class

from batchDefinition.ml1m.batchDefMLFuzzyDHondtDirectOptimizeThompsonSampling import BatchDefMLFuzzyDHondtDirectOptimizeThompsonSampling #class
from batchDefinition.ml1m.batchDefMLFuzzyDHondt import BatchDefMLFuzzyDHondt #class

from portfolioModel.pModelBandit import PModelBandit #class
from portfolioModel.pModelDHondtBanditsVotes import PModelDHondtBanditsVotes #class
from portfolioModel.pModelDHondt import PModelDHondt #class


class BatchDefSTFuzzyDHondtDirectOptimizeThompsonSampling(ABatchDefinitionST):

    SLCTR_ROULETTE1:str = BatchDefMLFuzzyDHondtDirectOptimizeThompsonSampling.SLCTR_ROULETTE1
    SLCTR_ROULETTE2:str = BatchDefMLFuzzyDHondtDirectOptimizeThompsonSampling.SLCTR_ROULETTE2
    SLCTR_FIXED:str = BatchDefMLFuzzyDHondtDirectOptimizeThompsonSampling.SLCTR_FIXED

    lrClicks:List[float] = [0.03]
    lrViewDivisors:List[float] = [250]

    selectorIDs:List[str] = BatchDefMLFuzzyDHondt.selectorIDs

    def getBatchName(self):
        return "FDHondtDirectOptimizeThompsonSampling"

    def getParameters(self):
        batchDefMLFuzzyDHondtDirectOptimizeThompsonSampling = BatchDefMLFuzzyDHondtDirectOptimizeThompsonSampling()
        batchDefMLFuzzyDHondtDirectOptimizeThompsonSampling.selectorIDs = self.selectorIDs
        return batchDefMLFuzzyDHondtDirectOptimizeThompsonSampling.getParameters()


    def run(self, batchID:str, jobID:str):

        divisionDatasetPercentualSize:int
        uBehaviour:str
        repetition:int
        divisionDatasetPercentualSize, uBehaviour, repetition = InputABatchDefinition().getBatchParameters(self.datasetID)[batchID]

        selector = self.getParameters()[jobID]

        eTool:AEvalTool = EvalToolDHondtBanditVotes({})

        rIDs, rDescs = InputRecomSTDefinition.exportPairOfRecomIdsAndRecomDescrs()

        aDescDHont:AggregationDescription = InputAggrDefinition.exportADescDHondtDirectOptimizeThompsonSampling(selector)

        pDescr:Portfolio1AggrDescription = Portfolio1AggrDescription(
            self.getBatchName() + jobID, rIDs, rDescs, aDescDHont)

        model:DataFrame = PModelDHondtBanditsVotes(pDescr.getRecommendersIDs())

        simulator:Simulator = InputSimulatorDefinition.exportSimulatorSlantour(
                batchID, divisionDatasetPercentualSize, uBehaviour, repetition)
        simulator.simulate([pDescr], [model], [eTool], [HistoryHierDF(pDescr.getPortfolioID())])



if __name__ == "__main__":
    os.chdir("..")
    os.chdir("..")
    print(os.getcwd())

    BatchDefSTFuzzyDHondtDirectOptimizeThompsonSampling.generateAllBatches(InputABatchDefinition())