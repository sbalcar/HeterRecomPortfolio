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

from batchDefinition.ml1m.batchDefMLFuzzyDHondtDirectOptimizeThompsonSamplingMMR import BatchDefMLFuzzyDHondtDirectOptimizeThompsonSamplingMMR #class

from aggregation.aggrDHondtDirectOptimizeThompsonSampling import AggrDHondtDirectOptimizeThompsonSampling #class

from portfolioModel.pModelBandit import PModelBandit #class
from portfolioModel.pModelDHondtBanditsVotes import PModelDHondtBanditsVotes #class
from portfolioModel.pModelDHondt import PModelDHondt #class


class BatchDefSTFuzzyDHondtDirectOptimizeThompsonSamplingMMR(ABatchDefinitionST):

    SLCTR_ROULETTE1:str = BatchDefMLFuzzyDHondtDirectOptimizeThompsonSamplingMMR.SLCTR_ROULETTE1
    SLCTR_ROULETTE2:str = BatchDefMLFuzzyDHondtDirectOptimizeThompsonSamplingMMR.SLCTR_ROULETTE2
    SLCTR_FIXED:str = BatchDefMLFuzzyDHondtDirectOptimizeThompsonSamplingMMR.SLCTR_FIXED

    discFactors:List[str] = [AggrDHondtDirectOptimizeThompsonSampling.DISCFACTOR_DCG,
                              AggrDHondtDirectOptimizeThompsonSampling.DISCFACTOR_POWERLAW]

    def getBatchName(self):
        return "FDHondtDirectOptimizeThompsonSamplingMMR"

    def getParameters(self):
        batchDefMLFuzzyDHondtDirectOptimizeThompsonSamplingMMR = BatchDefMLFuzzyDHondtDirectOptimizeThompsonSamplingMMR()
        batchDefMLFuzzyDHondtDirectOptimizeThompsonSamplingMMR.selectorIDs = self.selectorIDs
        batchDefMLFuzzyDHondtDirectOptimizeThompsonSamplingMMR.discFactors = self.discFactors
        return batchDefMLFuzzyDHondtDirectOptimizeThompsonSamplingMMR.getParameters()


    def run(self, batchID:str, jobID:str):

        divisionDatasetPercentualSize:int
        uBehaviour:str
        repetition:int
        divisionDatasetPercentualSize, uBehaviour, repetition = InputABatchDefinition().getBatchParameters(self.datasetID)[batchID]

        selector, discFactor = self.getParameters()[jobID]

        eTool:AEvalTool = EvalToolDHondtBanditVotes({})

        rIDs, rDescs = InputRecomSTDefinition.exportPairOfRecomIdsAndRecomDescrs()

        aDescDHont:AggregationDescription = InputAggrDefinition.exportADescDHondtDirectOptimizeThompsonSamplingMMR(selector, discFactor)

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

    #BatchDefSTFuzzyDHondtDirectOptimizeThompsonSamplingMMR.generateAllBatches(InputABatchDefinition())
    BatchDefSTFuzzyDHondtDirectOptimizeThompsonSamplingMMR().run("stDiv90Ulinear0109R1", "FixedDCG")