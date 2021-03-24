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
from batchDefinition.modelDefinition import ModelDefinition

from batchDefinition.inputRecomMLDefinition import InputRecomMLDefinition #class

from aggregation.operators.aDHondtSelector import ADHondtSelector #class
from aggregation.operators.rouletteWheelSelector import RouletteWheelSelector #class
from aggregation.operators.theMostVotedItemSelector import TheMostVotedItemSelector #class
from aggregation.aggrDHondtDirectOptimizeThompsonSampling import AggrDHondtDirectOptimizeThompsonSampling #class

from batchDefinition.inputABatchDefinition import InputABatchDefinition
from batchDefinition.aBatchDefinitionML import ABatchDefinitionML #class
from batchDefinition.ml1m.batchDefMLFuzzyDHondtDirectOptimizeThompsonSampling import BatchDefMLFuzzyDHondtDirectOptimizeThompsonSampling #class

from batchDefinition.ml1m.batchDefMLFuzzyDHondt import BatchDefMLFuzzyDHondt #class

from batchDefinition.inputSimulatorDefinition import InputSimulatorDefinition #class

from simulator.simulator import Simulator #class

from history.historyHierDF import HistoryHierDF #class


class BatchDefMLFuzzyDHondtDirectOptimizeThompsonSamplingMMR(ABatchDefinitionML):

    SLCTR_ROULETTE1:str = BatchDefMLFuzzyDHondtDirectOptimizeThompsonSampling.SLCTR_ROULETTE1
    SLCTR_ROULETTE2:str = BatchDefMLFuzzyDHondtDirectOptimizeThompsonSampling.SLCTR_ROULETTE2
    SLCTR_FIXED:str = BatchDefMLFuzzyDHondtDirectOptimizeThompsonSampling.SLCTR_FIXED


    selectorIDs:List[str] = BatchDefMLFuzzyDHondt.selectorIDs
    discFactors:List[str] = [AggrDHondtDirectOptimizeThompsonSampling.DISCFACTOR_DCG,
                             AggrDHondtDirectOptimizeThompsonSampling.DISCFACTOR_POWERLAW]

    def getBatchName(self):
        return "FDHondtDirectOptimizeThompsonSamplingMMR"

    def getParameters(self):

        aDict:Dict[str,object] = {}
        for selectorIDI in self.selectorIDs:
            for discFactorsI in self.discFactors:
                keyIJ:str = str(selectorIDI) + discFactorsI
                selectorIJK:ADHondtSelector = BatchDefMLFuzzyDHondt().getSelectorParameters()[selectorIDI]
                aDict[keyIJ] = (selectorIJK, discFactorsI)
        return aDict


    def run(self, batchID:str, jobID:str):

        divisionDatasetPercentualSize:int
        uBehaviour:str
        repetition:int
        divisionDatasetPercentualSize, uBehaviour, repetition = InputABatchDefinition.getBatchParameters(self.datasetID)[batchID]

        selector, discFactor = self.getParameters()[jobID]

        eTool:AEvalTool = EvalToolDHondtBanditVotes({})

        rIDs, rDescs = InputRecomMLDefinition.exportPairOfRecomIdsAndRecomDescrs()

        aDescDHont:AggregationDescription = InputAggrDefinition.exportADescDHondtDirectOptimizeThompsonSamplingMMR(selector, discFactor)

        pDescr:Portfolio1AggrDescription = Portfolio1AggrDescription(
            self.getBatchName() + jobID, rIDs, rDescs, aDescDHont)

        model:DataFrame = ModelDefinition.createDHondtBanditsVotesModel(pDescr.getRecommendersIDs())

        simulator:Simulator = InputSimulatorDefinition.exportSimulatorML1M(
                batchID, divisionDatasetPercentualSize, uBehaviour, repetition)
        simulator.simulate([pDescr], [model], [eTool], [HistoryHierDF(pDescr.getPortfolioID())])



if __name__ == "__main__":
    os.chdir("..")
    os.chdir("..")
    print(os.getcwd())

    BatchDefMLFuzzyDHondtDirectOptimizeThompsonSamplingMMR.generateAllBatches()