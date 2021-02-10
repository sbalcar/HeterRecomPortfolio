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
from aggregation.aggrDHondtDirectOptimizeThompsonSampling import AggrDHondtDirectOptimizeThompsonSampling #class

from input.inputABatchDefinition import InputABatchDefinition
from input.aBatchML import ABatchML #class
from input.batchesML1m.batchMLFuzzyDHondtThompsonSampling import BatchMLFuzzyDHondtThompsonSampling #class
from input.batchesML1m.batchMLFuzzyDHondt import BatchMLFuzzyDHondt #class

from input.inputSimulatorDefinition import InputSimulatorDefinition #class

from simulator.simulator import Simulator #class

from history.historyHierDF import HistoryHierDF #class


class BatchMLFuzzyDHondtDirectOptimizeThompsonSampling(ABatchML):

    SLCTR_ROULETTE1:str = "Roulette1"
    SLCTR_ROULETTE2:str = "Roulette3"
    SLCTR_FIXED:str = "Fixed"

    selectorIDs:List[str] = BatchMLFuzzyDHondt.selectorIDs
    discFactors:List[str] = [AggrDHondtDirectOptimizeThompsonSampling.DISCFACTOR_DCG,
                             AggrDHondtDirectOptimizeThompsonSampling.DISCFACTOR_POWERLAW,
                             AggrDHondtDirectOptimizeThompsonSampling.DISCFACTOR_UNIFORM]

    @classmethod
    def getParameters(cls):

        aDict:Dict[str,object] = {}
        for selectorIDI in cls.selectorIDs:
            for discFactorsI in cls.discFactors:
                keyIJ:str = str(selectorIDI) + discFactorsI
                selectorIJK:ADHondtSelector = BatchMLFuzzyDHondt().getSelectorParameters()[selectorIDI]
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

        aDescDHont:AggregationDescription = InputAggrDefinition.exportADescDHondtDirectOptimizeThompsonSampling(selector, discFactor)

        pDescr:Portfolio1AggrDescription = Portfolio1AggrDescription(
            "FDHondtDirectOptimizeThompsonSampling" + jobID, rIDs, rDescs, aDescDHont)

        model:DataFrame = ModelDefinition.createDHondtBanditsVotesModel(pDescr.getRecommendersIDs())

        simulator:Simulator = InputSimulatorDefinition.exportSimulatorML1M(
                batchID, divisionDatasetPercentualSize, uBehaviour, repetition)
        simulator.simulate([pDescr], [model], [eTool], [HistoryHierDF(pDescr.getPortfolioID())])



if __name__ == "__main__":
    os.chdir("..")
    os.chdir("..")
    print(os.getcwd())

    BatchMLFuzzyDHondtDirectOptimizeThompsonSampling.generateAllBatches()