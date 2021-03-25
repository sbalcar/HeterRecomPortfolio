#!/usr/bin/python3

import os

from typing import List #class
from typing import Dict #class

from pandas.core.frame import DataFrame #class

from portfolioDescription.portfolio1AggrDescription import Portfolio1AggrDescription #class

from evaluationTool.aEvalTool import AEvalTool #class
from evaluationTool.evalToolDHondt import EvalToolDHondt #class

from aggregationDescription.aggregationDescription import AggregationDescription #class

from batchDefinition.inputAggrDefinition import InputAggrDefinition  # class
from batchDefinition.modelDefinition import ModelDefinition

from batchDefinition.inputRecomSTDefinition import InputRecomSTDefinition #class

from aggregation.operators.aDHondtSelector import ADHondtSelector #class
from aggregation.operators.rouletteWheelSelector import RouletteWheelSelector #class
from aggregation.operators.theMostVotedItemSelector import TheMostVotedItemSelector #class

from batchDefinition.inputABatchDefinition import InputABatchDefinition
from batchDefinition.aBatchDefinitionST import ABatchDefinitionST #class

from batchDefinition.inputSimulatorDefinition import InputSimulatorDefinition #class

from simulator.simulator import Simulator #class

from history.historyHierDF import HistoryHierDF #class

from batchDefinition.ml1m.batchDefMLFuzzyDHondtDirectOptimize import BatchDefMLFuzzyDHondtDirectOptimize #clas

from batchDefinition.ml1m.batchDefMLFuzzyDHondt import BatchDefMLFuzzyDHondt #class


class BatchDefSTFuzzyDHondtDirectOptimize(ABatchDefinitionST):

    SLCTR_ROULETTE1:str = BatchDefMLFuzzyDHondtDirectOptimize.SLCTR_ROULETTE1
    SLCTR_ROULETTE2:str = BatchDefMLFuzzyDHondtDirectOptimize.SLCTR_ROULETTE2
    SLCTR_FIXED:str = BatchDefMLFuzzyDHondtDirectOptimize.SLCTR_FIXED

    lrClicks:List[float] = [0.03]
    lrViewDivisors:List[float] = [250]

    selectorIDs:List[str] = BatchDefMLFuzzyDHondt.selectorIDs

    def getBatchName(self):
        return "FDHondtDirectOptimize"
    
    def getParameters(self):
        batchDefMLFuzzyDHondtDirectOptimize = BatchDefMLFuzzyDHondtDirectOptimize()
        batchDefMLFuzzyDHondtDirectOptimize.lrClicks:List[float] = self.lrClicks
        batchDefMLFuzzyDHondtDirectOptimize.lrViewDivisors:List[float] = self.lrViewDivisors
        batchDefMLFuzzyDHondtDirectOptimize.selectorIDs = self.selectorIDs
        return batchDefMLFuzzyDHondtDirectOptimize.getParameters()


    def run(self, batchID:str, jobID:str):

        divisionDatasetPercentualSize:int
        uBehaviour:str
        repetition:int
        divisionDatasetPercentualSize, uBehaviour, repetition = InputABatchDefinition.getBatchParameters(self.datasetID)[batchID]

        #eTool:AEvalTool
        selector, eTool = self.getParameters()[jobID]

        rIDs, rDescs = InputRecomSTDefinition.exportPairOfRecomIdsAndRecomDescrs()

        aDescDHont:AggregationDescription = InputAggrDefinition.exportADescDHondtDirectOptimize(selector)

        pDescr:Portfolio1AggrDescription = Portfolio1AggrDescription(
            self.getBatchName() + jobID, rIDs, rDescs, aDescDHont)

        model:DataFrame = ModelDefinition.createDHontModel(pDescr.getRecommendersIDs())

        simulator:Simulator = InputSimulatorDefinition.exportSimulatorSlantour(
                batchID, divisionDatasetPercentualSize, uBehaviour, repetition)
        simulator.simulate([pDescr], [model], [eTool], [HistoryHierDF(pDescr.getPortfolioID())])



if __name__ == "__main__":
    os.chdir("..")
    os.chdir("..")
    print(os.getcwd())

    BatchDefSTFuzzyDHondtDirectOptimize.generateAllBatches()