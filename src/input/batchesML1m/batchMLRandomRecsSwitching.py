#!/usr/bin/python3

import os
from typing import List #class
from typing import Dict #class

from pandas.core.frame import DataFrame #class

from portfolioDescription.portfolio1AggrDescription import Portfolio1AggrDescription #class

from evaluationTool.aEvalTool import AEvalTool #class
from evaluationTool.evalToolDoNothing import EToolDoNothing #class

from input.inputAggrDefinition import InputAggrDefinition  # class
from input.modelDefinition import ModelDefinition

from input.inputRecomMLDefinition import InputRecomMLDefinition #class

from input.batchesML1m.batchMLFuzzyDHondt import BatchMLFuzzyDHondt #class

from input.inputABatchDefinition import InputABatchDefinition
from input.aBatchML import ABatchML #class

from aggregation.aggrFuzzyDHondt import AggrFuzzyDHondt #class
from aggregation.operators.aDHondtSelector import ADHondtSelector #class
from input.inputSimulatorDefinition import InputSimulatorDefinition #class

from simulator.simulator import Simulator #class

from history.historyHierDF import HistoryHierDF #class

from aggregationDescription.aggregationDescription import AggregationDescription #class


class BatchMLRandomRecsSwitching(ABatchML):

    @staticmethod
    def getParameters():
        return {"": ""}


    def run(self, batchID:str, jobID:str):

        divisionDatasetPercentualSize:int
        uBehaviour:str
        repetition:int
        divisionDatasetPercentualSize, uBehaviour, repetition = InputABatchDefinition.getBatchParameters(self.datasetID)[batchID]

        eTool:AEvalTool = EToolDoNothing({})

        rIDs, rDescs = InputRecomMLDefinition.exportPairOfRecomIdsAndRecomDescrs()

        mainMethodID:str = "RecomKnn"  # the best method for ML
        aDescRandomRecsSwitching:AggregationDescription = InputAggrDefinition.exportADescRandomRecsSwitching(mainMethodID)

        pDescr:Portfolio1AggrDescription = Portfolio1AggrDescription(
            "RandomRecsSwitching" + jobID, rIDs, rDescs, aDescRandomRecsSwitching)

        model:DataFrame = DataFrame()

        simulator:Simulator = InputSimulatorDefinition.exportSimulatorML1M(
                batchID, divisionDatasetPercentualSize, uBehaviour, repetition)
        simulator.simulate([pDescr], [model], [eTool], [HistoryHierDF(pDescr.getPortfolioID())])



if __name__ == "__main__":
    os.chdir("..")
    os.chdir("..")
    print(os.getcwd())

    BatchMLRandomRecsSwitching.generateAllBatches()