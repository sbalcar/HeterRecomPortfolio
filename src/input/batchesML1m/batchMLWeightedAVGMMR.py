#!/usr/bin/python3

import os

from typing import List
from typing import Dict  # class

from pandas.core.frame import DataFrame  # class

from portfolioDescription.portfolio1AggrDescription import Portfolio1AggrDescription  # class

from evaluationTool.aEvalTool import AEvalTool  # class
from evaluationTool.evalToolDHondt import EvalToolDHondt  # class

from aggregationDescription.aggregationDescription import AggregationDescription  # class

from input.inputAggrDefinition import InputAggrDefinition  # class
from input.modelDefinition import ModelDefinition

from input.inputRecomMLDefinition import InputRecomMLDefinition  # class

from aggregation.operators.aDHondtSelector import ADHondtSelector  # class
from aggregation.operators.rouletteWheelSelector import RouletteWheelSelector  # class
from aggregation.operators.theMostVotedItemSelector import TheMostVotedItemSelector  # class

from input.inputABatchDefinition import InputABatchDefinition
from input.aBatchML import ABatchML  # class

from input.inputSimulatorDefinition import InputSimulatorDefinition  # class

from simulator.simulator import Simulator  # class

from history.historyHierDF import HistoryHierDF  # class

from input.batchesML1m.batchMLWeightedAVG import BatchMLWeightedAVG #class


class BatchMLWeightedAVGMMR(ABatchML):

    @classmethod
    def getParameters(cls):
        return BatchMLWeightedAVG.getParameters()

    def run(self, batchID: str, jobID: str):

        divisionDatasetPercentualSize: int
        uBehaviour: str
        repetition: int
        divisionDatasetPercentualSize, uBehaviour, repetition = \
        InputABatchDefinition.getBatchParameters(self.datasetID)[batchID]

        eTool: AEvalTool = self.getParameters()[jobID]

        rIDs, rDescs = InputRecomMLDefinition.exportPairOfRecomIdsAndRecomDescrs()

        aDescWeightedAVG:AggregationDescription = InputAggrDefinition.exportADescWeightedAVGMMR()

        pDescr: Portfolio1AggrDescription = Portfolio1AggrDescription(
            "WAVGMMR" + jobID, rIDs, rDescs, aDescWeightedAVG)

        model: DataFrame = ModelDefinition.createDHontModel(pDescr.getRecommendersIDs())

        simulator: Simulator = InputSimulatorDefinition.exportSimulatorML1M(
            batchID, divisionDatasetPercentualSize, uBehaviour, repetition)
        simulator.simulate([pDescr], [model], [eTool], [HistoryHierDF(pDescr.getPortfolioID())])


if __name__ == "__main__":
    os.chdir("..")
    os.chdir("..")
    print(os.getcwd())

    BatchMLWeightedAVG.generateAllBatches()