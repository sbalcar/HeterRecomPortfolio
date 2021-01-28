#!/usr/bin/python3

import os

from typing import List
from typing import Dict #class

from pandas.core.frame import DataFrame #class

from portfolioDescription.portfolio1AggrDescription import Portfolio1AggrDescription #class

from evaluationTool.aEvalTool import AEvalTool #class
from evaluationTool.evalToolDHondt import EvalToolDHondt #class

from aggregationDescription.aggregationDescription import AggregationDescription #class

from input.inputAggrDefinition import InputAggrDefinition  # class
from input.modelDefinition import ModelDefinition

from input.inputRecomSTDefinition import InputRecomSTDefinition #class

from aggregation.operators.aDHondtSelector import ADHondtSelector #class
from aggregation.operators.rouletteWheelSelector import RouletteWheelSelector #class
from aggregation.operators.theMostVotedItemSelector import TheMostVotedItemSelector #class

from input.aBatch import BatchParameters #class
from input.aBatchML import ABatchML #class

from input.inputSimulatorDefinition import InputSimulatorDefinition #class

from simulator.simulator import Simulator #class

from history.historyHierDF import HistoryHierDF #class

from input.batchesML1m.batchMLFuzzyDHondt import BatchMLFuzzyDHondt #class


class BatchMLWeightedAVG(ABatchML):

    lrClicks:List[float] = [0.2, 0.1, 0.02, 0.005]
    lrViewDivisors:List[float] = [200, 500, 1000]

    @staticmethod
    def getParameters():
        #lrClicks:List[float] = [0.2, 0.1, 0.02, 0.005]
        #lrClicks:List[float] = [0.03]
        #lrViewDivisors:List[float] = [200, 500, 1000]
        #lrViewDivisors:List[float] = [250]

        aDict:Dict[str,object] = {}
        for lrClickI in BatchMLWeightedAVG.lrClicks:
            for lrViewDivisorJ in BatchMLWeightedAVG.lrViewDivisors:
                keyIJ:str = "Clk" + str(lrClickI).replace(".", "") + "ViewDivisor" + str(lrViewDivisorJ).replace(".", "")
                lrViewIJ:float = lrClickI / lrViewDivisorJ
                eToolIJ:AEvalTool = EvalToolDHondt({EvalToolDHondt.ARG_LEARNING_RATE_CLICKS: lrClickI,
                                                  EvalToolDHondt.ARG_LEARNING_RATE_VIEWS: lrViewIJ})
                aDict[keyIJ] = eToolIJ
        return aDict



    def run(self, batchID:str, jobID:str):

        divisionDatasetPercentualSize:int
        uBehaviour:str
        repetition:int
        divisionDatasetPercentualSize, uBehaviour, repetition = BatchParameters.getBatchParameters(self.datasetID)[batchID]

        eTool:AEvalTool = self.getParameters()[jobID]

        rIDs, rDescs = InputRecomSTDefinition.exportPairOfRecomIdsAndRecomDescrs()

        aDescWeightedAVG:AggregationDescription = InputAggrDefinition.exportADescWeightedAVG()

        pDescr:Portfolio1AggrDescription = Portfolio1AggrDescription(
            "WAVG" + jobID, rIDs, rDescs, aDescWeightedAVG)

        model:DataFrame = ModelDefinition.createDHontModel(pDescr.getRecommendersIDs())

        simulator:Simulator = InputSimulatorDefinition.exportSimulatorSlantour(
                batchID, divisionDatasetPercentualSize, uBehaviour, repetition)
        simulator.simulate([pDescr], [model], [eTool], [HistoryHierDF(pDescr.getPortfolioID())])




if __name__ == "__main__":
    os.chdir("..")
    os.chdir("..")
    print(os.getcwd())
    BatchMLWeightedAVG.generateBatches()