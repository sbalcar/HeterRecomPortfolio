#!/usr/bin/python3

import os

from typing import List
from typing import Dict #class

from pandas.core.frame import DataFrame #class

from portfolioDescription.portfolio1AggrDescription import Portfolio1AggrDescription #class

from evaluationTool.aEvalTool import AEvalTool #class
from evaluationTool.evalToolDHondt import EvalToolDHondt #class

from aggregationDescription.aggregationDescription import AggregationDescription #class

from batchDefinition.inputAggrDefinition import InputAggrDefinition  # class

from batchDefinition.inputRecomMLDefinition import InputRecomMLDefinition #class

from aggregation.operators.aDHondtSelector import ADHondtSelector #class
from aggregation.operators.rouletteWheelSelector import RouletteWheelSelector #class
from aggregation.operators.theMostVotedItemSelector import TheMostVotedItemSelector #class

from batchDefinition.inputABatchDefinition import InputABatchDefinition
from batchDefinition.aBatchDefinitionML import ABatchDefinitionML #class

from batchDefinition.inputSimulatorDefinition import InputSimulatorDefinition #class

from simulator.simulator import Simulator #class

from history.historyHierDF import HistoryHierDF #class

from batchDefinition.ml1m.batchDefMLFuzzyDHondt import BatchDefMLFuzzyDHondt #class

from portfolioModel.pModelBandit import PModelBandit #class
from portfolioModel.pModelDHondtBanditsVotes import PModelDHondtBanditsVotes #class
from portfolioModel.pModelDHondt import PModelDHondt #class


class BatchDefMLWeightedAVG(ABatchDefinitionML):

    lrClicks:List[float] = [0.2, 0.1, 0.03, 0.005]
    lrViewDivisors:List[float] = [250, 500, 1000]

    def getBatchName(self):
        return "WAVG"

    def getParameters(self):

        aDict:Dict[str,object] = {}
        for lrClickI in self.lrClicks:
            for lrViewDivisorJ in self.lrViewDivisors:
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
        divisionDatasetPercentualSize, uBehaviour, repetition = InputABatchDefinition().getBatchParameters(self.datasetID)[batchID]

        eTool:AEvalTool = self.getParameters()[jobID]

        rIDs, rDescs = InputRecomMLDefinition.exportPairOfRecomIdsAndRecomDescrs()

        aDescWeightedAVG:AggregationDescription = InputAggrDefinition.exportADescWeightedAVG()

        pDescr:Portfolio1AggrDescription = Portfolio1AggrDescription(
            "WAVG" + jobID, rIDs, rDescs, aDescWeightedAVG)

        model:DataFrame = PModelDHondt(pDescr.getRecommendersIDs())

        simulator:Simulator = InputSimulatorDefinition().exportSimulatorML1M(
                batchID, divisionDatasetPercentualSize, uBehaviour, repetition)
        simulator.simulate([pDescr], [model], [eTool], [HistoryHierDF(pDescr.getPortfolioID())])




if __name__ == "__main__":
    os.chdir("..")
    os.chdir("..")
    print(os.getcwd())

    BatchDefMLWeightedAVG.generateAllBatches(InputABatchDefinition())