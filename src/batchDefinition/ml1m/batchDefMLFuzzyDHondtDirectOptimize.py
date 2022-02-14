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

from batchDefinition.inputRecomMLDefinition import InputRecomMLDefinition #class

from aggregation.operators.aDHondtSelector import ADHondtSelector #class
from aggregation.operators.rouletteWheelSelector import RouletteWheelSelector #class
from aggregation.operators.theMostVotedItemSelector import TheMostVotedItemSelector #class

from batchDefinition.inputABatchDefinition import InputABatchDefinition
from batchDefinition.aBatchDefinitionML import ABatchDefinitionML #class

from batchDefinition.inputSimulatorDefinition import InputSimulatorDefinition #class

from simulator.simulator import Simulator #class

from history.historyHierDF import HistoryHierDF #class

from portfolioModel.pModelBandit import PModelBandit #class
from portfolioModel.pModelDHondtBanditsVotes import PModelDHondtBanditsVotes #class
from portfolioModel.pModelDHondt import PModelDHondt #class


class BatchDefMLFuzzyDHondtDirectOptimize(ABatchDefinitionML):

    SLCTR_ROULETTE1:str = "Roulette1"
    SLCTR_ROULETTE2:str = "Roulette2"
    SLCTR_ROULETTE3:str = "Roulette3"
    SLCTR_ROULETTE4:str = "Roulette4"
    SLCTR_ROULETTE5:str = "Roulette5"
    SLCTR_FIXED:str = "Fixed"

    lrClicks: List[float] = [0.2, 0.1, 0.03, 0.005]
    lrViewDivisors: List[float] = [250, 500, 1000]
    selectorIds: List[str] = [SLCTR_ROULETTE1, SLCTR_ROULETTE2, SLCTR_ROULETTE3, SLCTR_ROULETTE4, SLCTR_ROULETTE5, SLCTR_FIXED]

    def getBatchName(self):
        return "FDHondtDirectOptimize"

    def getSelectorParameters(self):

        selectorRoulette1:ADHondtSelector = RouletteWheelSelector({RouletteWheelSelector.ARG_EXPONENT:1})
        selectorRoulette2:ADHondtSelector = RouletteWheelSelector({RouletteWheelSelector.ARG_EXPONENT:2})
        selectorRoulette3:ADHondtSelector = RouletteWheelSelector({RouletteWheelSelector.ARG_EXPONENT:3})
        selectorRoulette4:ADHondtSelector = RouletteWheelSelector({RouletteWheelSelector.ARG_EXPONENT:4})
        selectorRoulette5:ADHondtSelector = RouletteWheelSelector({RouletteWheelSelector.ARG_EXPONENT:5})
        selectorFixed:ADHondtSelector = TheMostVotedItemSelector({})

        aDict:Dict[str,object] = {}
        aDict[BatchDefMLFuzzyDHondtDirectOptimize.SLCTR_ROULETTE1] = selectorRoulette1
        aDict[BatchDefMLFuzzyDHondtDirectOptimize.SLCTR_ROULETTE2] = selectorRoulette2
        aDict[BatchDefMLFuzzyDHondtDirectOptimize.SLCTR_ROULETTE3] = selectorRoulette3
        aDict[BatchDefMLFuzzyDHondtDirectOptimize.SLCTR_ROULETTE4] = selectorRoulette4
        aDict[BatchDefMLFuzzyDHondtDirectOptimize.SLCTR_ROULETTE5] = selectorRoulette5
        aDict[BatchDefMLFuzzyDHondtDirectOptimize.SLCTR_FIXED] = selectorFixed

        aSubDict:Dict[str,object] = {selIdI: aDict[selIdI] for selIdI in aDict.keys() if selIdI in self.selectorIds}
        return aSubDict

    def getParameters(self):
        selectorIDs:List[str] = self.getSelectorParameters().keys()
        aDict:dict = {}
        for selectorIDI in selectorIDs:
            for lrClickJ in self.lrClicks:
                for lrViewDivisorK in self.lrViewDivisors:
                    keyIJ:str = selectorIDI + "Clk" + str(lrClickJ).replace(".", "") + "ViewDivisor" + str(lrViewDivisorK).replace(".", "")
                    lrViewIJK:float = lrClickJ / lrViewDivisorK
                    eToolIJK:AEvalTool = EvalToolDHondt({EvalToolDHondt.ARG_LEARNING_RATE_CLICKS: lrClickJ,
                                                      EvalToolDHondt.ARG_LEARNING_RATE_VIEWS: lrViewIJK})
                    selectorIJK:ADHondtSelector = self.getSelectorParameters()[selectorIDI]
                    aDict[keyIJ] = (selectorIJK, eToolIJK)
        return aDict


    def run(self, batchID:str, jobID:str):

        divisionDatasetPercentualSize:int
        uBehaviour:str
        repetition:int
        divisionDatasetPercentualSize, uBehaviour, repetition = InputABatchDefinition().getBatchParameters(self.datasetID)[batchID]

        #eTool:AEvalTool
        selector, eTool = self.getParameters()[jobID]

        rIDs, rDescs = InputRecomMLDefinition.exportPairOfRecomIdsAndRecomDescrs()

        aDescDHont:AggregationDescription = InputAggrDefinition.exportADescDHondtDirectOptimize(selector)

        pDescr:Portfolio1AggrDescription = Portfolio1AggrDescription(
            self.getBatchName() + jobID, rIDs, rDescs, aDescDHont)

        model:DataFrame = PModelDHondt(pDescr.getRecommendersIDs())

        simulator:Simulator = InputSimulatorDefinition.exportSimulatorML1M(
                batchID, divisionDatasetPercentualSize, uBehaviour, repetition)
        simulator.simulate([pDescr], [model], [eTool], [HistoryHierDF(pDescr.getPortfolioID())])



if __name__ == "__main__":
    os.chdir("..")
    os.chdir("..")
    print(os.getcwd())

    BatchDefMLFuzzyDHondtDirectOptimize().generateAllBatches(InputABatchDefinition())