#!/usr/bin/python3

import os

from typing import List

from pandas.core.frame import DataFrame #class

from portfolioDescription.portfolio1AggrDescription import Portfolio1AggrDescription #class

from evaluationTool.aEvalTool import AEvalTool #class
from evaluationTool.evalToolDHondt import EvalToolDHondt #class

from aggregationDescription.aggregationDescription import AggregationDescription #class

from input.inputAggrDefinition import InputAggrDefinition, ModelDefinition  # class

from input.inputRecomDefinition import InputRecomDefinition #class

from aggregation.operators.aDHondtSelector import ADHondtSelector #class
from aggregation.operators.rouletteWheelSelector import RouletteWheelSelector #class
from aggregation.operators.theMostVotedItemSelector import TheMostVotedItemSelector #class

from input.aBatch import BatchParameters #class
from input.aBatchML import ABatchML #class

from input.inputSimulatorDefinition import InputSimulatorDefinition #class

from simulator.simulator import Simulator #class

from history.historyHierDF import HistoryHierDF #class



class BatchMLFuzzyDHondtDirectOptimize(ABatchML):

    SLCTR_ROULETTE1:str = "Roulette1"
    SLCTR_ROULETTE2:str = "Roulette3"
    SLCTR_FIXED:str = "Fixed"

    @staticmethod
    def getSelectorParameters():

        selectorRoulette1:ADHondtSelector = RouletteWheelSelector({RouletteWheelSelector.ARG_EXPONENT:1})
        selectorRoulette3:ADHondtSelector = RouletteWheelSelector({RouletteWheelSelector.ARG_EXPONENT:3})
        selectorFixed:ADHondtSelector = TheMostVotedItemSelector({})

        aDict:dict = {}
        aDict[BatchMLFuzzyDHondtDirectOptimize.SLCTR_ROULETTE1] = selectorRoulette1
        aDict[BatchMLFuzzyDHondtDirectOptimize.SLCTR_ROULETTE2] = selectorRoulette3
        aDict[BatchMLFuzzyDHondtDirectOptimize.SLCTR_FIXED] = selectorFixed
        return aDict

    @staticmethod
    def getParameters():
        selectorIDs:List[str] = BatchMLFuzzyDHondtDirectOptimize.getSelectorParameters().keys()
        lrClicks:List[float] = [0.2, 0.1, 0.02, 0.005]
        #lrClicks:List[float] = [0.1]
        lrViewDivisors:List[float] = [200, 500, 1000]
        #lrViewDivisors:List[float] = [500]

        aDict:dict = {}
        for selectorIDI in selectorIDs:
            for lrClickJ in lrClicks:
                for lrViewDivisorK in lrViewDivisors:
                    keyIJ:str = selectorIDI + "Clk" + str(lrClickJ).replace(".", "") + "ViewDivisor" + str(lrViewDivisorK).replace(".", "")
                    lrViewIJK:float = lrClickJ / lrViewDivisorK
                    eToolIJK:AEvalTool = EvalToolDHondt({EvalToolDHondt.ARG_LEARNING_RATE_CLICKS: lrClickJ,
                                                      EvalToolDHondt.ARG_LEARNING_RATE_VIEWS: lrViewIJK})
                    selectorIJK:ADHondtSelector = BatchMLFuzzyDHondtDirectOptimize.getSelectorParameters()[selectorIDI]
                    aDict[keyIJ] = (selectorIJK, eToolIJK)
        return aDict


    def run(self, batchID:str, jobID:str):

        divisionDatasetPercentualSize:int
        uBehaviour:str
        repetition:int
        divisionDatasetPercentualSize, uBehaviour, repetition = BatchParameters.getBatchParameters(self.datasetID)[batchID]

        #eTool:AEvalTool
        selector, eTool = self.getParameters()[jobID]

        datasetID:str = "ml1m" + "Div" + str(divisionDatasetPercentualSize)

        rIDs, rDescs = InputRecomDefinition.exportPairOfRecomIdsAndRecomDescrsML()

        aDescDHont:AggregationDescription = InputAggrDefinition.exportADescDHontDirectOptimize(selector)

        pDescr:Portfolio1AggrDescription = Portfolio1AggrDescription(
            "FDHontDirectOptimize" + jobID, rIDs, rDescs, aDescDHont)

        model:DataFrame = ModelDefinition.createDHontModel(pDescr.getRecommendersIDs())

        simulator:Simulator = InputSimulatorDefinition.exportSimulatorML1M(
                batchID, divisionDatasetPercentualSize, uBehaviour, repetition)
        simulator.simulate([pDescr], [model], [eTool], HistoryHierDF)



if __name__ == "__main__":
    os.chdir("..")
    os.chdir("..")
    print(os.getcwd())
    BatchMLFuzzyDHondtDirectOptimize.generateBatches()