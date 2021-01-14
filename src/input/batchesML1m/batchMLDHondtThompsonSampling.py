#!/usr/bin/python3

import os

from typing import List

from pandas.core.frame import DataFrame #class

from portfolioDescription.portfolio1AggrDescription import Portfolio1AggrDescription #class

from evaluationTool.aEvalTool import AEvalTool #class
from evaluationTool.evalToolDHondtBanditVotes import EvalToolDHondtBanditVotes #class

from aggregationDescription.aggregationDescription import AggregationDescription #class

from input.inputAggrDefinition import InputAggrDefinition  # class
from input.modelDefinition import ModelDefinition

from input.inputRecomMLDefinition import InputRecomMLDefinition #class

from input.batchesML1m.batchMLFuzzyDHondt import BatchMLFuzzyDHondt #class

from aggregation.operators.aDHondtSelector import ADHondtSelector #class

from input.aBatch import BatchParameters #class
from input.aBatchML import ABatchML #class
from input.inputSimulatorDefinition import InputSimulatorDefinition #class

from simulator.simulator import Simulator #class

from history.historyHierDF import HistoryHierDF #class



class BatchMLDHondtThompsonSampling(ABatchML):

    @staticmethod
    def getParameters():
        selectorIDs:List[str] = BatchMLFuzzyDHondt().getSelectorParameters().keys()

        aDict:dict = {}
        for selectorIDI in selectorIDs:
            keyIJ:str = str(selectorIDI)
            eTool:AEvalTool = EvalToolDHondtBanditVotes({})
            selectorIJK:ADHondtSelector = BatchMLFuzzyDHondt().getSelectorParameters()[selectorIDI]
            aDict[keyIJ] = (selectorIJK, eTool)
        return aDict


    def run(self, batchID:str, jobID:str):

        divisionDatasetPercentualSize:int
        uBehaviour:str
        repetition:int
        divisionDatasetPercentualSize, uBehaviour, repetition = BatchParameters.getBatchParameters(self.datasetID)[batchID]


        #eTool:AEvalTool
        selector, eTool = self.getParameters()[jobID]

        rIDs, rDescs = InputRecomMLDefinition.exportPairOfRecomIdsAndRecomDescrs()

        aDescDHontThompsonSamplingI:AggregationDescription = InputAggrDefinition.exportADescDHondtThompsonSampling(selector)

        pDescr:Portfolio1AggrDescription = Portfolio1AggrDescription(
            "DHondtThompsonSampling" + jobID, rIDs, rDescs, aDescDHontThompsonSamplingI)

        model:DataFrame = ModelDefinition.createDHondtBanditsVotesModel(pDescr.getRecommendersIDs())

        simulator:Simulator = InputSimulatorDefinition.exportSimulatorML1M(
                batchID, divisionDatasetPercentualSize, uBehaviour, repetition)
        simulator.simulate([pDescr], [model], [eTool], HistoryHierDF)




if __name__ == "__main__":
    os.chdir("..")
    os.chdir("..")
    print(os.getcwd())
    BatchMLDHondtThompsonSampling.generateBatches()
