#!/usr/bin/python3

import os

from typing import List

from pandas.core.frame import DataFrame #class

from portfolioDescription.portfolio1AggrDescription import Portfolio1AggrDescription #class

from evaluationTool.aEvalTool import AEvalTool #class
from evaluationTool.evalToolDHondtBanditVotes import EvalToolDHondtBanditVotes #class

from aggregationDescription.aggregationDescription import AggregationDescription #class

from input.inputAggrDefinition import InputAggrDefinition, ModelDefinition  # class

from input.inputRecomDefinition import InputRecomDefinition #class

from input.batchesML1m.batchFuzzyDHondt import BatchFuzzyDHondt #class

from aggregation.operators.aDHondtSelector import ADHondtSelector #class
from input.aBatch import ABatch #class
from input.inputSimulatorDefinition import InputSimulatorDefinition #class

from simulator.simulator import Simulator #class

from history.historyHierDF import HistoryHierDF #class



class BatchDHondtThompsonSampling(ABatch):


    def getParameters(self):
        selectorIDs:List[str] = BatchFuzzyDHondt().getSelectorParameters().keys()

        aDict:dict = {}
        for selectorIDI in selectorIDs:
            keyIJ:str = str(selectorIDI)
            eTool:AEvalTool = EvalToolDHondtBanditVotes({})
            selectorIJK:ADHondtSelector = BatchFuzzyDHondt().getSelectorParameters()[selectorIDI]
            aDict[keyIJ] = (selectorIJK, eTool)
        return aDict


    def run(self, batchID:str, jobID:str):

        from execute.generateBatches import BatchParameters #class
        divisionDatasetPercentualSize:int
        uBehaviour:str
        repetition:int
        divisionDatasetPercentualSize, uBehaviour, repetition = BatchParameters.getBatchParameters()[batchID]


        #eTool:AEvalTool
        selector, eTool = self.getParameters()[jobID]

        datasetID:str = "ml1m" + "Div" + str(divisionDatasetPercentualSize)

        rIDs, rDescs = InputRecomDefinition.exportPairOfRecomIdsAndRecomDescrs(datasetID)

        aDescDHontThompsonSamplingI:AggregationDescription = InputAggrDefinition.exportADescDHontThompsonSampling(selector)

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
    BatchDHondtThompsonSampling.generateBatches()
