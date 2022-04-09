#!/usr/bin/python3

import os

from typing import List

from pandas.core.frame import DataFrame #class

from portfolioDescription.portfolio1AggrDescription import Portfolio1AggrDescription #class

from evaluationTool.aEvalTool import AEvalTool #class
from evaluationTool.evalToolDHondtBanditVotes import EvalToolDHondtBanditVotes #class

from aggregationDescription.aggregationDescription import AggregationDescription #class

from batchDefinition.inputAggrDefinition import InputAggrDefinition  # class

from batchDefinition.inputRecomMLDefinition import InputRecomMLDefinition #class

from batchDefinition.ml1m.batchDefMLFuzzyDHondt import BatchDefMLFuzzyDHondt #class

from aggregation.operators.aDHondtSelector import ADHondtSelector #class

from batchDefinition.inputABatchDefinition import InputABatchDefinition
from batchDefinition.aBatchDefinitionML import ABatchDefinitionML #class
from batchDefinition.inputSimulatorDefinition import InputSimulatorDefinition #class

from simulator.simulator import Simulator #class

from history.historyHierDF import HistoryHierDF #class

from portfolioModel.pModelBandit import PModelBandit #class
from portfolioModel.pModelDHondtBanditsVotes import PModelDHondtBanditsVotes #class
from portfolioModel.pModelDHondt import PModelDHondt #class


class BatchDefMLFuzzyDHondtThompsonSampling(ABatchDefinitionML):

    selectorIDs:List[str] = [BatchDefMLFuzzyDHondt.SLCTR_FIXED]

    def getBatchName(self):
        return "FuzzyDHondtThompsonSampling"
    
    def getParameters(self):

        aDict:dict = {}
        for selectorIDI in self.selectorIDs:
            keyIJ:str = str(selectorIDI)
            eTool:AEvalTool = EvalToolDHondtBanditVotes({})
            selectorIJK:ADHondtSelector = BatchDefMLFuzzyDHondt().getSelectorParameters()[selectorIDI]
            aDict[keyIJ] = (selectorIJK, eTool)
        return aDict


    def run(self, batchID:str, jobID:str):

        divisionDatasetPercentualSize:int
        uBehaviour:str
        repetition:int
        divisionDatasetPercentualSize, uBehaviour, repetition = InputABatchDefinition().getBatchParameters(self.datasetID)[batchID]


        #eTool:AEvalTool
        selector, eTool = self.getParameters()[jobID]

        rIDs, rDescs = InputRecomMLDefinition.exportPairOfRecomIdsAndRecomDescrs()

        aDescDHontThompsonSamplingI:AggregationDescription = InputAggrDefinition.exportADescDHondtThompsonSampling(selector)

        pDescr:Portfolio1AggrDescription = Portfolio1AggrDescription(
            self.getBatchName() + jobID, rIDs, rDescs, aDescDHontThompsonSamplingI)

        model:DataFrame = PModelDHondtBanditsVotes(pDescr.getRecommendersIDs())

        simulator:Simulator = InputSimulatorDefinition().exportSimulatorML1M(
                batchID, divisionDatasetPercentualSize, uBehaviour, repetition)
        simulator.simulate([pDescr], [model], [eTool], [HistoryHierDF(pDescr.getPortfolioID())])




if __name__ == "__main__":
    os.chdir("..")
    os.chdir("..")
    print(os.getcwd())
    BatchDefMLFuzzyDHondtThompsonSampling.generateAllBatches(InputABatchDefinition())
