#!/usr/bin/python3

import os
from typing import List #class
from typing import Dict #class

from pandas.core.frame import DataFrame #class

from portfolioDescription.portfolio1AggrDescription import Portfolio1AggrDescription #class

from evaluationTool.aEvalTool import AEvalTool #class
from evaluationTool.evalToolBanditTS import EvalToolBanditTS #class

from batchDefinition.inputAggrDefinition import InputAggrDefinition  # class
from batchDefinition.modelDefinition import ModelDefinition

from batchDefinition.inputRecomMLDefinition import InputRecomMLDefinition #class

from batchDefinition.ml1m.batchDefMLFuzzyDHondt import BatchDefMLFuzzyDHondt #class

from batchDefinition.inputABatchDefinition import InputABatchDefinition
from batchDefinition.aBatchDefinitionML import ABatchDefinitionML #class

from aggregation.aggrFuzzyDHondt import AggrFuzzyDHondt #class
from aggregation.operators.aDHondtSelector import ADHondtSelector #class
from batchDefinition.inputSimulatorDefinition import InputSimulatorDefinition #class

from simulator.simulator import Simulator #class

from history.historyHierDF import HistoryHierDF #class

from batchDefinition.ml1m.batchDefMLFuzzyDHondt import BatchDefMLFuzzyDHondt #class

class BatchDefMLBanditTS(ABatchDefinitionML):

    SLCTR_ROULETTE1:str = BatchDefMLFuzzyDHondt.SLCTR_ROULETTE1
    SLCTR_ROULETTE3:str = BatchDefMLFuzzyDHondt.SLCTR_ROULETTE3
    SLCTR_FIXED:str = BatchDefMLFuzzyDHondt.SLCTR_FIXED

    selectorIDs:List[str] = [SLCTR_ROULETTE1, SLCTR_ROULETTE3, SLCTR_FIXED]

    def getBatchName(self):
        return "BanditTS"

    def getParameters(self):
        batchDefMLFuzzyDHondt = BatchDefMLFuzzyDHondt()
        batchDefMLFuzzyDHondt.selectorIDs = self.selectorIDs
        return batchDefMLFuzzyDHondt.getSelectorParameters()


    def run(self, batchID:str, jobID:str):

        divisionDatasetPercentualSize:int
        uBehaviour:str
        repetition:int
        divisionDatasetPercentualSize, uBehaviour, repetition = InputABatchDefinition().getBatchParameters(self.datasetID)[batchID]

        selector:ADHondtSelector = self.getParameters()[jobID]

        rIDs, rDescs = InputRecomMLDefinition.exportPairOfRecomIdsAndRecomDescrs()

        pDescr: Portfolio1AggrDescription = Portfolio1AggrDescription(
            self.getBatchName() + jobID, rIDs, rDescs, InputAggrDefinition.exportADescBanditTS(selector))

        eTool:AEvalTool = EvalToolBanditTS({})
        model:DataFrame = ModelDefinition.createBanditModel(pDescr.getRecommendersIDs())

        simulator:Simulator = InputSimulatorDefinition.exportSimulatorML1M(
                batchID, divisionDatasetPercentualSize, uBehaviour, repetition)
        simulator.simulate([pDescr], [model], [eTool], [HistoryHierDF(pDescr.getPortfolioID())])




if __name__ == "__main__":
    os.chdir("..")
    os.chdir("..")
    print(os.getcwd())

    BatchDefMLBanditTS.generateAllBatches(InputABatchDefinition())