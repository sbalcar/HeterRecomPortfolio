#!/usr/bin/python3

import os
from typing import List #class
from typing import Dict #class

from pandas.core.frame import DataFrame #class

from portfolioDescription.portfolio1AggrDescription import Portfolio1AggrDescription #class

from evaluationTool.aEvalTool import AEvalTool #class
from evaluationTool.evalToolBanditTS import EvalToolBanditTS #class

from input.inputAggrDefinition import InputAggrDefinition  # class
from input.modelDefinition import ModelDefinition

from input.inputRecomMLDefinition import InputRecomMLDefinition #class

from input.batchesML1m.batchMLFuzzyDHondt import BatchMLFuzzyDHondt #class

from input.aBatch import BatchParameters #class
from input.aBatchML import ABatchML #class

from aggregation.aggrFuzzyDHondt import AggrFuzzyDHondt #class
from aggregation.operators.aDHondtSelector import ADHondtSelector #class
from input.inputSimulatorDefinition import InputSimulatorDefinition #class

from simulator.simulator import Simulator #class

from history.historyHierDF import HistoryHierDF #class



class BatchMLBanditTS(ABatchML):

    @staticmethod
    def getParameters():
        selectorIDs:List[str] = BatchMLFuzzyDHondt().getSelectorParameters().keys()

        aDict:Dict[str,ADHondtSelector] = {}
        for selectorIDI in selectorIDs:
            keyI:str = selectorIDI
            selectorI:ADHondtSelector = BatchMLFuzzyDHondt().getSelectorParameters()[selectorIDI]
            aDict[keyI] = (selectorI)
        return aDict

    def run(self, batchID:str, jobID:str):

        divisionDatasetPercentualSize:int
        uBehaviour:str
        repetition:int
        divisionDatasetPercentualSize, uBehaviour, repetition = BatchParameters.getBatchParameters(self.datasetID)[batchID]

        selector:ADHondtSelector = self.getParameters()[jobID]

        rIDs, rDescs = InputRecomMLDefinition.exportPairOfRecomIdsAndRecomDescrs()

        pDescr: Portfolio1AggrDescription = Portfolio1AggrDescription(
            "BanditTS" + jobID, rIDs, rDescs, InputAggrDefinition.exportADescBanditTS(selector))

        eTool:AEvalTool = EvalToolBanditTS({})
        model:DataFrame = ModelDefinition.createBanditModel(pDescr.getRecommendersIDs())

        simulator:Simulator = InputSimulatorDefinition.exportSimulatorML1M(
                batchID, divisionDatasetPercentualSize, uBehaviour, repetition)
        simulator.simulate([pDescr], [model], [eTool], [HistoryHierDF(pDescr.getPortfolioID())])




if __name__ == "__main__":
    os.chdir("..")
    os.chdir("..")
    print(os.getcwd())

    BatchMLBanditTS.generateBatches()