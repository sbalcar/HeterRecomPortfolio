#!/usr/bin/python3

import os
from typing import List

from pandas.core.frame import DataFrame #class

from portfolioDescription.portfolio1AggrDescription import Portfolio1AggrDescription #class

from evaluationTool.aEvalTool import AEvalTool #class
from evaluationTool.evalToolBanditTS import EvalToolBanditTS #class

from input.inputAggrDefinition import InputAggrDefinition, ModelDefinition  # class

from input.inputRecomDefinition import InputRecomDefinition #class

from input.batchesML1m.batchFuzzyDHondt import BatchFuzzyDHondt #class

from input.aBatch import ABatch #class

from aggregation.aggrFuzzyDHondt import AggrFuzzyDHondt #class
from aggregation.operators.aDHondtSelector import ADHondtSelector #class
from input.inputSimulatorDefinition import InputSimulatorDefinition #class

from simulator.simulator import Simulator #class

from history.historyHierDF import HistoryHierDF #class



class BatchBanditTS(ABatch):

    def getParameters(self):
        selectorIDs:List[str] = BatchFuzzyDHondt().getSelectorParameters().keys()

        aDict:dict = {}
        for selectorIDI in selectorIDs:
            keyI:str = selectorIDI
            selectorI:ADHondtSelector = BatchFuzzyDHondt().getSelectorParameters()[selectorIDI]
            aDict[keyI] = (selectorI)
        return aDict

    def run(self, batchID:str, jobID:str):

        from execute.generateBatches import BatchParameters #class
        divisionDatasetPercentualSize:int
        uBehaviour:str
        repetition:int
        divisionDatasetPercentualSize, uBehaviour, repetition = BatchParameters.getBatchParameters()[batchID]

        selector:ADHondtSelector = self.getParameters()[jobID]

        datasetID:str = "ml1m" + "Div" + str(divisionDatasetPercentualSize)

        rIDs, rDescs = InputRecomDefinition.exportPairOfRecomIdsAndRecomDescrs(datasetID)

        pDescr: Portfolio1AggrDescription = Portfolio1AggrDescription(
            "BanditTS" + jobID, rIDs, rDescs, InputAggrDefinition.exportADescBanditTS(selector))

        eTool:AEvalTool = EvalToolBanditTS({})
        model:DataFrame = ModelDefinition.createBanditModel(pDescr.getRecommendersIDs())

        simulator:Simulator = InputSimulatorDefinition.exportSimulatorML1M(
                batchID, divisionDatasetPercentualSize, uBehaviour, repetition)
        simulator.simulate([pDescr], [model], [eTool], HistoryHierDF)




if __name__ == "__main__":
    os.chdir("..")
    os.chdir("..")
    print(os.getcwd())
    #os.chdir("batches")
    BatchBanditTS.generateBatches()