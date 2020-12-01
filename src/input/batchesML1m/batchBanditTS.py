#!/usr/bin/python3

import os
from typing import List

from pandas.core.frame import DataFrame #class

from portfolioDescription.portfolio1AggrDescription import Portfolio1AggrDescription #class

from evaluationTool.aEvalTool import AEvalTool #class
from evaluationTool.evalToolBanditTS import EvalToolBanditTS #class

from input.inputAggrDefinition import InputAggrDefinition, ModelDefinition  # class

from input.InputRecomDefinition import InputRecomDefinition #class

from input.batchesML1m.aML1MConfig import AML1MConf #function
from input.batchesML1m.batchFuzzyDHondt import BatchFuzzyDHondt #class

from input.aBatch import ABatch #class

from aggregation.aggrFuzzyDHondt import AggrFuzzyDHondt #class
from aggregation.operators.aDHondtSelector import ADHondtSelector #class



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

        aConf:AML1MConf = AML1MConf(batchID, divisionDatasetPercentualSize, uBehaviour, repetition)

        rIDs, rDescs = InputRecomDefinition.exportPairOfRecomIdsAndRecomDescrs(aConf.datasetID)

        pDescr: Portfolio1AggrDescription = Portfolio1AggrDescription(
            "BanditTS" + jobID, rIDs, rDescs, InputAggrDefinition.exportADescBanditTS(selector))

        evalTool:AEvalTool = EvalToolBanditTS({})
        model:DataFrame = ModelDefinition.createBanditModel(pDescr.getRecommendersIDs())

        aConf.run(pDescr, model, evalTool)



if __name__ == "__main__":
    os.chdir("..")
    os.chdir("..")
    print(os.getcwd())
    #os.chdir("batches")
    BatchBanditTS.generateBatches()