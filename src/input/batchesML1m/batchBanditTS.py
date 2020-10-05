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

from datasets.behaviours import Behaviours #class


class BatchBanditTS:

    @staticmethod
    def run(batchID:str, divisionDatasetPercentualSize:int, uBehaviour:str, repetition:int):
        aConf:AML1MConf = AML1MConf(batchID, divisionDatasetPercentualSize, uBehaviour, repetition)

        rIDs, rDescs = InputRecomDefinition.exportPairOfRecomIdsAndRecomDescrs(aConf.datasetID)

        pDescr: Portfolio1AggrDescription = Portfolio1AggrDescription(
            "BanditTS", rIDs, rDescs, InputAggrDefinition.exportADescBanditTS())

        evalTool:AEvalTool = EvalToolBanditTS({})
        model:DataFrame = ModelDefinition.createBanditModel(pDescr.getRecommendersIDs())

        aConf.run(pDescr, model, evalTool)


    @staticmethod
    def generateBatches():

        divisionsDatasetPercentualSize:List[int] = [90]
        uBehaviours:List[str] = [Behaviours.BHVR_LINEAR0109, Behaviours.BHVR_STATIC08,
                                 Behaviours.BHVR_STATIC06, Behaviours.BHVR_STATIC04,
                                 Behaviours.BHVR_STATIC02]
        repetitions:List[int] = [1, 2, 3, 5]

        for divisionDatasetPercentualSizeI in divisionsDatasetPercentualSize:
            for uBehaviourJ in uBehaviours:
                for repetitionK in repetitions:

                        batchID:str = "ml1mDiv" + str(divisionDatasetPercentualSizeI) + "U" + uBehaviourJ + "R" + str(repetitionK)
                        batchesDir:str = ".." + os.sep + "batches" + os.sep + batchID
                        if not os.path.exists(batchesDir):
                            os.mkdir(batchesDir)

                        job:str = str(BatchBanditTS.__name__)
                        text:str = str(BatchBanditTS.__name__) + ".run('"\
                                   + str(batchID) + "', "\
                                   + str(divisionDatasetPercentualSizeI) + ", '"\
                                   + str(uBehaviourJ) + "', "\
                                   + str(repetitionK) + ")"

                        jobFile:str = batchesDir + os.sep + job + ".txt"
                        BatchBanditTS.__writeToFile(jobFile, text)

    @staticmethod
    def __writeToFile(fileName:str, text:str):
        f = open(fileName, "w")
        f.write(text)
        f.close()



if __name__ == "__main__":
    os.chdir("..")
    os.chdir("..")
    print(os.getcwd())
    #os.chdir("batches")
    BatchBanditTS.generateBatches()