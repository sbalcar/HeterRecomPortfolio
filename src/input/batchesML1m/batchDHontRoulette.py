#!/usr/bin/python3

import os

from typing import List

from pandas.core.frame import DataFrame #class

from portfolioDescription.portfolio1AggrDescription import Portfolio1AggrDescription #class

from evaluationTool.aEvalTool import AEvalTool #class
from evaluationTool.evalToolDHont import EvalToolDHont #class

from aggregationDescription.aggregationDescription import AggregationDescription #class

from input.inputAggrDefinition import InputAggrDefinition, ModelDefinition  # class

from input.InputRecomDefinition import InputRecomDefinition #class

from input.batchesML1m.aML1MConfig import AML1MConf #function

from datasets.behaviours import Behaviours #class


class BatchDHontRoulette:

    @staticmethod
    def getParameters():
        rouletteExps:List[int] = [1, 3]
        lrClicks:List[float] = [0.2, 0.1, 0.02, 0.005]
        lrViews:List[float] = [0.1 / 500, 0.1 / 200, 0.1 / 1000]

        aDict:dict = {}
        for rouletteExpI in rouletteExps:
            for lrClickJ in lrClicks:
                for lrViewK in lrViews:
                    keyIJ:str = str(rouletteExpI) + "Clk" + str(lrClickJ).replace(".", "") + "View" + str(lrViewK).replace(".", "")
                    eTool:AEvalTool = EvalToolDHont({EvalToolDHont.ARG_LEARNING_RATE_CLICKS: lrClickJ,
                                                 EvalToolDHont.ARG_LEARNING_RATE_VIEWS: lrViewK})
                    aDict[keyIJ] = (rouletteExpI, eTool)
        return aDict

    @staticmethod
    def run(batchID:str, divisionDatasetPercentualSize:int, uBehaviour:str, repetition:int, jobID:str):

        #eTool:AEvalTool
        rouletteExp, eTool = BatchDHontRoulette.getParameters()[jobID]

        aConf:AML1MConf = AML1MConf(batchID, divisionDatasetPercentualSize, uBehaviour, repetition)

        rIDs, rDescs = InputRecomDefinition.exportPairOfRecomIdsAndRecomDescrs(aConf.datasetID)

        aDescDHontRouletteI:AggregationDescription = None
        if rouletteExp == 1:
            aDescDHontRouletteI = InputAggrDefinition.exportADescDHontRoulette()
        elif rouletteExp == 3:
            aDescDHontRouletteI = InputAggrDefinition.exportADescDHontRoulette3()

        pDescr:Portfolio1AggrDescription = Portfolio1AggrDescription(
            "DHontRoulette" + jobID, rIDs, rDescs, aDescDHontRouletteI)

        model:DataFrame = ModelDefinition.createDHontModel(pDescr.getRecommendersIDs())

        aConf.run(pDescr, model, eTool)


    @staticmethod
    def generateBatches():

        divisionsDatasetPercentualSize:List[int] = [90]
        uBehaviours:List[str] = [Behaviours.BHVR_LINEAR0109, Behaviours.BHVR_STATIC08,
                                 Behaviours.BHVR_STATIC06, Behaviours.BHVR_STATIC04,
                                 Behaviours.BHVR_STATIC02]
        repetitions:List[int] = [1, 2, 3, 5]

        jobIDs:List[str] = list(BatchDHontRoulette.getParameters().keys())

        for divisionDatasetPercentualSizeI in divisionsDatasetPercentualSize:
            for uBehaviourJ in uBehaviours:
                for repetitionK in repetitions:
                    for jobIDL in jobIDs:

                        batchID:str = "ml1mDiv" + str(divisionDatasetPercentualSizeI) + "U" + uBehaviourJ + "R" + str(repetitionK)
                        batchesDir:str = ".." + os.sep + "batches" + os.sep + batchID
                        if not os.path.exists(batchesDir):
                            os.mkdir(batchesDir)

                        job:str = str(BatchDHontRoulette.__name__) + jobIDL
                        text:str = str(BatchDHontRoulette.__name__) + ".run('"\
                                   + str(batchID) + "', "\
                                   + str(divisionDatasetPercentualSizeI) + ", '"\
                                   + str(uBehaviourJ) + "', "\
                                   + str(repetitionK) + ", "\
                                   + "'" + str(jobIDL) + "'" + ")"

                        jobFile:str = batchesDir + os.sep + job + ".txt"
                        BatchDHontRoulette.__writeToFile(jobFile, text)


    @staticmethod
    def __writeToFile(fileName:str, text:str):
        f = open(fileName, "w")
        f.write(text)
        f.close()


if __name__ == "__main__":
    os.chdir("..")
    os.chdir("..")
    print(os.getcwd())
    BatchDHontRoulette.generateBatches()