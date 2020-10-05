#!/usr/bin/python3

import os

from typing import List

from pandas.core.frame import DataFrame #class

from portfolioDescription.portfolio1AggrDescription import Portfolio1AggrDescription #class

from evaluationTool.aEvalTool import AEvalTool #class
from evaluationTool.evalToolDHondt import EvalToolDHondt #class

from aggregationDescription.aggregationDescription import AggregationDescription #class

from input.inputAggrDefinition import InputAggrDefinition, ModelDefinition  # class

from input.InputRecomDefinition import InputRecomDefinition #class

from input.batchesML1m.batchNegDHondtFixed import BatchNegDHondtFixed #class

from aggregation.toolsDHontNF.penalizationOfResultsByNegImpFeedback.aPenalization import APenalization #class

from input.batchesML1m.aML1MConfig import AML1MConf #function

from datasets.behaviours import Behaviours #class


class BatchNegDHondtRoulette:

    @staticmethod
    def getParameters():
        rouletteExps:List[int] = [1, 3]
        negativeImplFeedback:List[str] = BatchNegDHondtFixed.getNegativeImplFeedbackParameters().keys()
        lrClicks:List[float] = [0.2, 0.1, 0.02, 0.005]
        lrViews:List[float] = [0.1 / 500, 0.1 / 200, 0.1 / 1000]

        aDict:dict = {}
        for rouletteExpH in rouletteExps:
            for nImplFeedbackI in negativeImplFeedback:
                for lrClickJ in lrClicks:
                    for lrViewK in lrViews:
                        keyIJ:str = str(rouletteExpH) + "Clk" + str(lrClickJ).replace(".", "") + "View" + str(lrViewK).replace(".", "") + nImplFeedbackI
                        eTool:AEvalTool = EvalToolDHondt({EvalToolDHondt.ARG_LEARNING_RATE_CLICKS: lrClickJ,
                                                          EvalToolDHondt.ARG_LEARNING_RATE_VIEWS: lrViewK})
                        nImplFeedback:APenalization = BatchNegDHondtFixed.getNegativeImplFeedbackParameters()[nImplFeedbackI]
                        aDict[keyIJ] = (rouletteExpH, nImplFeedback, eTool)
        return aDict


    @staticmethod
    def run(batchID:str, divisionDatasetPercentualSize:int, uBehaviour:str, repetition:int, jobID:str):

        #eTool:AEvalTool
        rouletteExp, nImplFeedback, eTool = BatchNegDHondtRoulette.getParameters()[jobID]

        aConf:AML1MConf = AML1MConf(batchID, divisionDatasetPercentualSize, uBehaviour, repetition)

        rIDs, rDescs = InputRecomDefinition.exportPairOfRecomIdsAndRecomDescrs(aConf.datasetID)

        aDescDHontRouletteI:AggregationDescription = None
        if rouletteExp == 1:
            aDescDHontRouletteI = InputAggrDefinition.exportADescNegDHontRoulette(nImplFeedback)
        elif rouletteExp == 3:
            aDescDHontRouletteI = InputAggrDefinition.exportADescNegDHontRoulette3(nImplFeedback)

        pDescr:Portfolio1AggrDescription = Portfolio1AggrDescription(
            "NegDHontRoulette" + jobID, rIDs, rDescs, aDescDHontRouletteI)

        model:DataFrame = ModelDefinition.createDHontModel(pDescr.getRecommendersIDs())

        aConf.run(pDescr, model, eTool)


    @staticmethod
    def generateBatches():

        divisionsDatasetPercentualSize:List[int] = [90]
        uBehaviours:List[str] = [Behaviours.BHVR_LINEAR0109, Behaviours.BHVR_STATIC08,
                                 Behaviours.BHVR_STATIC06, Behaviours.BHVR_STATIC04,
                                 Behaviours.BHVR_STATIC02]
        repetitions:List[int] = [1, 2, 3, 5]

        jobIDs:List[str] = list(BatchNegDHondtRoulette.getParameters().keys())

        for divisionDatasetPercentualSizeI in divisionsDatasetPercentualSize:
            for uBehaviourJ in uBehaviours:
                for repetitionK in repetitions:
                    for jobIDL in jobIDs:

                        batchID:str = "ml1mDiv" + str(divisionDatasetPercentualSizeI) + "U" + uBehaviourJ + "R" + str(repetitionK)
                        batchesDir:str = ".." + os.sep + "batches" + os.sep + batchID
                        if not os.path.exists(batchesDir):
                            os.mkdir(batchesDir)

                        job:str = str(BatchNegDHondtRoulette.__name__) + jobIDL
                        text:str = str(BatchNegDHondtRoulette.__name__) + ".run('" \
                                   + str(batchID) + "', " \
                                   + str(divisionDatasetPercentualSizeI) + ", '" \
                                   + str(uBehaviourJ) + "', " \
                                   + str(repetitionK) + ", " \
                                   + "'" + str(jobIDL) + "'" + ")"

                        jobFile:str = batchesDir + os.sep + job + ".txt"
                        BatchNegDHondtRoulette.__writeToFile(jobFile, text)


    @staticmethod
    def __writeToFile(fileName:str, text:str):
        f = open(fileName, "w")
        f.write(text)
        f.close()


if __name__ == "__main__":
   os.chdir("..")
   os.chdir("..")
   print(os.getcwd())
   BatchNegDHondtRoulette.generateBatches()