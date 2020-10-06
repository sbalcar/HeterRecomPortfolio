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

from input.batchesML1m.batchDHondt import BatchDHondt #class

from aggregation.toolsDHontNF.penalizationOfResultsByNegImpFeedback.aPenalization import APenalization #class

from input.batchesML1m.aML1MConfig import AML1MConf #function

from datasets.behaviours import Behaviours #class

from aggregation.operators.aDHondtSelector import ADHondtSelector #class



class BatchNegDHondt:

    @staticmethod
    def getNegativeImplFeedbackParameters():

        pToolOLin0802HLin1002:APenalization = InputAggrDefinition.exportAPenaltyToolOLin0802HLin1002(AML1MConf.numberOfAggrItems)

        pToolOStat08HLin1002:APenalization = InputAggrDefinition.exportAPenaltyToolOStat08HLin1002(AML1MConf.numberOfAggrItems)

        aDict:dict = {}
        aDict["OLin0802HLin1002"] = pToolOLin0802HLin1002
        aDict["OStat08HLin1002"] = pToolOStat08HLin1002
        return aDict

    @staticmethod
    def getParameters():
        selectorIDs:List[str] = BatchDHondt.getSelectorParameters().keys()
        negativeImplFeedback:List[str] = BatchNegDHondt.getNegativeImplFeedbackParameters().keys()
        lrClicks:List[float] = [0.2, 0.1, 0.02, 0.005]
        lrViews:List[float] = [0.1 / 500, 0.1 / 200, 0.1 / 1000]

        aDict:dict = {}
        for selectorIDH in selectorIDs:
            for nImplFeedbackI in negativeImplFeedback:
                for lrClickJ in lrClicks:
                    for lrViewK in lrViews:
                        keyIJ:str = str(selectorIDH) + "Clk" + str(lrClickJ).replace(".", "") + "View" + str(lrViewK).replace(".", "") + nImplFeedbackI
                        eTool:AEvalTool = EvalToolDHondt({EvalToolDHondt.ARG_LEARNING_RATE_CLICKS: lrClickJ,
                                                          EvalToolDHondt.ARG_LEARNING_RATE_VIEWS: lrViewK})
                        nImplFeedback:APenalization = BatchNegDHondt.getNegativeImplFeedbackParameters()[nImplFeedbackI]
                        selectorH:ADHondtSelector = BatchDHondt.getSelectorParameters()[selectorIDH]

                        aDict[keyIJ] = (selectorH, nImplFeedback, eTool)
        return aDict


    @staticmethod
    def run(batchID:str, divisionDatasetPercentualSize:int, uBehaviour:str, repetition:int, jobID:str):

        #eTool:AEvalTool
        selector, nImplFeedback, eTool = BatchNegDHondt.getParameters()[jobID]

        aConf:AML1MConf = AML1MConf(batchID, divisionDatasetPercentualSize, uBehaviour, repetition)

        rIDs, rDescs = InputRecomDefinition.exportPairOfRecomIdsAndRecomDescrs(aConf.datasetID)

        aDescNegDHont:AggregationDescription = InputAggrDefinition.exportADescNegDHont(selector, nImplFeedback)

        pDescr:Portfolio1AggrDescription = Portfolio1AggrDescription(
            "NegDHont" + jobID, rIDs, rDescs, aDescNegDHont)

        model:DataFrame = ModelDefinition.createDHontModel(pDescr.getRecommendersIDs())

        aConf.run(pDescr, model, eTool)


    @staticmethod
    def generateBatches():

        divisionsDatasetPercentualSize:List[int] = [90]
        uBehaviours:List[str] = [Behaviours.BHVR_LINEAR0109, Behaviours.BHVR_STATIC08,
                                 Behaviours.BHVR_STATIC06, Behaviours.BHVR_STATIC04,
                                 Behaviours.BHVR_STATIC02]
        repetitions:List[int] = [1, 2, 3, 5]

        jobIDs:List[str] = list(BatchNegDHondt.getParameters().keys())

        for divisionDatasetPercentualSizeI in divisionsDatasetPercentualSize:
            for uBehaviourJ in uBehaviours:
                for repetitionK in repetitions:
                    for jobIDL in jobIDs:

                        batchID:str = "ml1mDiv" + str(divisionDatasetPercentualSizeI) + "U" + uBehaviourJ + "R" + str(repetitionK)
                        batchesDir:str = ".." + os.sep + "batches" + os.sep + batchID
                        if not os.path.exists(batchesDir):
                            os.mkdir(batchesDir)

                        job:str = str(BatchNegDHondt.__name__) + jobIDL
                        text:str = str(BatchNegDHondt.__name__) + ".run('" \
                                   + str(batchID) + "', " \
                                   + str(divisionDatasetPercentualSizeI) + ", '" \
                                   + str(uBehaviourJ) + "', " \
                                   + str(repetitionK) + ", " \
                                   + "'" + str(jobIDL) + "'" + ")"

                        jobFile:str = batchesDir + os.sep + job + ".txt"
                        BatchNegDHondt.__writeToFile(jobFile, text)


    @staticmethod
    def __writeToFile(fileName:str, text:str):
        f = open(fileName, "w")
        f.write(text)
        f.close()


if __name__ == "__main__":
   os.chdir("..")
   os.chdir("..")
   print(os.getcwd())
   BatchNegDHondt.generateBatches()