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

from input.batchesML1m.aML1MConfig import AML1MConf #function

from datasets.behaviours import Behaviours #class

from aggregation.operators.aDHondtSelector import ADHondtSelector #class
from aggregation.operators.rouletteWheelSelector import RouletteWheelSelector #class
from aggregation.operators.theMostVotedItemSelector import TheMostVotedItemSelector #class



class BatchDHondt:

    SLCTR_ROULETTE1:str = "Roulette1"
    SLCTR_ROULETTE2:str = "Roulette3"
    SLCTR_FIXED:str = "Fixed"

    @staticmethod
    def getSelectorParameters():

        selectorRoulette1:ADHondtSelector = RouletteWheelSelector({RouletteWheelSelector.ARG_EXPONENT:1})
        selectorRoulette3:ADHondtSelector = RouletteWheelSelector({RouletteWheelSelector.ARG_EXPONENT:3})
        selectorFixed:ADHondtSelector = TheMostVotedItemSelector({})

        aDict:dict = {}
        aDict[BatchDHondt.SLCTR_ROULETTE1] = selectorRoulette1
        aDict[BatchDHondt.SLCTR_ROULETTE2] = selectorRoulette3
        aDict[BatchDHondt.SLCTR_FIXED] = selectorFixed
        return aDict


    @staticmethod
    def getParameters():
        selectorIDs:List[str] = BatchDHondt.getSelectorParameters().keys()
        lrClicks:List[float] = [0.2, 0.1, 0.02, 0.005]
        lrViews:List[float] = [0.1 / 500, 0.1 / 200, 0.1 / 1000]

        aDict:dict = {}
        for selectorIDI in selectorIDs:
            for lrClickJ in lrClicks:
                for lrViewK in lrViews:
                    keyIJ:str = selectorIDI + "Clk" + str(lrClickJ).replace(".", "") + "View" + str(lrViewK).replace(".", "")
                    eTool:AEvalTool = EvalToolDHondt({EvalToolDHondt.ARG_LEARNING_RATE_CLICKS: lrClickJ,
                                                      EvalToolDHondt.ARG_LEARNING_RATE_VIEWS: lrViewK})
                    selectorI:ADHondtSelector = BatchDHondt.getSelectorParameters()[selectorIDI]
                    aDict[keyIJ] = (selectorI, eTool)
        return aDict

    @staticmethod
    def run(batchID:str, divisionDatasetPercentualSize:int, uBehaviour:str, repetition:int, jobID:str):

        #eTool:AEvalTool
        selector, eTool = BatchDHondt.getParameters()[jobID]

        aConf:AML1MConf = AML1MConf(batchID, divisionDatasetPercentualSize, uBehaviour, repetition)

        rIDs, rDescs = InputRecomDefinition.exportPairOfRecomIdsAndRecomDescrs(aConf.datasetID)

        aDescDHont:AggregationDescription = InputAggrDefinition.exportADescDHont(selector)

        pDescr:Portfolio1AggrDescription = Portfolio1AggrDescription(
            "DHont" + jobID, rIDs, rDescs, aDescDHont)

        model:DataFrame = ModelDefinition.createDHontModel(pDescr.getRecommendersIDs())

        aConf.run(pDescr, model, eTool)


    @staticmethod
    def generateBatches():

        divisionsDatasetPercentualSize:List[int] = [90]
        uBehaviours:List[str] = [Behaviours.BHVR_LINEAR0109, Behaviours.BHVR_STATIC08,
                                 Behaviours.BHVR_STATIC06, Behaviours.BHVR_STATIC04,
                                 Behaviours.BHVR_STATIC02]
        repetitions:List[int] = [1, 2, 3, 5]

        jobIDs:List[str] = list(BatchDHondt.getParameters().keys())

        for divisionDatasetPercentualSizeI in divisionsDatasetPercentualSize:
            for uBehaviourJ in uBehaviours:
                for repetitionK in repetitions:
                    for jobIDL in jobIDs:

                        batchID:str = "ml1mDiv" + str(divisionDatasetPercentualSizeI) + "U" + uBehaviourJ + "R" + str(repetitionK)
                        batchesDir:str = ".." + os.sep + "batches" + os.sep + batchID
                        if not os.path.exists(batchesDir):
                            os.mkdir(batchesDir)

                        job:str = str(BatchDHondt.__name__) + jobIDL
                        text:str = str(BatchDHondt.__name__) + ".run('" \
                                   + str(batchID) + "', " \
                                   + str(divisionDatasetPercentualSizeI) + ", '" \
                                   + str(uBehaviourJ) + "', " \
                                   + str(repetitionK) + ", " \
                                   + "'" + str(jobIDL) + "'" + ")"

                        jobFile:str = batchesDir + os.sep + job + ".txt"
                        BatchDHondt.__writeToFile(jobFile, text)


    @staticmethod
    def __writeToFile(fileName:str, text:str):
        f = open(fileName, "w")
        f.write(text)
        f.close()


if __name__ == "__main__":
    os.chdir("..")
    os.chdir("..")
    print(os.getcwd())
    BatchDHondt.generateBatches()