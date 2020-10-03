#!/usr/bin/python3

import os

from typing import List

from pandas.core.frame import DataFrame #class

from portfolioDescription.portfolio1AggrDescription import Portfolio1AggrDescription #class

from evaluationTool.aEvalTool import AEvalTool #class
from evaluationTool.evalToolDHondtBanditVotes import EvalToolDHondtBanditVotes #class

from aggregationDescription.aggregationDescription import AggregationDescription #class

from input.inputAggrDefinition import InputAggrDefinition, ModelDefinition  # class

from input.InputRecomDefinition import InputRecomDefinition #class

from input.batchesML1m.aML1MConfig import AML1MConf #function

from datasets.behaviours import Behaviours #class


class BatchDHondtBanditsVotesRoulette:

    @staticmethod
    def getParameters():
        rouletteExps:List[int] = [1, 3]
        lrClicks:List[float] = [0.2]
        lrViews:List[float] = [0.1 / 500]

        aDict:dict = {}
        for rouletteExpI in rouletteExps:
            for lrClickJ in lrClicks:
                for lrViewK in lrViews:
                    keyIJ:str = str(rouletteExpI) + "Clk" + str(lrClickJ).replace(".", "") + "View" + str(lrViewK).replace(".", "")
                    eTool:AEvalTool = EvalToolDHondtBanditVotes({})
                    aDict[keyIJ] = (rouletteExpI, eTool)
        return aDict

    @staticmethod
    def run(batchID:str, divisionDatasetPercentualSize:int, uBehaviour:str, repetition:int, jobID:str):

        #eTool:AEvalTool
        rouletteExp, eTool = BatchDHondtBanditsVotesRoulette.getParameters()[jobID]

        aConf:AML1MConf = AML1MConf(batchID, divisionDatasetPercentualSize, uBehaviour, repetition)

        rIDs, rDescs = InputRecomDefinition.exportPairOfRecomIdsAndRecomDescrs(aConf.datasetID)

        aDescDHontRouletteI:AggregationDescription = None
        if rouletteExp == 1:
            aDescDHontRouletteI = InputAggrDefinition.exportADescDHontBanditVotesRoulette()
        elif rouletteExp == 3:
            aDescDHontRouletteI = InputAggrDefinition.exportADescDHontBanditVotesRoulette3()

        pDescr:Portfolio1AggrDescription = Portfolio1AggrDescription(
            "DHondtRouletteBanditsVotes" + jobID, rIDs, rDescs, aDescDHontRouletteI)

        model:DataFrame = ModelDefinition.createDHondtBanditsVotesModel(pDescr.getRecommendersIDs())

        aConf.run(pDescr, model, eTool)


    @staticmethod
    def generateBatches():

        divisionsDatasetPercentualSize:List[int] = [90]
        uBehaviours:List[str] = [Behaviours.COL_LINEAR0109,# Behaviours.COL_STATIC08,
                                 #Behaviours.COL_STATIC06, Behaviours.COL_STATIC04,
                                 Behaviours.COL_STATIC02]
        repetitions:List[int] = [1, 2, 3, 5]

        jobIDs:List[str] = list(BatchDHondtBanditsVotesRoulette.getParameters().keys())

        for divisionDatasetPercentualSizeI in divisionsDatasetPercentualSize:
            for uBehaviourJ in uBehaviours:
                for repetitionK in repetitions:
                    for jobIDL in jobIDs:
                        

                        batchID:str = "ml1mDiv" + str(divisionDatasetPercentualSizeI) + "U" + uBehaviourJ + "R" + str(repetitionK)
                        batchesDir:str = ".." + os.sep + "batches" + os.sep + batchID
                        if not os.path.exists(batchesDir):
                            os.mkdir(batchesDir)
                        print(batchID,batchesDir)

                        job:str = str(BatchDHondtBanditsVotesRoulette.__name__) + jobIDL
                        text:str = str(BatchDHondtBanditsVotesRoulette.__name__) + ".run('"\
                                   + str(batchID) + "', "\
                                   + str(divisionDatasetPercentualSizeI) + ", '"\
                                   + str(uBehaviourJ) + "', "\
                                   + str(repetitionK) + ", "\
                                   + "'" + str(jobIDL) + "'" + ")"

                        jobFile:str = batchesDir + os.sep + job + ".txt"
                        BatchDHondtBanditsVotesRoulette.__writeToFile(jobFile, text)


    @staticmethod
    def __writeToFile(fileName:str, text:str):
        f = open(fileName, "w")
        f.write(text)
        f.close()


if __name__ == "__main__":
    os.chdir("..")
    os.chdir("..")
    print(os.getcwd())
    BatchDHondtBanditsVotesRoulette.generateBatches()
