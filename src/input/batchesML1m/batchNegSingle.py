#!/usr/bin/python3

import os
import pandas as pd

from typing import List

from pandas.core.frame import DataFrame #class

from portfolioDescription.portfolio1MethDescription import Portfolio1MethDescription #class
from portfolioDescription.portfolioNeg1MethDescription import PortfolioNeg1MethDescription #class

from input.InputRecomDefinition import InputRecomDefinition #class

from portfolioDescription.aPortfolioDescription import APortfolioDescription #class

from input.batchesML1m.aML1MConfig import AML1MConf #function

from evaluationTool.evalToolSingleMethod import EToolSingleMethod #class

from input.aBatch import ABatch #class
from input.batchesML1m.batchSingle import BatchSingle #class
from input.batchesML1m.batchNegDHondt import BatchNegDHondt #class

from aggregation.negImplFeedback.aPenalization import APenalization #class

from recommenderDescription.recommenderDescription import RecommenderDescription #class


class BatchNegSingle(ABatch):

    def getParameters(self):

        recommenderIDs:List[str] = BatchSingle().getParameters().keys()
        negativeImplFeedback:List[str] = BatchNegDHondt().getNegativeImplFeedbackParameters().keys()

        aDict:dict = {}
        for recommenderIDKeyI in recommenderIDs:
            for nImplFeedbackI in negativeImplFeedback:
                keyIJ:str = str(recommenderIDKeyI) + nImplFeedbackI

                recommenderID:str = BatchSingle().getParameters()[recommenderIDKeyI]
                nImplFeedback:APenalization = BatchNegDHondt().getNegativeImplFeedbackParameters()[nImplFeedbackI]

                aDict[keyIJ] = (recommenderID, nImplFeedback)
        return aDict


    def run(self, batchID:str, jobID:str):

        from execute.generateBatches import BatchParameters #class
        divisionDatasetPercentualSize:int
        uBehaviour:str
        repetition:int
        divisionDatasetPercentualSize, uBehaviour, repetition = BatchParameters.getBatchParameters()[batchID]

        recommenderID:str
        nImplFeedback:APenalization
        recommenderID, nImplFeedback = self.getParameters()[jobID]

        aConf:AML1MConf = AML1MConf(batchID, divisionDatasetPercentualSize, uBehaviour, repetition)

        rDescr:RecommenderDescription = InputRecomDefinition.exportInputRecomDefinition(recommenderID, aConf.datasetID)

        pDescr:APortfolioDescription = PortfolioNeg1MethDescription(jobID.title(), recommenderID, rDescr, nImplFeedback)

        model:DataFrame = pd.DataFrame()
        eTool:List = EToolSingleMethod({})

        aConf.run(pDescr, model, eTool)



if __name__ == "__main__":
   os.chdir("..")
   os.chdir("..")
   print(os.getcwd())
   BatchSingle.generateBatches()