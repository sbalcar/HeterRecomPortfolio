#!/usr/bin/python3

import os
import pandas as pd

from typing import List

from pandas.core.frame import DataFrame #class

from portfolioDescription.portfolio1MethDescription import Portfolio1MethDescription #class
from portfolioDescription.portfolioNeg1MethDescription import PortfolioNeg1MethDescription #class

from input.inputRecomDefinition import InputRecomDefinition #class

from portfolioDescription.aPortfolioDescription import APortfolioDescription #class

from evaluationTool.evalToolSingleMethod import EToolSingleMethod #class

from input.aBatch import BatchParameters #class

from input.aBatchML import ABatchML #class
from input.batchesML1m.batchSingle import BatchSingle #class
from input.batchesML1m.batchFuzzyDHondtINF import BatchFuzzyDHondtINF #class

from aggregation.negImplFeedback.aPenalization import APenalization #class

from recommenderDescription.recommenderDescription import RecommenderDescription #class

from input.inputSimulatorDefinition import InputSimulatorDefinition #class

from simulator.simulator import Simulator #class

from history.historyHierDF import HistoryHierDF #class


class BatchSingleINF(ABatchML):

    @staticmethod
    def getParameters():

        recommenderIDs:List[str] = BatchSingle().getParameters().keys()
        negativeImplFeedback:List[str] = BatchFuzzyDHondtINF().getNegativeImplFeedbackParameters().keys()

        aDict:dict = {}
        for recommenderIDKeyI in recommenderIDs:
            for nImplFeedbackI in negativeImplFeedback:
                keyIJ:str = str(recommenderIDKeyI) + nImplFeedbackI

                recommenderID:str = BatchSingle().getParameters()[recommenderIDKeyI]
                nImplFeedback:APenalization = BatchFuzzyDHondtINF().getNegativeImplFeedbackParameters()[nImplFeedbackI]

                aDict[keyIJ] = (recommenderID, nImplFeedback)
        return aDict


    def run(self, batchID:str, jobID:str):

        divisionDatasetPercentualSize:int
        uBehaviour:str
        repetition:int
        divisionDatasetPercentualSize, uBehaviour, repetition = BatchParameters.getBatchParameters(self.datasetID)[batchID]

        datasetID:str = "ml1m" + "Div" + str(divisionDatasetPercentualSize)

        recommenderID:str
        nImplFeedback:APenalization
        recommenderID, nImplFeedback = self.getParameters()[jobID]

        rDescr:RecommenderDescription = InputRecomDefinition.exportInputRecomDefinition(recommenderID)

        pDescr:APortfolioDescription = PortfolioNeg1MethDescription(jobID.title(), recommenderID, rDescr, nImplFeedback)

        simulator:Simulator = InputSimulatorDefinition.exportSimulatorML1M(
                batchID, divisionDatasetPercentualSize, uBehaviour, repetition)
        simulator.simulate([pDescr], [DataFrame()], [EToolSingleMethod({})], HistoryHierDF)




if __name__ == "__main__":
   os.chdir("..")
   os.chdir("..")
   print(os.getcwd())
   BatchSingle.generateBatches()