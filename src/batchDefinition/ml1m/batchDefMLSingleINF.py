#!/usr/bin/python3

import os
import pandas as pd

from typing import List

from pandas.core.frame import DataFrame #class

from portfolioDescription.portfolio1MethDescription import Portfolio1MethDescription #class
from portfolioDescription.portfolioNeg1MethDescription import PortfolioNeg1MethDescription #class

from batchDefinition.inputRecomMLDefinition import InputRecomMLDefinition #class

from portfolioDescription.aPortfolioDescription import APortfolioDescription #class

from evaluationTool.evalToolDoNothing import EToolDoNothing #class

from batchDefinition.inputABatchDefinition import InputABatchDefinition

from batchDefinition.aBatchDefinitionML import ABatchDefinitionML #class
from batchDefinition.ml1m.batchDefMLSingle import BatchDefMLSingle #class
from batchDefinition.ml1m.batchDefMLFuzzyDHondtINF import BatchDefMLFuzzyDHondtINF #class

from aggregation.negImplFeedback.aPenalization import APenalization #class

from recommenderDescription.recommenderDescription import RecommenderDescription #class

from batchDefinition.inputSimulatorDefinition import InputSimulatorDefinition #class

from simulator.simulator import Simulator #class

from history.historyHierDF import HistoryHierDF #class


class BatchDefMLSingleINF(ABatchDefinitionML):

    def getBatchName(self):
        return "SingleINF"

    @staticmethod
    def getParameters():

        recommenderIDs:List[str] = BatchDefMLSingle().getParameters().keys()
        negativeImplFeedback:List[str] = BatchDefMLFuzzyDHondtINF().getNegativeImplFeedbackParameters().keys()

        aDict:dict = {}
        for recommenderIDKeyI in recommenderIDs:
            for nImplFeedbackI in negativeImplFeedback:
                keyIJ:str = str(recommenderIDKeyI) + "INF" + nImplFeedbackI

                recommenderID:str = BatchDefMLSingle().getParameters()[recommenderIDKeyI]
                nImplFeedback:APenalization = BatchDefMLFuzzyDHondtINF().getNegativeImplFeedbackParameters()[nImplFeedbackI]

                aDict[keyIJ] = (recommenderID, nImplFeedback)
        return aDict


    def run(self, batchID:str, jobID:str):

        divisionDatasetPercentualSize:int
        uBehaviour:str
        repetition:int
        divisionDatasetPercentualSize, uBehaviour, repetition = InputABatchDefinition().getBatchParameters(self.datasetID)[batchID]

        datasetID:str = "ml1m" + "Div" + str(divisionDatasetPercentualSize)

        recommenderID:str
        nImplFeedback:APenalization
        recommenderID, nImplFeedback = self.getParameters()[jobID]

        rDescr:RecommenderDescription = InputRecomMLDefinition.exportInputRecomDefinition(recommenderID)

        pDescr:APortfolioDescription = PortfolioNeg1MethDescription(jobID.title(), recommenderID, rDescr, nImplFeedback)

        simulator:Simulator = InputSimulatorDefinition().exportSimulatorML1M(
                batchID, divisionDatasetPercentualSize, uBehaviour, repetition)
        simulator.simulate([pDescr], [DataFrame()], [EToolDoNothing({})], [HistoryHierDF(pDescr.getPortfolioID())])




if __name__ == "__main__":
   os.chdir("..")
   os.chdir("..")
   print(os.getcwd())
   
   BatchDefMLSingle.generateAllBatches(InputABatchDefinition())