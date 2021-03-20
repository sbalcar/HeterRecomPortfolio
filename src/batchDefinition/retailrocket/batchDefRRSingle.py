#!/usr/bin/python3

import os

from typing import List

from pandas.core.frame import DataFrame #class

from portfolioDescription.portfolio1MethDescription import Portfolio1MethDescription #class

from batchDefinition.inputRecomMLDefinition import InputRecomMLDefinition #class

from portfolioDescription.aPortfolioDescription import APortfolioDescription #class

from evaluationTool.evalToolDoNothing import EToolDoNothing #class

from recommenderDescription.recommenderDescription import RecommenderDescription #class

from batchDefinition.inputSimulatorDefinition import InputSimulatorDefinition #class

from simulator.simulator import Simulator #class

from history.historyHierDF import HistoryHierDF #class

import pandas as pd
from batchDefinition.aBatchDefinition import ABatchDefinition #class
from batchDefinition.inputABatchDefinition import InputABatchDefinition

from batchDefinition.aBatchDefinitionRR import ABatchDefinitionRR #class


class BatchDefRRSingle(ABatchDefinitionRR):

    def getBatchName(self):
        return "Single"
    
    def getParameters(self):

        aDict:dict = {}
        aDict[InputRecomMLDefinition.THE_MOST_POPULAR] = InputRecomMLDefinition.THE_MOST_POPULAR
        aDict[InputRecomMLDefinition.BPRMFf100i10lr0003r01] = InputRecomMLDefinition.BPRMFf100i10lr0003r01

        return aDict


    def run(self, batchID:str, jobID:str):

        divisionDatasetPercentualSize:int
        uBehaviour:str
        repetition:int
        divisionDatasetPercentualSize, uBehaviour, repetition = InputABatchDefinition.getBatchParameters(self.datasetID)[batchID]

        recommenderID:str = self.getParameters()[jobID]

        rDescr:RecommenderDescription = InputRecomMLDefinition.exportInputRecomDefinition(recommenderID)

        pDescr:APortfolioDescription = Portfolio1MethDescription(recommenderID.title(), recommenderID, rDescr)

        simulator:Simulator = InputSimulatorDefinition.exportSimulatorRetailRocket(
                batchID, divisionDatasetPercentualSize, uBehaviour, repetition)
        simulator.simulate([pDescr], [DataFrame()], [EToolDoNothing({})], [HistoryHierDF(pDescr.getPortfolioID())])



if __name__ == "__main__":
   os.chdir("..")
   os.chdir("..")
   print(os.getcwd())
   BatchDefRRSingle.generateAllBatches()