#!/usr/bin/python3

import os

from typing import List
from typing import Dict

from pandas.core.frame import DataFrame #class

from portfolioDescription.portfolio1MethDescription import Portfolio1MethDescription #class

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
from batchDefinition.inputRecomRRDefinition import InputRecomRRDefinition #class


class BatchDefRRSingle(ABatchDefinitionRR):

    def getBatchName(self):
        return "Single"
    
    def getParameters(self):

        aDict:Dict[str,str] = {}
        aDict[InputRecomRRDefinition.THE_MOST_POPULAR] = InputRecomRRDefinition.THE_MOST_POPULAR
        aDict[InputRecomRRDefinition.KNN] = InputRecomRRDefinition.KNN

        return aDict


    def run(self, batchID:str, jobID:str):

        divisionDatasetPercentualSize:int
        uBehaviour:str
        repetition:int
        divisionDatasetPercentualSize, uBehaviour, repetition = InputABatchDefinition().getBatchParameters(self.datasetID)[batchID]

        recommenderID:str = self.getParameters()[jobID]

        rDescr:RecommenderDescription = InputRecomRRDefinition.exportInputRecomDefinition(recommenderID)

        pDescr:APortfolioDescription = Portfolio1MethDescription(recommenderID.title(), recommenderID, rDescr)

        simulator:Simulator = InputSimulatorDefinition.exportSimulatorRetailRocket(
                batchID, divisionDatasetPercentualSize, uBehaviour, repetition)
        simulator.simulate([pDescr], [DataFrame()], [EToolDoNothing({})], [HistoryHierDF(pDescr.getPortfolioID())])



if __name__ == "__main__":
   os.chdir("..")
   os.chdir("..")
   print(os.getcwd())

   BatchDefRRSingle().generateAllBatches(InputABatchDefinition())

   #BatchDefRRSingle().run("rrDiv90Ulinear0109R1", "TheMostPopular")
   #BatchDefRRSingle().run('rrDiv90Ulinear0109R1', 'KNN')
