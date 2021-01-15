#!/usr/bin/python3

import os

from typing import List #class
from typing import Dict #class

from pandas.core.frame import DataFrame #class

from portfolioDescription.portfolio1MethDescription import Portfolio1MethDescription #class

from input.inputRecomSTDefinition import InputRecomSTDefinition #class

from portfolioDescription.aPortfolioDescription import APortfolioDescription #class

from evaluationTool.evalToolSingleMethod import EToolSingleMethod #class

from recommenderDescription.recommenderDescription import RecommenderDescription #class

from input.inputSimulatorDefinition import InputSimulatorDefinition #class

from simulator.simulator import Simulator #class

from history.historyHierDF import HistoryHierDF #class

import pandas as pd
from input.aBatch import ABatch #class
from input.aBatch import BatchParameters #class

from input.aBatchST import ABatchST #class


class BatchSTSingle(ABatchST):

    @staticmethod
    def getParameters():

        aDict:Dict[str,object] = {}
        aDict[InputRecomSTDefinition.THE_MOST_POPULAR] = InputRecomSTDefinition.THE_MOST_POPULAR
        aDict[InputRecomSTDefinition.KNN] = InputRecomSTDefinition.KNN
        #aDict[InputRecomSTDefinition.VMC_KNN] = InputRecomSTDefinition.VMC_KNN

        return aDict


    def run(self, batchID:str, jobID:str):

        divisionDatasetPercentualSize:int
        uBehaviour:str
        repetition:int
        divisionDatasetPercentualSize, uBehaviour, repetition = BatchParameters.getBatchParameters(self.datasetID)[batchID]

        recommenderID:str = self.getParameters()[jobID]

        rDescr:RecommenderDescription = InputRecomSTDefinition.exportInputRecomDefinition(recommenderID)

        pDescr:APortfolioDescription = Portfolio1MethDescription(recommenderID.title(), recommenderID, rDescr)

        simulator:Simulator = InputSimulatorDefinition.exportSimulatorSlantour(
                batchID, divisionDatasetPercentualSize, uBehaviour, repetition)
        simulator.simulate([pDescr], [DataFrame()], [EToolSingleMethod({})], [HistoryHierDF(pDescr.getPortfolioID())])



if __name__ == "__main__":
   os.chdir("..")
   os.chdir("..")
   #print(os.getcwd())

   BatchSTSingle.generateBatches()