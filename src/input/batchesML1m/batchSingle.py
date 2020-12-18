#!/usr/bin/python3

import os

from typing import List

from pandas.core.frame import DataFrame #class

from portfolioDescription.portfolio1MethDescription import Portfolio1MethDescription #class

from input.inputRecomDefinition import InputRecomDefinition #class

from portfolioDescription.aPortfolioDescription import APortfolioDescription #class

from evaluationTool.evalToolSingleMethod import EToolSingleMethod #class

from recommenderDescription.recommenderDescription import RecommenderDescription #class

from input.inputSimulatorDefinition import InputSimulatorDefinition #class

from simulator.simulator import Simulator #class

from history.historyHierDF import HistoryHierDF #class

import pandas as pd
from input.aBatch import BatchParameters #class
from input.aBatchML import ABatchML #class


class BatchSingle(ABatchML):

    @staticmethod
    def getParameters():

        aDict:dict = {}
        aDict[InputRecomDefinition.COS_CB_MEAN] = InputRecomDefinition.COS_CB_MEAN
        aDict[InputRecomDefinition.COS_CB_WINDOW3] = InputRecomDefinition.COS_CB_WINDOW3
        aDict[InputRecomDefinition.THE_MOST_POPULAR] = InputRecomDefinition.THE_MOST_POPULAR
        aDict[InputRecomDefinition.W2V_POSNEG_MEAN] = InputRecomDefinition.W2V_POSNEG_MEAN
        aDict[InputRecomDefinition.W2V_POSNEG_WINDOW3] = InputRecomDefinition.W2V_POSNEG_WINDOW3
        aDict[InputRecomDefinition.KNN] = InputRecomDefinition.KNN
        return aDict


    def run(self, batchID:str, jobID:str):

        divisionDatasetPercentualSize:int
        uBehaviour:str
        repetition:int
        divisionDatasetPercentualSize, uBehaviour, repetition = BatchParameters.getBatchParameters(self.datasetID)[batchID]

        datasetID:str = "ml1m" + "Div" + str(divisionDatasetPercentualSize)

        recommenderID:str = self.getParameters()[jobID]

        rDescr:RecommenderDescription = InputRecomDefinition.exportInputRecomDefinition(recommenderID, datasetID)

        pDescr:APortfolioDescription = Portfolio1MethDescription(recommenderID.title(), recommenderID, rDescr)

        simulator:Simulator = InputSimulatorDefinition.exportSimulatorML1M(
                batchID, divisionDatasetPercentualSize, uBehaviour, repetition)
        simulator.simulate([pDescr], [DataFrame()], [EToolSingleMethod({})], HistoryHierDF)



if __name__ == "__main__":
   os.chdir("..")
   os.chdir("..")
   print(os.getcwd())
   BatchSingle.generateBatches()