#!/usr/bin/python3

import os

from typing import List

from pandas.core.frame import DataFrame #class

from portfolioDescription.portfolio1MethDescription import Portfolio1MethDescription #class

from input.inputRecomMLDefinition import InputRecomMLDefinition #class

from portfolioDescription.aPortfolioDescription import APortfolioDescription #class

from evaluationTool.evalToolSingleMethod import EToolSingleMethod #class

from recommenderDescription.recommenderDescription import RecommenderDescription #class

from input.inputSimulatorDefinition import InputSimulatorDefinition #class

from simulator.simulator import Simulator #class

from history.historyHierDF import HistoryHierDF #class

import pandas as pd
from input.aBatch import BatchParameters #class
from input.aBatchML import ABatchML #class


class BatchMLSingle(ABatchML):

    @staticmethod
    def getParameters():

        aDict:dict = {}
        aDict[InputRecomMLDefinition.COS_CB_MEAN] = InputRecomMLDefinition.COS_CB_MEAN
        aDict[InputRecomMLDefinition.COS_CB_WINDOW3] = InputRecomMLDefinition.COS_CB_WINDOW3
        aDict[InputRecomMLDefinition.THE_MOST_POPULAR] = InputRecomMLDefinition.THE_MOST_POPULAR
        aDict[InputRecomMLDefinition.W2V_POSNEG_MEAN] = InputRecomMLDefinition.W2V_POSNEG_MEAN
        aDict[InputRecomMLDefinition.W2V_POSNEG_WINDOW3] = InputRecomMLDefinition.W2V_POSNEG_WINDOW3
        aDict[InputRecomMLDefinition.KNN] = InputRecomMLDefinition.KNN
        aDict[InputRecomMLDefinition.BPRMF] = InputRecomMLDefinition.BPRMF

        return aDict


    def run(self, batchID:str, jobID:str):

        divisionDatasetPercentualSize:int
        uBehaviour:str
        repetition:int
        divisionDatasetPercentualSize, uBehaviour, repetition = BatchParameters.getBatchParameters(self.datasetID)[batchID]

        recommenderID:str = self.getParameters()[jobID]

        rDescr:RecommenderDescription = InputRecomMLDefinition.exportInputRecomDefinition(recommenderID)

        pDescr:APortfolioDescription = Portfolio1MethDescription(recommenderID.title(), recommenderID, rDescr)

        simulator:Simulator = InputSimulatorDefinition.exportSimulatorML1M(
                batchID, divisionDatasetPercentualSize, uBehaviour, repetition)
        simulator.simulate([pDescr], [DataFrame()], [EToolSingleMethod({})], [HistoryHierDF(pDescr.getPortfolioID())])



if __name__ == "__main__":
   os.chdir("..")
   os.chdir("..")
   print(os.getcwd())
   BatchMLSingle.generateBatches()