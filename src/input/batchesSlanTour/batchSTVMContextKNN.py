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

from recommender.aRecommender import ARecommender #class
from recommender.recommenderVSKNN import RecommenderVMContextKNN #class

import pandas as pd
from input.aBatch import ABatch #class
from input.aBatch import BatchParameters #class

from input.aBatchST import ABatchST #class


class BatchSTVMContextKNN(ABatchST):

    @staticmethod
    def getParameters():

        aDict:dict = {}
        for kI in [25, 50, 75]:
            keyI:str = "K" + str(kI)
            aDict[keyI] = kI
        return aDict



    def run(self, batchID:str, jobID:str):

        divisionDatasetPercentualSize:int
        uBehaviour:str
        repetition:int
        divisionDatasetPercentualSize, uBehaviour, repetition = BatchParameters.getBatchParameters(self.datasetID)[batchID]

        kI:str = self.getParameters()[jobID]

        recommenderID:str = "vmContextKNN" + "K " + str(kI)

        rVMCtKNN:RecommenderDescription = RecommenderDescription(recommenderID, RecommenderVMContextKNN, {
                    RecommenderVMContextKNN.ARG_K: kI})

        pDescr:APortfolioDescription = Portfolio1MethDescription(recommenderID.title(), recommenderID, rVMCtKNN)

        simulator:Simulator = InputSimulatorDefinition.exportSimulatorSlantour(
                batchID, divisionDatasetPercentualSize, uBehaviour, repetition)
        simulator.simulate([pDescr], [DataFrame()], [EToolSingleMethod({})], HistoryHierDF)



if __name__ == "__main__":
   os.chdir("..")
   os.chdir("..")
   #print(os.getcwd())

   BatchSTVMContextKNN.generateBatches()