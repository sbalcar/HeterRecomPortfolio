#!/usr/bin/python3

import os

from typing import List #class
from typing import Dict #class

from pandas.core.frame import DataFrame #class

from portfolioDescription.portfolio1MethDescription import Portfolio1MethDescription #class

from batchDefinition.inputRecomMLDefinition import InputRecomMLDefinition #class

from portfolioDescription.aPortfolioDescription import APortfolioDescription #class

from evaluationTool.evalToolDoNothing import EToolDoNothing #class

from recommenderDescription.recommenderDescription import RecommenderDescription #class

from batchDefinition.inputSimulatorDefinition import InputSimulatorDefinition #class

from simulator.simulator import Simulator #class

from history.historyHierDF import HistoryHierDF #class

from recommender.aRecommender import ARecommender #class
from recommender.recommenderVSKNN import RecommenderVMContextKNN #class

import pandas as pd
from batchDefinition.aBatchDefinition import ABatchDefinition #class
from batchDefinition.inputABatchDefinition import InputABatchDefinition

from batchDefinition.aBatchDefinitionRR import ABatchDefinitionRR #class

from batchDefinition.ml1m.batchDefMLSingleVMContextKNNHT import BatchDefMLSingleVMContextKNNHT #class


class BatchDefRRSingleVMContextKNNHT(ABatchDefinitionRR):

    def getBatchName(self):
        return "Single"
    
    def getParameters(self):
        return BatchDefMLSingleVMContextKNNHT.getParameters()


    def run(self, batchID:str, jobID:str):

        divisionDatasetPercentualSize:int
        uBehaviour:str
        repetition:int
        divisionDatasetPercentualSize, uBehaviour, repetition = InputABatchDefinition.getBatchParameters(self.datasetID)[batchID]

        kI:str = self.getParameters()[jobID]

        recommenderID:str = "RecommendervmContextKNN" + "K" + str(kI)

        rVMCtKNN:RecommenderDescription = RecommenderDescription(RecommenderVMContextKNN, {
                    RecommenderVMContextKNN.ARG_K: kI})

        pDescr:APortfolioDescription = Portfolio1MethDescription(recommenderID.title(), recommenderID, rVMCtKNN)

        simulator:Simulator = InputSimulatorDefinition.exportSimulatorRetailRocket(
                batchID, divisionDatasetPercentualSize, uBehaviour, repetition)
        simulator.simulate([pDescr], [DataFrame()], [EToolDoNothing({})], [HistoryHierDF(pDescr.getPortfolioID())])



if __name__ == "__main__":
   os.chdir("..")
   os.chdir("..")
   #print(os.getcwd())

   BatchDefRRSingleVMContextKNNHT.generateAllBatches()