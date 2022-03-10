#!/usr/bin/python3

import os

from typing import List #class
from typing import Dict #class

from pandas.core.frame import DataFrame #class

from portfolioDescription.portfolio1MethDescription import Portfolio1MethDescription #class

from batchDefinition.inputABatchDefinition import InputABatchDefinition #class

from batchDefinition.inputRecomMLDefinition import InputRecomMLDefinition #class

from portfolioDescription.aPortfolioDescription import APortfolioDescription #class

from evaluationTool.evalToolDoNothing import EToolDoNothing #class

from recommenderDescription.recommenderDescription import RecommenderDescription #class

from recommender.aRecommender import ARecommender #class
from recommender.recommenderW2V import RecommenderW2V #class
from recommender.recommenderCosineCB import RecommenderCosineCB #class
from recommender.recommenderBPRMFImplicit import RecommenderBPRMFImplicit #class

from batchDefinition.inputSimulatorDefinition import InputSimulatorDefinition #class

from simulator.simulator import Simulator #class

from history.historyHierDF import HistoryHierDF #class

import pandas as pd
from batchDefinition.inputABatchDefinition import InputABatchDefinition
from batchDefinition.aBatchDefinitionRR import ABatchDefinitionRR #class

from batchDefinition.ml1m.batchDefMLSingleW2VHT import BatchDefMLSingleW2VHT #class

from configuration.configuration import Configuration #class


class BatchDefRRSingleW2VHT(ABatchDefinitionRR):

    def getBatchName(self):
        return "Single"
    
    def getParameters(self):
        BatchDefMLSingleW2VHT.learningRates = [0.1, 0.3, 0.6]
        BatchDefMLSingleW2VHT.trainVariants = ["all"]
        BatchDefMLSingleW2VHT.iterations:List[int] = [50000, 100000]
        BatchDefMLSingleW2VHT.windowSizes:List[int] = [5, 3, 1]
        BatchDefMLSingleW2VHT.vectorSizes:List[int] = [32, 64]
        BatchDefMLSingleW2VHT.userProfileStrategies:List[str] = ["weightedMean"]
        BatchDefMLSingleW2VHT.userProfileSizes:List[int] = [-1, 1, 3, 5, 7, 10]

        paramsDict:Dict[str, object] = BatchDefMLSingleW2VHT.getParameters()

        return paramsDict


    def run(self, batchID:str, jobID:str):
        divisionDatasetPercentualSize:int
        uBehaviour:str
        repetition:int
        divisionDatasetPercentualSize, uBehaviour, repetition = InputABatchDefinition().getBatchParameters(self.datasetID)[batchID]

        rDescr:RecommenderDescription = self.getParameters()[jobID]
        recommenderID:str = "W2V" + jobID

        pDescr:APortfolioDescription = Portfolio1MethDescription(recommenderID.title(), recommenderID, rDescr)

        simulator:Simulator = InputSimulatorDefinition.exportSimulatorRetailRocket(
            batchID, divisionDatasetPercentualSize, uBehaviour, repetition)
        simulator.simulate([pDescr], [DataFrame()], [EToolDoNothing({})], [HistoryHierDF(pDescr.getPortfolioID())])


if __name__ == "__main__":
    os.chdir("..")
    os.chdir("..")
    print(os.getcwd())

    BatchDefRRSingleW2VHT().generateAllBatches(InputABatchDefinition())