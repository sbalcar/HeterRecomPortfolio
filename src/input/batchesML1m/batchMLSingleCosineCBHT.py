#!/usr/bin/python3

import os

from typing import List

from pandas.core.frame import DataFrame #class

from portfolioDescription.portfolio1MethDescription import Portfolio1MethDescription #class

from input.inputRecomDefinition import InputRecomDefinition #class

from portfolioDescription.aPortfolioDescription import APortfolioDescription #class

from evaluationTool.evalToolSingleMethod import EToolSingleMethod #class

from recommenderDescription.recommenderDescription import RecommenderDescription #class

from recommender.aRecommender import ARecommender #class
from recommender.recommenderW2V import RecommenderW2V #class
from recommender.recommenderCosineCB import RecommenderCosineCB #class
from recommender.recommenderBPRMF import RecommenderBPRMF #class

from input.inputSimulatorDefinition import InputSimulatorDefinition #class

from simulator.simulator import Simulator #class

from history.historyHierDF import HistoryHierDF #class

import pandas as pd
from input.aBatch import BatchParameters #class
from input.aBatchML import ABatchML #class

from configuration.configuration import Configuration #class


class BatchMLSingleCosineCBHT(ABatchML):

    @staticmethod
    def getParameters():

        cbDataPaths:List[str] = [Configuration.cbDataFileWithPathTFIDF, Configuration.cbDataFileWithPathOHE]
        userProfileStrategies:List[str] = ["mean", "max", "weightedMean"]
        userProfileSizes:List[int] = [-1, 1, 3, 5, 7, 10]

        aDict:dict = {}
        for cbDataPathI in cbDataPaths:
            for userProfileStrategyI in userProfileStrategies:
                for userProfileSizeI in userProfileSizes:

                    cbDataPathStrI:str = ""
                    if cbDataPathI == Configuration.cbDataFileWithPathTFIDF:
                        cbDataPathStrI = "TFIDF"
                    elif cbDataPathI == Configuration.cbDataFileWithPathOHE:
                        cbDataPathStrI = "OHE"
                    else:
                        print("error")

                    keyI:str = "RecommenderCosineCB" + "cbd" + str(cbDataPathStrI) + "ups" + str(userProfileStrategyI) +\
                               "ups" + str(userProfileSizeI)

                    rCBI: ARecommender = RecommenderDescription(RecommenderCosineCB, {
                        RecommenderCosineCB.ARG_CB_DATA_PATH: cbDataPathI,
                        RecommenderCosineCB.ARG_USER_PROFILE_SIZE: userProfileSizeI,
                        RecommenderCosineCB.ARG_USER_PROFILE_STRATEGY: userProfileStrategyI})

                    aDict[keyI] = rCBI
        return aDict


    def run(self, batchID: str, jobID: str):
        divisionDatasetPercentualSize: int
        uBehaviour: str
        repetition: int
        divisionDatasetPercentualSize, uBehaviour, repetition = BatchParameters.getBatchParameters(self.datasetID)[batchID]

        rDescr:RecommenderDescription = self.getParameters()[jobID]
        recommenderID:str = jobID

        pDescr:APortfolioDescription = Portfolio1MethDescription(recommenderID.title(), recommenderID, rDescr)

        simulator:Simulator = InputSimulatorDefinition.exportSimulatorML1M(
            batchID, divisionDatasetPercentualSize, uBehaviour, repetition)
        simulator.simulate([pDescr], [DataFrame()], [EToolSingleMethod({})], HistoryHierDF)


if __name__ == "__main__":
    os.chdir("..")
    os.chdir("..")
    print(os.getcwd())

    BatchMLSingleCosineCBHT.generateBatches()