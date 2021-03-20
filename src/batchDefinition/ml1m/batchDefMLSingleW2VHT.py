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

from recommender.aRecommender import ARecommender #class
from recommender.recommenderW2V import RecommenderW2V #class
from recommender.recommenderCosineCB import RecommenderCosineCB #class
from recommender.recommenderBPRMF import RecommenderBPRMF #class

from batchDefinition.inputSimulatorDefinition import InputSimulatorDefinition #class

from simulator.simulator import Simulator #class

from history.historyHierDF import HistoryHierDF #class

import pandas as pd
from batchDefinition.inputABatchDefinition import InputABatchDefinition
from batchDefinition.aBatchDefinitionML import ABatchDefinitionML #class

from configuration.configuration import Configuration #class


class BatchDefMLSingleW2VHT(ABatchDefinitionML):

    trainVariants:List[str] = ["positive"]
    iterations:List[int] = [50000, 100000, 200000]
    windowSizes:List[int] = [5, 3, 1]
    vectorSizes:List[int] = [32, 64, 128]
    userProfileStrategies:List[str] = ["mean", "max", "weightedMean"]
    userProfileSizes:List[int] = [-1, 1, 3, 5, 7, 10]

    @classmethod
    def getParameters(cls):

        aDict:Dict[str,object] = {}
        for trainVariantI in cls.trainVariants:
            for iterationI in cls.iterations:
                for windowSizeI in cls.windowSizes:
                    for vectorSizeI in cls.vectorSizes:
                        for userProfileStrategyI in cls.userProfileStrategies:
                            for userProfileSizeI in cls.userProfileSizes:

                                keyI:str = "t" + str(trainVariantI) + "i" + str(iterationI) +\
                                            "ws" + str(windowSizeI) + "vs" + str(vectorSizeI) +\
                                            "ups" + userProfileStrategyI + "ups" + str(userProfileSizeI)

                                rW2V:ARecommender = RecommenderDescription(RecommenderW2V, {
                                    RecommenderW2V.ARG_ITERATIONS: iterationI,
                                    RecommenderW2V.ARG_TRAIN_VARIANT: trainVariantI,
                                    RecommenderW2V.ARG_USER_PROFILE_SIZE: userProfileSizeI,
                                    RecommenderW2V.ARG_USER_PROFILE_STRATEGY: userProfileStrategyI,
                                    RecommenderW2V.ARG_VECTOR_SIZE: vectorSizeI,
                                    RecommenderW2V.ARG_WINDOW_SIZE: windowSizeI})

                                aDict[keyI] = rW2V
        return aDict


    def run(self, batchID: str, jobID: str):
        divisionDatasetPercentualSize: int
        uBehaviour: str
        repetition: int
        divisionDatasetPercentualSize, uBehaviour, repetition = InputABatchDefinition.getBatchParameters(self.datasetID)[batchID]

        rDescr:RecommenderDescription = self.getParameters()[jobID]
        recommenderID:str = jobID

        pDescr:APortfolioDescription = Portfolio1MethDescription(recommenderID.title(), recommenderID, rDescr)

        simulator:Simulator = InputSimulatorDefinition.exportSimulatorML1M(
            batchID, divisionDatasetPercentualSize, uBehaviour, repetition)
        simulator.simulate([pDescr], [DataFrame()], [EToolDoNothing({})], [HistoryHierDF(pDescr.getPortfolioID())])


if __name__ == "__main__":
    os.chdir("..")
    os.chdir("..")
    print(os.getcwd())

    BatchDefMLSingleW2VHT.generateAllBatches()