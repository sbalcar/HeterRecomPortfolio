#!/usr/bin/python3

import os

from typing import List

from pandas.core.frame import DataFrame #class

from portfolioDescription.portfolio1MethDescription import Portfolio1MethDescription #class

from input.inputRecomMLDefinition import InputRecomMLDefinition #class

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


class BatchMLSingleBPRMFHT(ABatchML):

    @staticmethod
    def getParameters():

        factors:List[int] = [5, 10, 20, 50, 100]
        iterations:List[int] = [5, 10, 20, 50]
        learningRates:List[int] = [0.1, 0.03, 0.01, 0.003]
        regularizations:List[int] = [0.1, 0.03, 0.01, 0.003]

        aDict:dict = {}
        for factorI in factors:
            for iterationI in iterations:
                for learningRateI in learningRates:
                    for regularizationI in regularizations:
                        keyI:str = "RecommenderBPRMF" + "f" + str(factorI) + "i" + str(iterationI) +\
                                   "lr" + str(learningRateI).replace('.', '') + "r" + str(regularizationI).replace('.', '')

                        rBPRMFI:ARecommender = RecommenderDescription(RecommenderBPRMF, {
                            RecommenderBPRMF.ARG_FACTORS: factorI,
                            RecommenderBPRMF.ARG_ITERATIONS: iterationI,
                            RecommenderBPRMF.ARG_LEARNINGRATE: learningRateI,
                            RecommenderBPRMF.ARG_REGULARIZATION: regularizationI})

                        aDict[keyI] = rBPRMFI
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

    BatchMLSingleBPRMFHT.generateBatches()