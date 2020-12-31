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


class BatchMLSingle2(ABatchML):

    @staticmethod
    def getParameters():


        rCB1:ARecommender = RecommenderDescription(RecommenderCosineCB, {
                    RecommenderCosineCB.ARG_CB_DATA_PATH: Configuration.cbML1MDataFileWithPathTFIDF,
                    RecommenderCosineCB.ARG_USER_PROFILE_SIZE: 5,
                    RecommenderCosineCB.ARG_USER_PROFILE_STRATEGY: "max"})

        rCB2:ARecommender = RecommenderDescription(RecommenderCosineCB, {
                    RecommenderCosineCB.ARG_CB_DATA_PATH: Configuration.cbML1MDataFileWithPathOHE,
                    RecommenderCosineCB.ARG_USER_PROFILE_SIZE: 1,
                    RecommenderCosineCB.ARG_USER_PROFILE_STRATEGY: "max"})

        rCB3:ARecommender = RecommenderDescription(RecommenderCosineCB, {
                    RecommenderCosineCB.ARG_CB_DATA_PATH: Configuration.cbML1MDataFileWithPathOHE,
                    RecommenderCosineCB.ARG_USER_PROFILE_SIZE: 3,
                    RecommenderCosineCB.ARG_USER_PROFILE_STRATEGY: "weightedMean"})


        rW2V1:ARecommender = RecommenderDescription(RecommenderW2V, {
                    RecommenderW2V.ARG_ITERATIONS: 50000,
                    RecommenderW2V.ARG_TRAIN_VARIANT: 'positive',
                    RecommenderW2V.ARG_USER_PROFILE_SIZE: -1,
                    RecommenderW2V.ARG_USER_PROFILE_STRATEGY: 'weightedMean',
                    RecommenderW2V.ARG_VECTOR_SIZE: 128,
                    RecommenderW2V.ARG_WINDOW_SIZE: 5})

        rW2V2:ARecommender = RecommenderDescription(RecommenderW2V, {
                    RecommenderW2V.ARG_ITERATIONS: 50000,
                    RecommenderW2V.ARG_TRAIN_VARIANT: 'positive',
                    RecommenderW2V.ARG_USER_PROFILE_SIZE: -1,
                    RecommenderW2V.ARG_USER_PROFILE_STRATEGY: 'mean',
                    RecommenderW2V.ARG_VECTOR_SIZE: 128,
                    RecommenderW2V.ARG_WINDOW_SIZE: 5})

        rW2V3:ARecommender = RecommenderDescription(RecommenderW2V, {
                    RecommenderW2V.ARG_ITERATIONS: 50000,
                    RecommenderW2V.ARG_TRAIN_VARIANT: 'positive',
                    RecommenderW2V.ARG_USER_PROFILE_SIZE: -1,
                    RecommenderW2V.ARG_USER_PROFILE_STRATEGY: 'weightedMean',
                    RecommenderW2V.ARG_VECTOR_SIZE: 128,
                    RecommenderW2V.ARG_WINDOW_SIZE: 3})


        rBPRMF1:ARecommender = RecommenderDescription(RecommenderBPRMF, {
                    RecommenderBPRMF.ARG_FACTORS: 20,
                    RecommenderBPRMF.ARG_ITERATIONS: 50,
                    RecommenderBPRMF.ARG_LEARNINGRATE: 0.003,
                    RecommenderBPRMF.ARG_REGULARIZATION: 0.003})

        rBPRMF2:ARecommender = RecommenderDescription(RecommenderBPRMF, {
                    RecommenderBPRMF.ARG_FACTORS: 20,
                    RecommenderBPRMF.ARG_ITERATIONS: 10,
                    RecommenderBPRMF.ARG_LEARNINGRATE: 0.01,
                    RecommenderBPRMF.ARG_REGULARIZATION: 0.003})

        rBPRMF3:ARecommender = RecommenderDescription(RecommenderBPRMF, {
                    RecommenderBPRMF.ARG_FACTORS: 10,
                    RecommenderBPRMF.ARG_ITERATIONS: 50,
                    RecommenderBPRMF.ARG_LEARNINGRATE: 0.003,
                    RecommenderBPRMF.ARG_REGULARIZATION: 0.01})


        aDict:dict = {}

        aDict["rCB1 "] = rCB1
        aDict["rCB2 "] = rCB2
        aDict["rCB3 "] = rCB3

        aDict["rW2V1"] = rW2V1
        aDict["rW2V2"] = rW2V2
        aDict["rW2V3"] = rW2V3

        aDict["rBPRMF1"] = rBPRMF1
        aDict["rBPRMF2"] = rBPRMF2
        aDict["rBPRMF3"] = rBPRMF3

        return aDict


    def run(self, batchID:str, jobID:str):

        divisionDatasetPercentualSize:int
        uBehaviour:str
        repetition:int
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
   BatchMLSingle2.generateBatches()