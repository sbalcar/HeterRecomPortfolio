#!/usr/bin/python3

import os

from typing import List

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
from batchDefinition.aBatchDefinitionST import ABatchDefinitionST #class
from batchDefinition.ml1m.batchDefMLSingleBPRMFHT import BatchDefMLSingleBPRMFHT #class

from configuration.configuration import Configuration #class


class BatchDefSTSingleBPRMFHT(ABatchDefinitionST):

    @staticmethod
    def getParameters():

        return BatchDefMLSingleBPRMFHT.getParameters()


    def run(self, batchID: str, jobID: str):
        divisionDatasetPercentualSize: int
        uBehaviour: str
        repetition: int
        divisionDatasetPercentualSize, uBehaviour, repetition = InputABatchDefinition.getBatchParameters(self.datasetID)[batchID]

        rDescr:RecommenderDescription = self.getParameters()[jobID]
        recommenderID:str = jobID

        pDescr:APortfolioDescription = Portfolio1MethDescription(recommenderID.title(), recommenderID, rDescr)

        simulator:Simulator = InputSimulatorDefinition.exportSimulatorSlantour(
            batchID, divisionDatasetPercentualSize, uBehaviour, repetition)
        simulator.simulate([pDescr], [DataFrame()], [EToolDoNothing({})], [HistoryHierDF(pDescr.getPortfolioID())])


if __name__ == "__main__":
    os.chdir("..")
    os.chdir("..")
    print(os.getcwd())

    BatchDefSTSingleBPRMFHT.generateAllBatches()