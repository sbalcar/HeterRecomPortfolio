#!/usr/bin/python3

import os

from typing import List #class
from typing import Dict #class

from pandas.core.frame import DataFrame #class

from portfolioDescription.portfolio1MethDescription import Portfolio1MethDescription #class

from batchDefinition.inputRecomSTDefinition import InputRecomSTDefinition #class

from portfolioDescription.aPortfolioDescription import APortfolioDescription #class

from evaluationTool.evalToolDoNothing import EToolDoNothing #class

from recommenderDescription.recommenderDescription import RecommenderDescription #class

from recommender.recommenderCosineCB import RecommenderCosineCB #class

from batchDefinition.inputSimulatorDefinition import InputSimulatorDefinition #class

from simulator.simulator import Simulator #class

from history.historyHierDF import HistoryHierDF #class

import pandas as pd
from batchDefinition.aBatchDefinition import ABatchDefinition #class
from batchDefinition.inputABatchDefinition import InputABatchDefinition

from batchDefinition.aBatchDefinitionST import ABatchDefinitionST #class


class BatchDefSTSingle(ABatchDefinitionST):

    def getBatchName(self):
        return ""
    
    def getParameters(self):

        rIDs, rDescr = InputRecomSTDefinition.exportPairOfRecomIdsAndRecomDescrs()

        rID1:str = "Recom" + "CosineCBcbdOHEupsmeanups1" + "MMR05"
        rID2:str = "Recom" + "CosineCBcbdOHEupsweightedMeanups5" + "MMR05"

        cosineCBMMR1:RecommenderDescription = InputRecomSTDefinition.exportRDescCosineCBcbdOHEupsmeanups1()
        cosineCBMMR1.getArguments()[RecommenderCosineCB.ARG_USE_DIVERSITY] = True
        cosineCBMMR1.getArguments()[RecommenderCosineCB.ARG_MMR_LAMBDA] = 0.5

        cosineCBMMR2:RecommenderDescription = InputRecomSTDefinition.exportRDescCosineCBcbdOHEupsweightedMeanups5()
        cosineCBMMR2.getArguments()[RecommenderCosineCB.ARG_USE_DIVERSITY] = True
        cosineCBMMR2.getArguments()[RecommenderCosineCB.ARG_MMR_LAMBDA] = 0.5

        aDict:Dict[str,object] = dict(zip(rIDs + [rID1, rID2], rDescr + [cosineCBMMR1, cosineCBMMR2]))

        return aDict


    def run(self, batchID:str, jobID:str):

        divisionDatasetPercentualSize:int
        uBehaviour:str
        repetition:int
        divisionDatasetPercentualSize, uBehaviour, repetition = InputABatchDefinition.getBatchParameters(self.datasetID)[batchID]

        rDescr:str = self.getParameters()[jobID]
        recommenderID:str = jobID

        pDescr:APortfolioDescription = Portfolio1MethDescription(self.getBatchName() + recommenderID.title(), recommenderID, rDescr)

        simulator:Simulator = InputSimulatorDefinition.exportSimulatorSlantour(
                batchID, divisionDatasetPercentualSize, uBehaviour, repetition)
        simulator.simulate([pDescr], [DataFrame()], [EToolDoNothing({})], [HistoryHierDF(pDescr.getPortfolioID())])



if __name__ == "__main__":
   os.chdir("..")
   os.chdir("..")
   #print(os.getcwd())

   BatchDefSTSingle.generateAllBatches()