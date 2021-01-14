#!/usr/bin/python3

import time
import sys
import os

import random
import traceback
import numpy as np

from typing import List
from typing import Dict

from pandas import DataFrame

from input.batchesML1m.batchMLBanditTS import BatchMLBanditTS #class
from input.batchesML1m.batchMLFuzzyDHondt import BatchMLFuzzyDHondt #class
from input.batchesML1m.batchMLDHondtThompsonSampling import BatchMLDHondtThompsonSampling #class
from input.batchesML1m.batchMLFuzzyDHondtINF import BatchMLFuzzyDHondtINF #class
from input.batchesML1m.batchMLDHondtThompsonSamplingINF import BatchMLDHondtThompsonSamplingINF #class
from input.batchesML1m.batchMLFuzzyDHondtDirectOptimize import BatchMLFuzzyDHondtDirectOptimize #class
from input.batchesML1m.batchMLFuzzyDHondtDirectOptimizeINF import BatchMLFuzzyDHondtDirectOptimizeINF #class
from input.batchesML1m.batchMLSingle import BatchMLSingle #class
from input.batchesML1m.batchMLSingleINF import BatchMLSingleINF #class
from input.batchesRetailrocket.batchRRSingle import BatchRRSingle #class

from input.batchesML1m.batchMLSingle2 import BatchMLSingle2 #class
from input.batchesML1m.batchMLSingleBPRMFHT import BatchMLSingleBPRMFHT #class
from input.batchesML1m.batchMLSingleW2VHT import BatchMLSingleW2VHT #class
from input.batchesML1m.batchMLSingleCosineCBHT import BatchMLSingleCosineCBHT #class

from input.batchesRetailrocket.batchRRSingle import BatchRRSingle #class
from input.batchesRetailrocket.batchRRSingleW2VHT import BatchRRSingleW2VHT #class

from input.batchesSlanTour.batchSTSingle import BatchSTSingle #class
from input.batchesSlanTour.batchSTSingleBPRMFHT import BatchSTSingleBPRMFHT #class
from input.batchesSlanTour.batchSTSingleW2VHT import BatchSTSingleW2VHT #class
from input.batchesSlanTour.batchSTSingleCosineCBHT import BatchSTSingleCosineCBHT #class

from history.aHistory import AHistory #class
from history.historyDF import HistoryDF #class

from datasets.aDataset import ADataset #class
from datasets.datasetML import DatasetML
from datasets.datasetST import DatasetST #class

from recommenderDescription.recommenderDescription import RecommenderDescription #class

from recommender.aRecommender import ARecommender #class
from recommender.recommenderTheMostPopular import RecommenderTheMostPopular #class
from recommender.recommenderCosineCB import RecommenderCosineCB #class
from recommender.recommenderW2V import RecommenderW2V #class
from recommender.recommenderItemBasedKNN import RecommenderItemBasedKNN #class
from recommender.recommenderBPRMF import RecommenderBPRMF #class

from portfolio.aPortfolio import APortfolio #class
from portfolioDescription.portfolio1MethDescription import Portfolio1MethDescription #class
from portfolioDescription.portfolio1AggrDescription import Portfolio1AggrDescription #class

from evaluationTool.aEvalTool import AEvalTool #class
from evaluationTool.evalToolSingleMethod import EToolSingleMethod #class

from httpServer.httpServer import HeterRecomHTTPHandler #class
from http.server import BaseHTTPRequestHandler, HTTPServer

from input.inputAggrDefinition import InputAggrDefinition  # class
from input.inputRecomSTDefinition import InputRecomSTDefinition #class
from input.modelDefinition import ModelDefinition

from aggregationDescription.aggregationDescription import AggregationDescription #class

from evaluationTool.evalToolDHondt import EvalToolDHondt #class

from aggregation.operators.aDHondtSelector import ADHondtSelector #class
from aggregation.operators.rouletteWheelSelector import RouletteWheelSelector #class
from aggregation.operators.theMostVotedItemSelector import TheMostVotedItemSelector #class


def startHttpServer():

  np.random.seed(42)
  random.seed(42)

  print("StartHTTPServer")

  #portA, modelA, evalToolA = getTheMostPopular()
  portA, modelA, evalToolA = getFuzzyDHont()

  portfolioDict:Dict[str,APortfolio] = {HeterRecomHTTPHandler.VARIANT_A:portA}
  modelsDict:Dict[str,int] = {HeterRecomHTTPHandler.VARIANT_A:modelA}
  evalToolsDict:Dict[str, AEvalTool] = {HeterRecomHTTPHandler.VARIANT_A:evalToolA}


  HeterRecomHTTPHandler.portfolioDict = portfolioDict
  HeterRecomHTTPHandler.modelsDict = modelsDict
  HeterRecomHTTPHandler.evalToolsDict = evalToolsDict
  HeterRecomHTTPHandler.evaluation:Dict = {}
  #HeterRecomHTTPHandler.datasetClass = DatasetML
  HeterRecomHTTPHandler.datasetClass = DatasetST

  server = HTTPServer(('', 8080), HeterRecomHTTPHandler)
  server.serve_forever()



def getTheMostPopular():

  rDescr:RecommenderDescription = RecommenderDescription(RecommenderTheMostPopular, {})

  recommenderID:str = "TheMostPopular"
  pDescr:Portfolio1MethDescription = Portfolio1MethDescription(recommenderID.title(), recommenderID, rDescr)

  dataset:ADataset = DatasetST.readDatasets()

  history:AHistory = HistoryDF("test")
  port:APortfolio = pDescr.exportPortfolio("jobID", history)
  port.train(history, dataset)

  model:DataFrame = DataFrame()
  evalTool:AEvalTool = EToolSingleMethod({})

  return (port, model, evalTool)

def getFuzzyDHont():

  dataset:ADataset = DatasetST.readDatasets()

  selector:ADHondtSelector = RouletteWheelSelector({RouletteWheelSelector.ARG_EXPONENT: 1})

  aDescDHont:AggregationDescription = InputAggrDefinition.exportADescDHont(selector)

  rIDs, rDescs = InputRecomSTDefinition.exportPairOfRecomIdsAndRecomDescrs()

  pDescr:Portfolio1AggrDescription = Portfolio1AggrDescription(
    "FuzzyDHondt", rIDs, rDescs, aDescDHont)

  history:AHistory = HistoryDF("test")

  port:APortfolio = pDescr.exportPortfolio("jobID", history)
  port.train(history, dataset)

  model:DataFrame = ModelDefinition.createDHontModel(pDescr.getRecommendersIDs())

  evalTool:AEvalTool = EvalToolDHondt({EvalToolDHondt.ARG_LEARNING_RATE_CLICKS: 0.03,
                                        EvalToolDHondt.ARG_LEARNING_RATE_VIEWS: 0.03 / 500})

  return (port, model, evalTool)




if __name__ == "__main__":
    os.chdir("..")

    startHttpServer()