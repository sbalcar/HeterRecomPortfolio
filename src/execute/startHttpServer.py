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
from input.batchesML1m.batchFuzzyDHondtINF import BatchFuzzyDHondtINF #class
from input.batchesML1m.batchMLDHondtThompsonSamplingINF import BatchMLDHondtThompsonSamplingINF #class
from input.batchesML1m.batchMLFuzzyDHondtDirectOptimize import BatchMLFuzzyDHondtDirectOptimize #class
from input.batchesML1m.batchFuzzyDHondtDirectOptimizeINF import BatchFuzzyDHondtDirectOptimizeINF #class
from input.batchesML1m.batchMLSingle import BatchMLSingle #class
from input.batchesML1m.batchMLSingleINF import BatchMLSingleINF #class
from input.batchesRetailrocket.batchRRSingle import BatchRRSingle #class

from input.batchesML1m.batchMLSingle2 import BatchMLSingle2 #class
from input.batchesML1m.batchMLSingleBPRMFHT import BatchMLSingleBPRMFHT #class
from input.batchesML1m.batchMLSingleW2VHT import BatchMLSingleW2VHT #class
from input.batchesML1m.batchMLSingleCosineCBHT import BatchMLSingleCosineCBHT #class

from input.batchesRetailrocket.batchRRSingle import BatchRRSingle #class
from input.batchesRetailrocket.batchRRSingleW2VHT import BatchRRSingleW2VHT #class

from input.batchSlanTour.batchSTSingle import BatchSTSingle #class
from input.batchSlanTour.batchSTSingleBPRMFHT import BatchSTSingleBPRMFHT #class
from input.batchSlanTour.batchSTSingleW2VHT import BatchSTSingleW2VHT #class
from input.batchSlanTour.batchSTSingleCosineCBHT import BatchSTSingleCosineCBHT #class

from history.aHistory import AHistory #class
from history.historyDF import HistoryDF #class

from datasets.aDataset import ADataset #class
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

from evaluationTool.aEvalTool import AEvalTool #class
from evaluationTool.evalToolSingleMethod import EToolSingleMethod #class

from httpServer.httpServer import HeterRecomHTTPHandler #class
from http.server import BaseHTTPRequestHandler, HTTPServer



def startHttpServer():

  np.random.seed(42)
  random.seed(42)

  print("StartHTTPServer")


  rDescr:RecommenderDescription = RecommenderDescription(RecommenderTheMostPopular, {})

  recommenderID:str = "TheMostPopular"
  pDescr:Portfolio1MethDescription = Portfolio1MethDescription(recommenderID.title(), recommenderID, rDescr)

  dataset:ADataset = DatasetST.readDatasets()

  history:AHistory = HistoryDF("test")
  p:APortfolio = pDescr.exportPortfolio("jobID", history)
  p.train(history, dataset)


  portfolioDict:Dict[str,APortfolio] = {HeterRecomHTTPHandler.VARIANT_A:p}
  modelsDict:Dict[str,int] = {HeterRecomHTTPHandler.VARIANT_A:DataFrame()}
  evalToolsDict:Dict[str, AEvalTool] = {HeterRecomHTTPHandler.VARIANT_A:EToolSingleMethod({})}


  HeterRecomHTTPHandler.portfolioDict = portfolioDict
  HeterRecomHTTPHandler.modelsDict = modelsDict
  HeterRecomHTTPHandler.evalToolsDict = evalToolsDict

  server = HTTPServer(('', 8080), HeterRecomHTTPHandler)
  server.serve_forever()



if __name__ == "__main__":
    os.chdir("..")

    startHttpServer()