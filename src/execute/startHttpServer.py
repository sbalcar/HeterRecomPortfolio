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
from history.historyHierDF import HistoryHierDF #class

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
from evaluationTool.evalToolDHondtBanditVotes import EvalToolDHondtBanditVotes #class
from evaluationTool.evalToolContext import EvalToolContext  # class

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

from aggregation.negImplFeedback.aPenalization import APenalization #class

from input.inputAggrDefinition import PenalizationToolDefinition #class

from input.inputSimulatorDefinition import InputSimulatorDefinition #class


def startHttpServer():

  np.random.seed(42)
  random.seed(42)

  print("TrainingPortfolios")

  #port1, model1, evalTool1 = getTheMostPopular()
  #port1, model1, evalTool1, history1 = getFuzzyDHont()
  #port2, model2, evalTool2, history2 = getFuzzyDHontINF()
  #port3, model3, evalTool3, history3 = getFuzzyDHontThompsonSamplingINF()
  port4, model4, evalTool4, history4 = getContextFuzzyDHondt()
  port5, model5, evalTool5, history5 = getContextFuzzyDHondtINF()


  portfolios:List[APortfolio] = [port4]
  models:List[DataFrame] = [model4]
  evalTools:List[AEvalTool] = [evalTool4]
  histories:List[AHistory] = [history4]

#  portfolios:List[APortfolio] = [port4, port5]
#  models:List[DataFrame] = [model4, model5]
#  evalTools:List[AEvalTool] = [evalTool4, evalTool5]
#  histories:List[AHistory] = [history4, history5]


  HeterRecomHTTPHandler.initialization(portfolios, models, evalTools, histories, DatasetST)

  print("StartHTTPServer")

  server = HTTPServer(('', 5003), HeterRecomHTTPHandler)
  server.serve_forever()

  print("Serving forever")


def getTheMostPopular():

  taskID:str = "TheMostPopular"
  rDescr:RecommenderDescription = InputRecomSTDefinition.exportRDescTheMostPopular()

  recommenderID:str = "TheMostPopular"
  pDescr:Portfolio1MethDescription = Portfolio1MethDescription(recommenderID.title(), recommenderID, rDescr)

  dataset:ADataset = DatasetST.readDatasets()

  history:AHistory = HistoryHierDF(taskID)
  port:APortfolio = pDescr.exportPortfolio(taskID, history)
  port.train(history, dataset)

  model:DataFrame = DataFrame()
  evalTool:AEvalTool = EToolSingleMethod({})

  return (port, model, evalTool, history)


def getFuzzyDHont():

  #taskID:str = "Web" + "FuzzyDHondt" + "Roulette1"
  taskID:str = "Web" + "FuzzyDHondt" + "Fixed"
  dataset:ADataset = DatasetST.readDatasets()

  #selector:ADHondtSelector = RouletteWheelSelector({RouletteWheelSelector.ARG_EXPONENT: 1})
  selector:ADHondtSelector = TheMostVotedItemSelector({})

  aDescDHont:AggregationDescription = InputAggrDefinition.exportADescDHondt(selector)

  rIDs, rDescs = InputRecomSTDefinition.exportPairOfRecomIdsAndRecomDescrs()

  pDescr:Portfolio1AggrDescription = Portfolio1AggrDescription(
    taskID, rIDs, rDescs, aDescDHont)

  history:AHistory = HistoryHierDF(taskID)

  port:APortfolio = pDescr.exportPortfolio(taskID, history)
  port.train(history, dataset)

  model:DataFrame = ModelDefinition.createDHontModel(pDescr.getRecommendersIDs())

  evalTool:AEvalTool = EvalToolDHondt({EvalToolDHondt.ARG_LEARNING_RATE_CLICKS: 0.03,
                                        EvalToolDHondt.ARG_LEARNING_RATE_VIEWS: 0.03 / 500})

  return (port, model, evalTool, history)



def getFuzzyDHontThompsonSamplingINF():

  taskID:str = "Web" + "FuzzyDHondtThompsonSamplingINF" + "Fixed" + "OLin0802HLin1002"

  selector:ADHondtSelector = TheMostVotedItemSelector({})

  penalization:APenalization = PenalizationToolDefinition.exportProbPenaltyToolOLin0802HLin1002(20)

  aDescDHont:AggregationDescription = InputAggrDefinition.exportADescDHondtThompsonSamplingINF(selector, penalization)

  rIDs, rDescs = InputRecomSTDefinition.exportPairOfRecomIdsAndRecomDescrs()

  pDescr:Portfolio1AggrDescription = Portfolio1AggrDescription(
    taskID, rIDs, rDescs, aDescDHont)

  history:AHistory = HistoryHierDF(taskID)

  dataset:ADataset = DatasetST.readDatasets()

  port:APortfolio = pDescr.exportPortfolio(taskID, history)
  port.train(history, dataset)

  model:DataFrame = ModelDefinition.createDHondtBanditsVotesModel(pDescr.getRecommendersIDs())

  evalTool:AEvalTool = EvalToolDHondtBanditVotes({})

  return (port, model, evalTool, history)



def getFuzzyDHontINF():

  #taskID:str = "Web" + "FuzzyDHondtINF" + "Roulette1"
  taskID:str = "Web" + "FuzzyDHondt" + "Fixed"
  dataset:ADataset = DatasetST.readDatasets()

  #selector:ADHondtSelector = RouletteWheelSelector({RouletteWheelSelector.ARG_EXPONENT: 1})
  selector:ADHondtSelector = TheMostVotedItemSelector({})

  pToolOLin0802HLin1002:APenalization = PenalizationToolDefinition.exportPenaltyToolOLin0802HLin1002(
    InputSimulatorDefinition.numberOfAggrItems)

  aDescDHont:AggregationDescription = InputAggrDefinition.exportADescDHondtINF(selector, pToolOLin0802HLin1002)

  rIDs, rDescs = InputRecomSTDefinition.exportPairOfRecomIdsAndRecomDescrs()

  pDescr:Portfolio1AggrDescription = Portfolio1AggrDescription(
    taskID, rIDs, rDescs, aDescDHont)

  history:AHistory = HistoryHierDF(taskID)

  port:APortfolio = pDescr.exportPortfolio(taskID, history)
  port.train(history, dataset)

  model:DataFrame = ModelDefinition.createDHontModel(pDescr.getRecommendersIDs())

  evalTool:AEvalTool = EvalToolDHondt({EvalToolDHondt.ARG_LEARNING_RATE_CLICKS: 0.03,
                                        EvalToolDHondt.ARG_LEARNING_RATE_VIEWS: 0.03 / 500})

  return (port, model, evalTool, history)


def getContextFuzzyDHondt():

  #taskID:str = "Web" + "ContextFuzzyDHondt" + "Roulette1"
  taskID:str = "Web" + "ContextFuzzyDHondt" + "Fixed"
  dataset:ADataset = DatasetST.readDatasets()

  #selector:ADHondtSelector = RouletteWheelSelector({RouletteWheelSelector.ARG_EXPONENT: 1})
  selector:ADHondtSelector = TheMostVotedItemSelector({})

  #pToolOLin0802HLin1002:APenalization = PenalizationToolDefinition.exportPenaltyToolOLin0802HLin1002(
  #  InputSimulatorDefinition.numberOfAggrItems)
    
  rIDs, rDescs = InputRecomSTDefinition.exportPairOfRecomIdsAndRecomDescrs()

  history:AHistory = HistoryHierDF(taskID)

  # Init eTool
  evalTool:AEvalTool = EvalToolContext({
    EvalToolContext.ARG_ITEMS: dataset.serialsDF,  # ITEMS
    EvalToolContext.ARG_EVENTS: dataset.eventsDF,  # EVENTS (FOR CALCULATING HISTORY OF USER)
    EvalToolContext.ARG_DATASET: "st",  # WHAT DATASET ARE WE IN
    EvalToolContext.ARG_HISTORY: history})  # empty instance of AHistory is OK for ST dataset


  aDescDHont:AggregationDescription = InputAggrDefinition.exportADescDContextHondt(selector, evalTool)


  pDescr:Portfolio1AggrDescription = Portfolio1AggrDescription(
    taskID, rIDs, rDescs, aDescDHont)

  port:APortfolio = pDescr.exportPortfolio(taskID, history)
  port.train(history, dataset)

  model:DataFrame = ModelDefinition.createDHontModel(pDescr.getRecommendersIDs())

  return (port, model, evalTool, history)


def getContextFuzzyDHondtINF():
  # taskID:str = "Web" + "ContextFuzzyDHondtINF" + "Roulette1"
  taskID:str = "Web" + "ContextFuzzyDHondtINF" + "Fixed"
  dataset:ADataset = DatasetST.readDatasets()

  # selector:ADHondtSelector = RouletteWheelSelector({RouletteWheelSelector.ARG_EXPONENT: 1})
  selector:ADHondtSelector = TheMostVotedItemSelector({})

  pToolOLin0802HLin1002:APenalization = PenalizationToolDefinition.exportProbPenaltyToolOLin0802HLin1002(
    InputSimulatorDefinition.numberOfAggrItems)

  rIDs, rDescs = InputRecomSTDefinition.exportPairOfRecomIdsAndRecomDescrs()

  history:AHistory = HistoryHierDF(taskID)

  # Init eTool
  evalTool:AEvalTool = EvalToolContext({
    EvalToolContext.ARG_ITEMS: dataset.serialsDF,  # ITEMS
    EvalToolContext.ARG_EVENTS: dataset.eventsDF,  # EVENTS (FOR CALCULATING HISTORY OF USER)
    EvalToolContext.ARG_DATASET: "st",  # WHAT DATASET ARE WE IN
    EvalToolContext.ARG_HISTORY: history})  # empty instance of AHistory is OK for ST dataset

  aDescDHont:AggregationDescription = InputAggrDefinition.exportADescDContextHondtINF(selector, pToolOLin0802HLin1002, evalTool)

  pDescr:Portfolio1AggrDescription = Portfolio1AggrDescription(
    taskID, rIDs, rDescs, aDescDHont)

  port:APortfolio = pDescr.exportPortfolio(taskID, history)
  port.train(history, dataset)

  model:DataFrame = ModelDefinition.createDHontModel(pDescr.getRecommendersIDs())

  return (port, model, evalTool, history)


if __name__ == "__main__":
    os.chdir("..")

    startHttpServer()