#!/usr/bin/python3

import os

from portfolio.portfolioDynamic import PortfolioDynamic #class

import pandas as pd
from pandas import DataFrame
from typing import List
from typing import Dict #class

from datasets.datasetRetailRocket import DatasetRetailRocket #class
from datasets.retailrocket.behavioursRR import BehavioursRR #class

from portfolioDescription.aPortfolioDescription import APortfolioDescription #class
from portfolioDescription.portfolio1AggrDescription import Portfolio1AggrDescription #class

from simulator.simulator import Simulator #class
from simulation.simulationRR import SimulationRR #class

from history.aHistory import AHistory #class
from history.historyHierDF import HistoryHierDF #class

from batchDefinition.inputAggrDefinition import InputAggrDefinition #class
from batchDefinition.inputRecomRRDefinition import InputRecomRRDefinition #class

from evaluationTool.aEvalTool import AEvalTool #class

from aggregation.operators.aDHondtSelector import ADHondtSelector #class
from aggregation.operators.rouletteWheelSelector import RouletteWheelSelector #class
from aggregation.operators.theMostVotedItemSelector import TheMostVotedItemSelector #class

from aggregation.negImplFeedback.aPenalization import APenalization #class
from batchDefinition.inputAggrDefinition import PenalizationToolDefinition #class

from batchDefinition.inputSimulatorDefinition import InputSimulatorDefinition #class

from aggregationDescription.aggregationDescription import AggregationDescription #class

from aggregation.aggrD21 import AggrD21 #class

from portfolioModel.pModelBandit import PModelBandit #class
from portfolioModel.pModelDHondtBanditsVotes import PModelDHondtBanditsVotes #class
from portfolioModel.pModelDHondt import PModelDHondt #class
from portfolioModel.pModelDHondtPersonalisedStat import PModelDHondtPersonalisedStat #class

from evaluationTool.aEvalTool import AEvalTool  #class
from evaluationTool.evalToolBanditTS import EvalToolBanditTS  #class
from evaluationTool.evalToolDoNothing import EToolDoNothing  #class
from evaluationTool.evalToolDHondtBanditVotes import EvalToolDHondtBanditVotes  #class
from evaluationTool.evalToolDHondtPersonal import EvalToolDHondtPersonal  #class
from evaluationTool.evalToolDHondt import EvalToolDHondt  #class

from portfolioDescription.portfolioDynamicDescription import PortfolioDynamicDescription  #class

from recommenderDescription.recommenderDescription import RecommenderDescription #class

from recommender.aRecommender import ARecommender #class
from recommender.recommenderTheMostPopular import RecommenderTheMostPopular #class
from recommender.recommenderCosineCB import RecommenderCosineCB #class
from recommender.recommenderW2V import RecommenderW2V #class
from recommender.recommenderItemBasedKNN import RecommenderItemBasedKNN #class
from recommender.recommenderBPRMFImplicit import RecommenderBPRMFImplicit #class

argsSimulationDict:Dict[str,object] = {SimulationRR.ARG_WINDOW_SIZE: 5,
                            SimulationRR.ARG_RECOM_REPETITION_COUNT: 1,
                            SimulationRR.ARG_NUMBER_OF_RECOMM_ITEMS: 100,
                            SimulationRR.ARG_NUMBER_OF_AGGR_ITEMS: InputSimulatorDefinition.numberOfAggrItems,
                            SimulationRR.ARG_DIV_DATASET_PERC_SIZE: 90,
                            SimulationRR.ARG_HISTORY_LENGTH: 10}


def test01():

    print("Simulation: RR Dynamic")

    lrClick:float = 0.03
    #lrView:float = lrClick / 300
    lrViewDivisor:float = 250

    jobID:str = "Fixed" + "Clk" + str(lrClick).replace(".", "") + "ViewDivisor" + str(lrViewDivisor).replace(".", "")

    selector:ADHondtSelector = TheMostVotedItemSelector({})


    rIDs, rDescs = InputRecomRRDefinition.exportPairOfRecomIdsAndRecomDescrs()

    p1AggrDescr:Portfolio1AggrDescription = Portfolio1AggrDescription(
        "FDHont" + jobID, rIDs, rDescs, InputAggrDefinition.exportADescDHondt(selector))

    recommenderID:str = "TheMostPopular"
    rDescr:RecommenderDescription = RecommenderDescription(RecommenderTheMostPopular, {})


    pDescr:APortfolioDescription = PortfolioDynamicDescription(
        "Dynamic" + "FDHontPersStat" + jobID, recommenderID, rDescr, "FDHondt",  p1AggrDescr)


    batchID:str = "rrDiv90Ulinear0109R1"
    dataset:DatasetRetailRocket = DatasetRetailRocket.readDatasetsWithFilter(minEventCount=50)
    behaviourFile:str = BehavioursRR.getFile(BehavioursRR.BHVR_LINEAR0109)
    behavioursDF:DataFrame = BehavioursRR.readFromFileRR(behaviourFile)

    model:DataFrame = PModelDHondtPersonalisedStat(p1AggrDescr.getRecommendersIDs())

    eTool:AEvalTool = EvalToolDHondtPersonal({EvalToolDHondt.ARG_LEARNING_RATE_CLICKS: lrClick,
                                            EvalToolDHondt.ARG_LEARNING_RATE_VIEWS: lrClick / lrViewDivisor})


    # simulation of portfolio
    simulator:Simulator = Simulator(batchID, SimulationRR, argsSimulationDict, dataset, behavioursDF)
    simulator.simulate([pDescr], [model], [eTool], [HistoryHierDF(pDescr.getPortfolioID())])






if __name__ == "__main__":
    os.chdir("..")

    test01()
