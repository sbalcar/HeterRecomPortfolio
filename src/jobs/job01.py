#!/usr/bin/python3

from typing import List

from recommenderDescription.recommenderDescription import RecommenderDescription #class

from recommender.dummy.recommenderTheMostPopular import RecommenderTheMostPopular #class
from recommender.dummy.recommenderDummyRedirector import RecommenderDummyRedirector #class
from recommender.recommenderCosineCB import RecommenderCosineCB #class
from recommender.recommenderW2V import RecommenderW2V #class

from datasets.ratings import Ratings #class
from datasets.users import Users #class
from datasets.items import Items #class
from datasets.configuration import Configuration #class

from portfolioDescription.portfolio1MethDescription import Portfolio1MethDescription #class
from portfolioDescription.portfolio1AggrDescription import Portfolio1AggrDescription #class

from aggregationDescription.aggregationDescription import AggregationDescription #class
from aggregation.aggrDHont import AggrDHont #class
from aggregation.aggrBanditTS import AggrBanditTS #class

import pandas as pd

from pandas.core.frame import DataFrame #class

from simulator.simulator import Simulator #class

from simulation.recommendToItem.simulatorOfPortfoliosRecomToItemSeparatedUsers import SimulationPortfoliosRecomToItemSeparatedUsers #class
from simulation.recommendToUser.simulatorOfPortfoliosRecommToUser import SimulationPortfolioToUser #class

from history.aHistory import AHistory #class
from history.historyDF import HistoryDF #class
from history.historyHierDF import HistoryHierDF #class

from portfolioDescription.aPortfolioDescription import APortfolioDescription #class

from evaluationTool.aEvalTool import AEvalTool #class
from evaluationTool.singleMethod.eToolSingleMethod import EToolSingleMethod #class
from evaluationTool.dHont.eToolDHontHit1 import EToolDHontHit1 #class
from evaluationTool.banditTS.eToolBanditTSHit1 import EToolBanditTSHit1 #class

from userBehaviourDescription.userBehaviourDescription import UserBehaviourDescription #class
from userBehaviourDescription.userBehaviourDescription import observationalLinearProbabilityFnc #function

from input.inputsML1MDefinition import InputsML1MDefinition #class


def job01():

        d = InputsML1MDefinition

        argsSimulationDict:dict = {SimulationPortfolioToUser.ARG_WINDOW_SIZE:3, SimulationPortfolioToUser.ARG_REPETITION_OF_RECOMMENDATION:1, SimulationPortfolioToUser.ARG_NUMBER_OF_ITEMS:20}

        # simulation of portfolio
        #simulator:Simulator = Simulator(SimulationPortfoliosRecomToItemSeparatedUsers, ratingsDF, usersDF, itemsDF, uBehaviourDesc
        simulator:Simulator = Simulator(SimulationPortfolioToUser, argsSimulationDict, d.ratingsDF, d.usersDF, d.itemsDF, d.uBehaviourDesc)

        #evaluations:List[dict] = simulator.simulate([pDescCCB], [modelCCBDF], [EToolSingleMethod], [historyCCB])
        #evaluations:List[dict] = simulator.simulate([pDescW2V], [modelW2VDF], [EToolSingleMethod], [historyW2V])
        #evaluations:List[dict] = simulator.simulate([pDescTheMostPopular], [modelTheMostPopularDF], [EToolSingleMethod], [historyTheMostPopular])
        #evaluations:List[dict] = simulator.simulate([pDescDHont], [modelDHontDF], [EToolDHontHit1], [historyDHont])
        #evaluations:List[dict] = simulator.simulate([pDescBanditTS], [modelBanditTSDF], [EToolBanditTSHit1], [historyBanditTS])
        evaluations:List[dict] = simulator.simulate([d.pDescDHont, d.pDescBanditTS], [d.modelDHontDF, d.modelBanditTSDF], [EToolDHontHit1, EToolBanditTSHit1], [d.historyDHont, d.historyBanditTS])
