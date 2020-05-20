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

from simulation.recommendToItem.simulatorOfPortfoliosRecomToItemSeparatedUsers import SimulationPortfoliosRecomToItemSeparatedUsers #class
from simulation.recommendToUser.simulatorOfPortfoliosRecommToUser import SimulationPortfolioToUser #class

from history.aHistory import AHistory #class
from history.historyDF import HistoryDF #class

from portfolioDescription.aPortfolioDescription import APortfolioDescription #class

from evaluationTool.aEvalTool import AEvalTool #class
from evaluationTool.singleMethod.eToolSingleMethod import EToolSingleMethod #class
from evaluationTool.dHont.eToolDHontHit1 import EToolDHontHit1 #class
from evaluationTool.banditTS.eToolBanditTSHit1 import EToolBanditTSHit1 #class

from userBehaviourDescription.userBehaviourDescription import UserBehaviourDescription #class
from userBehaviourDescription.userBehaviourDescription import observationalLinearProbabilityFnc #function


class Simulator:

    def __init__(self, simulatorClass, ratingsDF:DataFrame, usersDF:DataFrame, itemsDF:DataFrame, uBehaviourDesc:UserBehaviourDescription, repetitionOfRecommendation:int=1, numberOfItems:int=20):
        if type(ratingsDF) is not DataFrame:
            raise ValueError("Argument ratingsDF isn't type DataFrame.")
        if type(usersDF) is not DataFrame:
            raise ValueError("Argument usersDF isn't type DataFrame.")
        if type(itemsDF) is not DataFrame:
            raise ValueError("Argument itemsDF isn't type DataFrame.")
        if type(itemsDF) is not DataFrame:
            raise ValueError("Argument itemsDF isn't type DataFrame.")
        if type(uBehaviourDesc) is not UserBehaviourDescription:
            raise ValueError("Argument uBehaviourDesc isn't type UserBehaviourDescription.")
        if type(repetitionOfRecommendation) is not int:
            raise ValueError("Argument repetitionOfRecommendation isn't type int.")
        if type(numberOfItems) is not int:
            raise ValueError("Argument numberOfItems isn't type int.")

        self._simulation:SimulationPortfolioToUser = simulatorClass(
            ratingsDF, usersDF, itemsDF, uBehaviourDesc, repetitionOfRecommendation=repetitionOfRecommendation, numberOfItems=numberOfItems)


    def simulate(self, pDescs:List[APortfolioDescription], portModels:List[DataFrame], eTools:List[AEvalTool], histories:List[AHistory]):

        if type(pDescs) is not list:
            raise ValueError("Argument histories isn't type list.")
        for pDescI in pDescs:
            if not isinstance(pDescI, APortfolioDescription):
               raise ValueError("Argument pDescs don't contain APortfolioDescription.")

        if type(portModels) is not list:
            raise ValueError("Argument portModels isn't type list.")
        for portModI in portModels:
            if type(portModI) is not DataFrame:
               raise ValueError("Argument portModels don't contain DataFrame.")

        if type(eTools) is not list:
            raise ValueError("Argument etools isn't type list.")

        if type(histories) is not list:
            raise ValueError("Argument histories isn't type list.")
        for historyI in histories:
            if not isinstance(historyI, AHistory):
               raise ValueError("Argument histories don't contain AHistory.")


        evaluations:List[dict] = self._simulation.run(pDescs, portModels, eTools, histories)

        i:int
        for i in range(len(pDescs)):

            pDescI:APortfolioDescription = pDescs[i]

            print()
            portfolioIdI:str = pDescI.getPortfolioID()
            print("PortfolioIdI: " + portfolioIdI)

            print()
            eToolClassI:str = eTools[i]
            print("EToolClass " + portfolioIdI)
            print(eToolClassI)

            print()
            portModelI:DataFrame = portModels[i]
            print("Model of " + portfolioIdI)
            print(portModelI)

            print()
            historyI:AHistory = histories[i]
            print("History of " + portfolioIdI)
            historyI.print()

        ids:List[str] = [pDescI.getPortfolioID() for pDescI in pDescs]

        print()
        print("ids: " + str(ids))
        print("Evaluations: " + str(evaluations))
        return evaluations