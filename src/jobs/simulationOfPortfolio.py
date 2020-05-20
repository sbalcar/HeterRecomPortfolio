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


def simulationOfPortfolio():

    # dataset reading

    ratingsDF:DataFrame = Ratings.readFromFileMl1m()
    usersDF:DataFrame = Users.readFromFileMl1m()
    itemsDF:DataFrame = Items.readFromFileMl1m()

    numberOfItems:int = 20

    #uBehaviourDesc:UserBehaviourDescription = UserBehaviourDescription(observationalStaticProbabilityFnc, [0.5])
    uBehaviourDesc:UserBehaviourDescription = UserBehaviourDescription(observationalLinearProbabilityFnc, [0.1, 0.9])

    # portfolio definiton
    rDescTheMostPopular:RecommenderDescription = RecommenderDescription(RecommenderTheMostPopular, {})
    rDescDummyRedirector:RecommenderDescription = RecommenderDescription(RecommenderDummyRedirector,
                            {RecommenderDummyRedirector.ARG_RESULT:pd.Series([0.05]*20, list(range(1, 21)), name="rating")} )

    rDescCB:RecommenderDescription = RecommenderDescription(RecommenderCosineCB, {RecommenderCosineCB.ARG_CB_DATA_PATH:Configuration.cbDataFileWithPathTFIDF})
    rDescW2v:RecommenderDescription = RecommenderDescription(RecommenderW2V, {RecommenderW2V.ARG_TRAIN_VARIANT:"all"})

    #rIDs:List[str] = ["RecommenderTheMostPopular", "RecommenderDummyRedirector"]
    #rDescs:List[RecommenderDescription] = [rDescTheMostPopular, rDescDummyRedirector]

    rIDs:List[str] = ["RecommenderCosineCB", "RecommenderW2V"]
    rDescs:List[RecommenderDescription] = [rDescCB, rDescW2v]

    aDescDHont:AggregationDescription = AggregationDescription(AggrDHont, {AggrDHont.ARG_SELECTORFNC:(AggrDHont.selectorOfRouletteWheelRatedItem,[])})
    aDescBanditTS:AggregationDescription = AggregationDescription(AggrBanditTS, {AggrBanditTS.ARG_SELECTORFNC:(AggrBanditTS.selectorOfRouletteWheelRatedItem,[])})



    # DHont Portfolio description
    pDescDHont:APortfolioDescription = Portfolio1AggrDescription(
            "DHont", rIDs, rDescs, aDescDHont)

    modelDHontData:List[str,float] = [[rIdI, 1] for rIdI in pDescDHont.getRecommendersIDs()]
    modelDHontDF:DataFrame = pd.DataFrame(modelDHontData, columns=["methodID", "votes"])
    modelDHontDF.set_index("methodID", inplace=True)

    historyDHont:AHistory = HistoryDF("DHont")



    # BanditTS Portfolio description
    pDescBanditTS:APortfolioDescription = Portfolio1AggrDescription(
            "BanditTS", rIDs, rDescs, aDescBanditTS)

    modelBanditTSData:List = [[rIdI, 1, 1, 1, 1] for rIdI in rIDs]
    modelBanditTSDF:DataFrame = pd.DataFrame(modelBanditTSData, columns=["methodID", "r", "n", "alpha0", "beta0"])
    modelBanditTSDF.set_index("methodID", inplace=True)

    historyBanditTS:AHistory = HistoryDF("BanditTS")



    # TheMostPopular Portfolio description
    pDescTheMostPopular:APortfolioDescription = Portfolio1MethDescription("theMostPopular", "theMostPopular", rDescTheMostPopular)
    modelTheMostPopularDF:DataFrame = pd.DataFrame()
    historyTheMostPopular:AHistory = HistoryDF("TheMostPopular")


    # Cosine CB Portfolio description
    pDescCCB:APortfolioDescription = Portfolio1MethDescription("cosineCB", "cosineCB", rDescCB)
    modelCCBDF:DataFrame = pd.DataFrame()
    historyCCB:AHistory = HistoryDF("cosineCB")


    # W2V Portfolio description
    pDescW2V:APortfolioDescription = Portfolio1MethDescription("w2v", "w2v", rDescW2v)
    modelW2VDF:DataFrame = pd.DataFrame()
    historyW2V:AHistory = HistoryDF("W2V")


    # simulation of portfolio
    #simulator:Simulator = Simulator(SimulationPortfoliosRecomToItemSeparatedUsers, ratingsDF, usersDF, itemsDF, uBehaviourDesc, repetitionOfRecommendation=1, numberOfItems=numberOfItems)
    simulator:Simulator = Simulator(SimulationPortfolioToUser, ratingsDF, usersDF, itemsDF, uBehaviourDesc, repetitionOfRecommendation=1, numberOfItems=numberOfItems)

    #evaluations:List[dict] = simulator.simulate([pDescCCB], [modelCCBDF], [EToolSingleMethod], [historyCCB])
    #evaluations:List[dict] = simulator.simulate([pDescW2V], [modelW2VDF], [EToolSingleMethod], [historyW2V])
    #evaluations:List[dict] = simulator.simulate([pDescTheMostPopular], [modelTheMostPopularDF], [EToolSingleMethod], [historyTheMostPopular])
    evaluations:List[dict] = simulator.simulate([pDescDHont], [modelDHontDF], [EToolDHontHit1], [historyDHont])
    #evaluations:List[dict] = simulator.simulate([pDescBanditTS], [modelBanditTSDF], [EToolBanditTSHit1], [historyBanditTS])
    #evaluations:List[dict] = simulator.simulate([pDescDHont, pDescBanditTS], [modelDHontDF, modelBanditTSDF], [EToolDHontHit1, EToolBanditTSHit1], [historyDHont, historyBanditTS])




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

        print()
        print("Evaluations: " + str(evaluations))
        return evaluations