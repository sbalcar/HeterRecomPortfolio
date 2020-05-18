#!/usr/bin/python3

from typing import List

from recommenderDescription.recommenderDescription import RecommenderDescription #class

from recommender.recommenderTheMostPopular import RecommenderTheMostPopular #class
from recommender.dummy.recommenderDummyRedirector import RecommenderDummyRedirector #class

from datasets.ratings import Ratings #class

from datasets.users import Users #class

from datasets.items import Items #class

from portfolioDescription.portfolio1MethDescription import Portfolio1MethDescription #class
from portfolioDescription.portfolio1AggrDescription import Portfolio1AggrDescription #class

from aggregationDescription.aggregationDescription import AggregationDescription #class
from aggregation.aggrDHont import AggrDHont #class
from aggregation.aggrBanditTS import AggrBanditTS #class

import pandas as pd

from pandas.core.frame import DataFrame #class

from simulation.simulationOfNonPersonalisedPortfolio import SimulationOfNonPersonalisedPortfolio #class
from simulation.simulationOfPersonalisedPortfolio import SimulationOfPersonalisedPortfolio #class

from history.aHistory import AHistory #class
from history.historyDF import HistoryDF #class

from portfolioDescription.aPortfolioDescription import APortfolioDescription #class

from evaluationTool.singleMethod.eToolSingleMethod import EToolSingleMethod #class
from evaluationTool.dHont.eToolDHontHit1 import EToolDHontHit1 #class
from evaluationTool.banditTS.eToolBanditTSHit1 import EToolBanditTSHit1 #class

from userBehaviourDescription.userBehaviourDescription import UserBehaviourDescription #class
from userBehaviourDescription.userBehaviourDescription import observationalStaticProbabilityFnc #function
from userBehaviourDescription.userBehaviourDescription import observationalLinearProbabilityFnc #function


def simulationOfPortfolio():

    # dataset reading
    ratingsDF: DataFrame = Ratings.readFromFileMl100k()
    usersDF: DataFrame = Users.readFromFileMl100k()
    itemsDF: DataFrame = Items.readFromFileMl100k()

    #ratingsDF: DataFrame = Ratings.readFromFileMl1m()
    #usersDF: DataFrame = Users.readFromFileMl1m()
    #itemsDF: DataFrame = Items.readFromFileMl1m()

    #ratingsDF: DataFrame = Ratings.readFromFileMl10M100K()
    #usersDF: DataFrame = Users.readFromFileMl10M100K()
    #itemsDF: DataFrame = Items.readFromFileMl10M100K()

    numberOfItems:int = 20

    #uBehaviourDesc:UserBehaviourDescription = UserBehaviourDescription(observationalStaticProbabilityFnc, [0.5])
    uBehaviourDesc:UserBehaviourDescription = UserBehaviourDescription(observationalLinearProbabilityFnc, [0.1, 0.9])

    # portfolio definiton
    rDescTheMostPopular:RecommenderDescription = RecommenderDescription(RecommenderTheMostPopular, {})
    rDescDummyRedirector:RecommenderDescription = RecommenderDescription(RecommenderDummyRedirector,
                            {RecommenderDummyRedirector.ARG_RESULT:pd.Series([0.05]*20, list(range(1, 21)), name="rating")} )

    rIDs:List[str] = ["RecommenderTheMostPopular", "RecommenderDummyRedirector"]
    rDescs:List[RecommenderDescription] = [rDescTheMostPopular, rDescDummyRedirector]

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



    # simulation of portfolio
    simulation:SimulationOfPersonalisedPortfolio = SimulationOfPersonalisedPortfolio(
    #simulation:SimulationOfNonPersonalisedPortfolio = SimulationOfNonPersonalisedPortfolio(
            ratingsDF, usersDF, itemsDF, uBehaviourDesc, repetitionOfRecommendation=1, numberOfItems=numberOfItems)

    #evaluations:List[dict] = simulation.run([pDescTheMostPopular], [modelTheMostPopularDF], [EToolSingleMethod], [historyTheMostPopular])
    #evaluations:List[dict] = simulation.run([pDescDHont], [modelDHontDF], [EToolDHontHitIncrementOfResponsibility], [historyDHont])
    #evaluations:List[dict] = simulation.run([pDescBanditTS], [modelBanditTSDF], [EToolBanditTSHit1], [historyBanditTS])
    evaluations:List[dict] = simulation.run([pDescDHont, pDescBanditTS], [modelDHontDF, modelBanditTSDF], [EToolDHontHit1, EToolBanditTSHit1], [historyDHont, historyBanditTS])


    print("Evaluations: " + str(evaluations))

    #print(historyTheMostPopular.print())
    print(historyBanditTS.print())