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
from configuration.configuration import Configuration #class

from portfolioDescription.portfolio1MethDescription import Portfolio1MethDescription #class
from portfolioDescription.portfolio1AggrDescription import Portfolio1AggrDescription #class

from aggregationDescription.aggregationDescription import AggregationDescription #class
from aggregation.aggrBanditTS import AggrBanditTS #class
from aggregation.aggrDHont import AggrDHont #class
from aggregation.aggrDHontNegativeImplFeedback import AggrDHontNegativeImplFeedback #class


import pandas as pd

from pandas.core.frame import DataFrame #class

from history.aHistory import AHistory #class
from history.historyHierDF import HistoryHierDF #class

from portfolioDescription.aPortfolioDescription import APortfolioDescription #class

from userBehaviourDescription.userBehaviourDescription import UserBehaviourDescription #class
from userBehaviourDescription.userBehaviourDescription import observationalLinearProbabilityFnc #function



class InputsML1MDefinition:

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
                            {RecommenderDummyRedirector.ARG_RESULT:pd.Series([0.05]*20, index=list(range(1, 21)))} )

    rDescCB:RecommenderDescription = RecommenderDescription(RecommenderCosineCB, {RecommenderCosineCB.ARG_CB_DATA_PATH:Configuration.cbDataFileWithPathTFIDF, RecommenderCosineCB.ARG_USER_PROFILE_STRATEGY:"mean"})
    rDescW2v:RecommenderDescription = RecommenderDescription(RecommenderW2V, {RecommenderW2V.ARG_TRAIN_VARIANT:"all", RecommenderW2V.ARG_USER_PROFILE_STRATEGY:"max"})

    #rIDs:List[str] = ["RecommenderTheMostPopular", "RecommenderDummyRedirector"]
    #rDescs:List[RecommenderDescription] = [rDescTheMostPopular, rDescDummyRedirector]

    rIDs:List[str] = ["RecommenderCosineCB", "RecommenderW2V"]
    rDescs:List[RecommenderDescription] = [rDescCB, rDescW2v]

    aDescBanditTS:AggregationDescription = AggregationDescription(AggrBanditTS,
                            {AggrBanditTS.ARG_SELECTORFNC:(AggrBanditTS.selectorOfRouletteWheelRatedItem,[])})
    aDescDHont:AggregationDescription = AggregationDescription(AggrDHont,
                            {AggrDHont.ARG_SELECTORFNC:(AggrDHont.selectorOfRouletteWheelRatedItem,[])})
    aDescDHontNF:AggregationDescription = AggregationDescription(AggrDHontNegativeImplFeedback,
                            {AggrDHontNegativeImplFeedback.ARG_SELECTORFNC:(AggrDHont.selectorOfRouletteWheelRatedItem,[]),
                             AggrDHontNegativeImplFeedback.ARG_MAX_PENALTY_VALUE:0.8,
                             AggrDHontNegativeImplFeedback.ARG_MIN_PENALTY_VALUE:0.2,
                             AggrDHontNegativeImplFeedback.ARG_LENGTH_OF_HISTORY:10})



    # TheMostPopular Portfolio description
    pDescTheMostPopular:APortfolioDescription = Portfolio1MethDescription("theMostPopular", "theMostPopular", rDescTheMostPopular)
    modelTheMostPopularDF:DataFrame = pd.DataFrame()
    historyTheMostPopular:AHistory = HistoryHierDF("TheMostPopular")


    # Cosine CB Portfolio description
    pDescCCB:APortfolioDescription = Portfolio1MethDescription("cosineCB", "cosineCB", rDescCB)
    modelCCBDF:DataFrame = pd.DataFrame()
    historyCCB:AHistory = HistoryHierDF("cosineCB")


    # W2V Portfolio description
    pDescW2V:APortfolioDescription = Portfolio1MethDescription("w2v", "w2v", rDescW2v)
    modelW2VDF:DataFrame = pd.DataFrame()
    historyW2V:AHistory = HistoryHierDF("W2V")


    # BanditTS Portfolio description
    pDescBanditTS:APortfolioDescription = Portfolio1AggrDescription(
            "BanditTS", rIDs, rDescs, aDescBanditTS)
    modelBanditTSData:List = [[rIdI, 1, 1, 1, 1] for rIdI in rIDs]
    modelBanditTSDF:DataFrame = pd.DataFrame(modelBanditTSData, columns=["methodID", "r", "n", "alpha0", "beta0"])
    modelBanditTSDF.set_index("methodID", inplace=True)
    historyBanditTS:AHistory = HistoryHierDF("BanditTS")


    # DHont Portfolio description
    pDescDHont:APortfolioDescription = Portfolio1AggrDescription(
            "DHont", rIDs, rDescs, aDescDHont)
    modelDHontData:List[List] = [[rIdI, 1] for rIdI in pDescDHont.getRecommendersIDs()]
    modelDHontDF:DataFrame = pd.DataFrame(modelDHontData, columns=["methodID", "votes"])
    modelDHontDF.set_index("methodID", inplace=True)
    historyDHont:AHistory = HistoryHierDF("DHont")


    # DHont Negative Implicit Feedback Portfolio description
    pDescDHontNF:APortfolioDescription = Portfolio1AggrDescription(
            "DHontNF", rIDs, rDescs, aDescDHontNF)
    modelDHontNFData:List[List] = [[rIdI, 1] for rIdI in pDescDHontNF.getRecommendersIDs()]
    modelDHontNFDF:DataFrame = pd.DataFrame(modelDHontNFData, columns=["methodID", "votes"])
    modelDHontNFDF.set_index("methodID", inplace=True)
    historyDHontNF:AHistory = HistoryHierDF("DHontNF")




