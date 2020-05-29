#!/usr/bin/python3

from typing import List

from recommenderDescription.recommenderDescription import RecommenderDescription #class

from recommender.recommenderTheMostPopular import RecommenderTheMostPopular #class
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

from evaluationTool.evalToolDHont import EvalToolDHont #class

import pandas as pd

from pandas.core.frame import DataFrame #class

from history.aHistory import AHistory #class
from history.historyHierDF import HistoryHierDF #class

from portfolioDescription.aPortfolioDescription import APortfolioDescription #class

from userBehaviourDescription.userBehaviourDescription import UserBehaviourDescription #class
from userBehaviourDescription.userBehaviourDescription import observationalStaticProbabilityFnc #function


from aggregation.toolsDHontNF.penalizationOfResultsByNegImpFeedback.aPenalization import APenalization #class
from aggregation.toolsDHontNF.penalizationOfResultsByNegImpFeedback.penalUsingFiltering import PenalUsingFiltering #class
from aggregation.toolsDHontNF.penalizationOfResultsByNegImpFeedback.penalUsingReduceRelevance import PenalUsingReduceRelevance #class
from aggregation.toolsDHontNF.penalizationOfResultsByNegImpFeedback.penalUsingReduceRelevance import penaltyStatic #function
from aggregation.toolsDHontNF.penalizationOfResultsByNegImpFeedback.penalUsingReduceRelevance import penaltyLinear #function


class Tools:

    def createDHontModel(recommendersIDs: List[str]):
        modelDHontData: List[List] = [[rIdI, 1] for rIdI in recommendersIDs]
        modelDHontDF: DataFrame = pd.DataFrame(modelDHontData, columns=["methodID", "votes"])
        modelDHontDF.set_index("methodID", inplace=True)
        EvalToolDHont.linearNormalizingPortfolioModelDHont(modelDHontDF)
        return modelDHontDF

    def createBanditModel(recommendersIDs:List[str]):
        modelBanditTSData:List = [[rIdI, 1, 1, 1, 1] for rIdI in recommendersIDs]
        modelBanditTSDF:DataFrame = pd.DataFrame(modelBanditTSData, columns=["methodID", "r", "n", "alpha0", "beta0"])
        modelBanditTSDF.set_index("methodID", inplace=True)
        return modelBanditTSDF

class InputsML1MDefinition:

    # dataset reading
    ratingsDF:DataFrame = Ratings.readFromFileMl1m()
    usersDF:DataFrame = Users.readFromFileMl1m()
    itemsDF:DataFrame = Items.readFromFileMl1m()

    numberOfRecommItems:int = 100
    numberOfAggrItems:int = 20

    #uBehaviourDesc:UserBehaviourDescription = UserBehaviourDescription(observationalLinearProbabilityFnc, [0.1, 0.9])
    uBehaviourDesc:UserBehaviourDescription = UserBehaviourDescription(observationalStaticProbabilityFnc, [0.8])

    # portfolio definiton
    rDescTheMostPopular:RecommenderDescription = RecommenderDescription(RecommenderTheMostPopular,
                            {})

    rDescCBmax:RecommenderDescription = RecommenderDescription(RecommenderCosineCB,
                            {RecommenderCosineCB.ARG_CB_DATA_PATH:Configuration.cbDataFileWithPathTFIDF,
                             RecommenderCosineCB.ARG_USER_PROFILE_STRATEGY:"max"})
    rDescCBwindow10:RecommenderDescription = RecommenderDescription(RecommenderCosineCB,
                            {RecommenderCosineCB.ARG_CB_DATA_PATH:Configuration.cbDataFileWithPathTFIDF,
                             RecommenderCosineCB.ARG_USER_PROFILE_STRATEGY:"window10"})

    rDescW2vPositiveMax:RecommenderDescription = RecommenderDescription(RecommenderW2V,
                            {RecommenderW2V.ARG_TRAIN_VARIANT:"positive",
                             RecommenderW2V.ARG_USER_PROFILE_STRATEGY:"max"})
    rDescW2vPositiveWindow10:RecommenderDescription = RecommenderDescription(RecommenderW2V,
                            {RecommenderW2V.ARG_TRAIN_VARIANT:"positive",
                             RecommenderW2V.ARG_USER_PROFILE_STRATEGY:"window10"})
    rDescW2vPosnegMax:RecommenderDescription = RecommenderDescription(RecommenderW2V,
                            {RecommenderW2V.ARG_TRAIN_VARIANT:"posneg",
                             RecommenderW2V.ARG_USER_PROFILE_STRATEGY:"max"})
    rDescW2vPosnegWindow10:RecommenderDescription = RecommenderDescription(RecommenderW2V,
                            {RecommenderW2V.ARG_TRAIN_VARIANT:"posneg",
                             RecommenderW2V.ARG_USER_PROFILE_STRATEGY:"window10"})

    rIDsCB:List[str] = ["RecomCBmax", "RecomCBwindow10"]
    rDescsCB:List[RecommenderDescription] = [rDescCBmax, rDescCBwindow10]

    #rIDsW2V:List[str] = ["RecomW2vPositiveMax", "RecomW2vPositiveWindow10", "RecomW2vPosnegMax", "RecomPosnegWindow10"]
    #rDescsW2V:List[RecommenderDescription] = [rDescW2vPositiveMax, rDescW2vPositiveWindow10, rDescW2vPosnegMax, rDescW2vPosnegWindow10]
    #rIDsW2V:List[str] = ["RecomW2vPositiveMax", "RecomW2vPosnegMax", "RecomPosnegWindow10"]
    #rDescsW2V:List[RecommenderDescription] = [rDescW2vPosnegMax, rDescW2vPosnegMax, rDescW2vPosnegWindow10]
    rIDsW2V:List[str] = ["RecomW2vPosnegMax", "RecomPosnegWindow10"]
    rDescsW2V:List[RecommenderDescription] = [rDescW2vPosnegMax, rDescW2vPosnegWindow10]

    rIDs:List[str] = ["RecomTheMostPopular"] + rIDsCB + rIDsW2V
    rDescs:List[RecommenderDescription] = [rDescTheMostPopular] + rDescsCB + rDescsW2V


    aDescBanditTS:AggregationDescription = AggregationDescription(AggrBanditTS,
                            {AggrBanditTS.ARG_SELECTORFNC:(AggrBanditTS.selectorOfRouletteWheelRatedItem,[])})
    aDescDHontFixed:AggregationDescription = AggregationDescription(AggrDHont,
                            {AggrDHont.ARG_SELECTORFNC:(AggrDHont.selectorOfTheMostVotedItem,[])})
    aDescDHontRoulette:AggregationDescription = AggregationDescription(AggrDHont,
                            {AggrDHont.ARG_SELECTORFNC:(AggrDHont.selectorOfRouletteWheelRatedItem,[])})
    aDescDHontRoulette3:AggregationDescription = AggregationDescription(AggrDHont,
                            {AggrDHont.ARG_SELECTORFNC:(AggrDHont.selectorOfRouletteWheelExpRatedItem,[3])})


    _penaltyToolOStat08HLin1002:APenalization = PenalUsingReduceRelevance(penaltyStatic, [1.0], penaltyLinear, [1.0, 0.2, 100])
    aDescNegDHontOStat08HLin1002:AggregationDescription = AggregationDescription(AggrDHontNegativeImplFeedback, {
                            AggrDHontNegativeImplFeedback.ARG_SELECTORFNC:(AggrDHont.selectorOfTheMostVotedItem,[]),
                            AggrDHontNegativeImplFeedback.ARG_PENALTY_TOOL:_penaltyToolOStat08HLin1002})

    _penaltyToolOLin0802HLin1002:APenalization = PenalUsingReduceRelevance(penaltyLinear, [0.8, 0.2, numberOfAggrItems], penaltyLinear, [1.0, 0.2, 100])
    aDescNegDHontOLin0802HLin1002:AggregationDescription = AggregationDescription(AggrDHontNegativeImplFeedback, {
                            AggrDHontNegativeImplFeedback.ARG_SELECTORFNC:(AggrDHont.selectorOfTheMostVotedItem,[]),
                            AggrDHontNegativeImplFeedback.ARG_PENALTY_TOOL:_penaltyToolOLin0802HLin1002})


    # Single method portfolios
    # TheMostPopular Portfolio description
    pDescTheMostPopular:APortfolioDescription = Portfolio1MethDescription("SingleTMPopular", "theMostPopular", rDescTheMostPopular)
    modelTheMostPopularDF:DataFrame = pd.DataFrame()

    # Cosine CB Portfolio description
    pDescCBmax:APortfolioDescription = Portfolio1MethDescription("CosCBmax", "cosCBmax", rDescCBmax)
    modelCBmaxDF:DataFrame = pd.DataFrame()

    pDescCBwindow10:APortfolioDescription = Portfolio1MethDescription("CosCBwindow10", "cosCBwindow10", rDescCBwindow10)
    modelCBwindow10DF:DataFrame = pd.DataFrame()

    # W2v Portfolio description
    pDescW2vPositiveMax:APortfolioDescription = Portfolio1MethDescription("W2vPositiveMax", "w2vPositiveMax", rDescW2vPositiveMax)
    modelW2vPositiveMaxDF:DataFrame = pd.DataFrame()

    pDescW2vPositiveWindow10:APortfolioDescription = Portfolio1MethDescription("W2vPositiveWindow10", "w2vPositiveWindow10", rDescW2vPositiveWindow10)
    modelW2vPositiveWindow10DF:DataFrame = pd.DataFrame()

    pDescW2vPosnegMax:APortfolioDescription = Portfolio1MethDescription("W2vPosnegMax", "w2vPosnegMax", rDescW2vPosnegMax)
    modelW2vPosnegMaxDF:DataFrame = pd.DataFrame()

    pDescW2vPosnegWindow10:APortfolioDescription = Portfolio1MethDescription("W2vPosnegWindow10", "w2vPosnegWindow10", rDescW2vPosnegWindow10)
    modelW2vPosnegWindow10DF:DataFrame = pd.DataFrame()



    # BanditTS Portfolio description
    pDescBanditTS:APortfolioDescription = Portfolio1AggrDescription(
            "BanditTS", rIDs, rDescs, aDescBanditTS)
    modelBanditTSDF:DataFrame = Tools.createBanditModel(rIDs)



    # DHont Fixed Portfolio description
    pDescDHontFixed:APortfolioDescription = Portfolio1AggrDescription(
            "DHontFixed", rIDs, rDescs, aDescDHontFixed)
    modelDHontFixedDF:DataFrame = Tools.createDHontModel(pDescDHontFixed.getRecommendersIDs())

    # DHont Roulette Portfolio description
    pDescDHontRoulette:APortfolioDescription = Portfolio1AggrDescription(
            "DHontRoulette", rIDs, rDescs, aDescDHontRoulette)
    modelDHontRouletteDF:DataFrame = Tools.createDHontModel(pDescDHontRoulette.getRecommendersIDs())

    # DHont Roulette3 Portfolio description
    pDescDHontRoulette3:APortfolioDescription = Portfolio1AggrDescription(
            "DHontRoulette3", rIDs, rDescs, aDescDHontRoulette3)
    modelDHontRoulette3DF:DataFrame = Tools.createDHontModel(pDescDHontRoulette3.getRecommendersIDs())



    # DHont Negative Implicit Feedback Portfolio description
    pDescNegDHontOStat08HLin1002:Portfolio1AggrDescription = Portfolio1AggrDescription(
            "NegDHontOStat08HLin1002", rIDs, rDescs, aDescNegDHontOStat08HLin1002)
    modelNegDHontOStat08HLin1002DF:DataFrame = Tools.createDHontModel(pDescNegDHontOStat08HLin1002.getRecommendersIDs())



    # DHont Negative Implicit Feedback Portfolio description
    pDescNegDHontOLin0802HLin1002:Portfolio1AggrDescription = Portfolio1AggrDescription(
            "NegDHontOLin0802HLin1002", rIDs, rDescs, aDescNegDHontOLin0802HLin1002)
    modelNegDHontOLin0802HLin1002DF:DataFrame = Tools.createDHontModel(pDescNegDHontOLin0802HLin1002.getRecommendersIDs())






