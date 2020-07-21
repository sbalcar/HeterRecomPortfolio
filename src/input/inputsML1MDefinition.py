#!/usr/bin/python3

from typing import List

from recommenderDescription.recommenderDescription import RecommenderDescription #class

from recommender.recommenderTheMostPopular import RecommenderTheMostPopular #class
from recommender.recommenderCosineCB import RecommenderCosineCB #class
from recommender.recommenderW2V import RecommenderW2V #class

from datasets.ratings import Ratings #class
from datasets.users import Users #class
from datasets.items import Items #class
from datasets.behaviours import Behaviours #class

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
from userBehaviourDescription.userBehaviourDescription import observationalLinearProbabilityFnc #function

from aggregation.toolsDHontNF.penalizationOfResultsByNegImpFeedback.aPenalization import APenalization #class
from aggregation.toolsDHontNF.penalizationOfResultsByNegImpFeedback.penalUsingFiltering import PenalUsingFiltering #class
from aggregation.toolsDHontNF.penalizationOfResultsByNegImpFeedback.penalUsingReduceRelevance import PenalUsingReduceRelevance #class
from aggregation.toolsDHontNF.penalizationOfResultsByNegImpFeedback.penalUsingReduceRelevance import penaltyStatic #function
from aggregation.toolsDHontNF.penalizationOfResultsByNegImpFeedback.penalUsingReduceRelevance import penaltyLinear #function


class Tools:

    def createDHontModel(recommendersIDs: List[str]):
        modelDHontData:List[List] = [[rIdI, 1] for rIdI in recommendersIDs]
        modelDHontDF:DataFrame = pd.DataFrame(modelDHontData, columns=["methodID", "votes"])
        modelDHontDF.set_index("methodID", inplace=True)
        EvalToolDHont.linearNormalizingPortfolioModelDHont(modelDHontDF)
        return modelDHontDF

    def createBanditModel(recommendersIDs:List[str]):
        modelBanditTSData:List = [[rIdI, 1, 1, 1, 1] for rIdI in recommendersIDs]
        modelBanditTSDF:DataFrame = pd.DataFrame(modelBanditTSData, columns=["methodID", "r", "n", "alpha0", "beta0"])
        modelBanditTSDF.set_index("methodID", inplace=True)
        return modelBanditTSDF

class InputsML1MDefinition:

    def __init__(self, divisionDatasetPercentualSize:int):

        self.divisionDatasetPercentualSize = divisionDatasetPercentualSize

        self.datasetID:str = "ml1m" + "Div" + str(divisionDatasetPercentualSize)

        # dataset reading
        self.ratingsDF:DataFrame = Ratings.readFromFileMl1m()
        self.usersDF:DataFrame = Users.readFromFileMl1m()
        self.itemsDF:DataFrame = Items.readFromFileMl1m()
        self.behavioursDF:DataFrame = Behaviours.readFromFileMl1m()

        self.numberOfRecommItems:int = 100
        self.numberOfAggrItems:int = 20

        # portfolio definiton
        self.rDescTheMostPopular:RecommenderDescription = RecommenderDescription(RecommenderTheMostPopular,
                                {})

        self.rDescCBmean:RecommenderDescription = RecommenderDescription(RecommenderCosineCB,
                                {RecommenderCosineCB.ARG_CB_DATA_PATH:Configuration.cbDataFileWithPathTFIDF,
                                 RecommenderCosineCB.ARG_USER_PROFILE_STRATEGY:"mean"})
        self.rDescCBwindow3:RecommenderDescription = RecommenderDescription(RecommenderCosineCB,
                                {RecommenderCosineCB.ARG_CB_DATA_PATH:Configuration.cbDataFileWithPathTFIDF,
                                 RecommenderCosineCB.ARG_USER_PROFILE_STRATEGY:"window3"})

        self.rDescW2vPositiveMax:RecommenderDescription = RecommenderDescription(RecommenderW2V,
                                {RecommenderW2V.ARG_TRAIN_VARIANT:"positive",
                                 RecommenderW2V.ARG_USER_PROFILE_STRATEGY:"max",
                                 RecommenderW2V.ARG_DATASET_ID: self.datasetID})
        self.rDescW2vPositiveWindow10:RecommenderDescription = RecommenderDescription(RecommenderW2V,
                                {RecommenderW2V.ARG_TRAIN_VARIANT:"positive",
                                 RecommenderW2V.ARG_USER_PROFILE_STRATEGY:"window10",
                                 RecommenderW2V.ARG_DATASET_ID: self.datasetID})
        self.rDescW2vPosnegMean:RecommenderDescription = RecommenderDescription(RecommenderW2V,
                                {RecommenderW2V.ARG_TRAIN_VARIANT:"posneg",
                                 RecommenderW2V.ARG_USER_PROFILE_STRATEGY:"mean",
                                 RecommenderW2V.ARG_DATASET_ID:self.datasetID})
        self.rDescW2vPosnegWindow3:RecommenderDescription = RecommenderDescription(RecommenderW2V,
                                {RecommenderW2V.ARG_TRAIN_VARIANT:"posneg",
                                 RecommenderW2V.ARG_USER_PROFILE_STRATEGY:"window3",
                                 RecommenderW2V.ARG_DATASET_ID: self.datasetID})

        rIDsCB:List[str] = ["RecomCBmean", "RecomCBwindow3"]
        rDescsCB:List[RecommenderDescription] = [self.rDescCBmean, self.rDescCBwindow3]

        #rIDsW2V:List[str] = ["RecomW2vPositiveMax", "RecomW2vPositiveWindow10", "RecomW2vPosnegMean", "RecomW2vPosnegWindow10"]
        #rDescsW2V:List[RecommenderDescription] = [rDescW2vPositiveMax, rDescW2vPositiveWindow10, rDescW2vPosnegMean, rDescW2vPosnegWindow10]
        #rIDsW2V:List[str] = ["RecomW2vPositiveMax", "RecomW2vPosnegMax", "RecomW2vPosnegWindow10"]
        #rDescsW2V:List[RecommenderDescription] = [rDescW2vPosnegMax, rDescW2vPosnegMax, rDescW2vPosnegWindow10]
        rIDsW2V:List[str] = ["RecomW2vPosnegMax", "RecomW2vPosnegWindow3"]
        rDescsW2V:List[RecommenderDescription] = [self.rDescW2vPosnegMean, self.rDescW2vPosnegWindow3]


        self.rIDs:List[str] = ["RecomTheMostPopular"] + rIDsCB + rIDsW2V
        self.rDescs:List[RecommenderDescription] = [self.rDescTheMostPopular] + rDescsCB + rDescsW2V


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

        _penaltyToolOLin0802HLin1002:APenalization = PenalUsingReduceRelevance(penaltyLinear, [0.8, 0.2, self.numberOfAggrItems], penaltyLinear, [1.0, 0.2, 100])
        aDescNegDHontOLin0802HLin1002:AggregationDescription = AggregationDescription(AggrDHontNegativeImplFeedback, {
                                AggrDHontNegativeImplFeedback.ARG_SELECTORFNC:(AggrDHont.selectorOfTheMostVotedItem,[]),
                                AggrDHontNegativeImplFeedback.ARG_PENALTY_TOOL:_penaltyToolOLin0802HLin1002})


        # Single method portfolios
        # TheMostPopular Portfolio description
        self.pDescTheMostPopular:APortfolioDescription = Portfolio1MethDescription("SingleTMPopular", "theMostPopular", self.rDescTheMostPopular)

        # Cosine CB Portfolio description
        self.pDescCBmax:APortfolioDescription = Portfolio1MethDescription("CosCBmax", "cosCBmax", self.rDescCBmean)
        self.pDescCBwindow10:APortfolioDescription = Portfolio1MethDescription("CosCBwindow3", "cosCBwindow3", self.rDescCBwindow3)

        # W2v Portfolio description
        #pDescW2vPositiveMax:APortfolioDescription = Portfolio1MethDescription("W2vPositiveMax", "w2vPositiveMax", rDescW2vPositiveMax)
        #pDescW2vPositiveWindow10:APortfolioDescription = Portfolio1MethDescription("W2vPositiveWindow10", "w2vPositiveWindow10", rDescW2vPositiveWindow10)
        self.pDescW2vPosnegMean:APortfolioDescription = Portfolio1MethDescription("W2vPosnegMean", "w2vPosnegMean", self.rDescW2vPosnegMean)
        self.pDescW2vPosnegWindow3:APortfolioDescription = Portfolio1MethDescription("W2vPosnegWindow3", "w2vPosnegWindow3", self.rDescW2vPosnegWindow3)


        # BanditTS Portfolio description
        self.pDescBanditTS:APortfolioDescription = Portfolio1AggrDescription(
                "BanditTS", self.rIDs, self.rDescs, aDescBanditTS)


        # DHont Fixed Portfolio description
        self.pDescDHontFixed:APortfolioDescription = Portfolio1AggrDescription(
                "DHontFixed", self.rIDs, self.rDescs, aDescDHontFixed)

        # DHont Roulette Portfolio description
        self.pDescDHontRoulette:APortfolioDescription = Portfolio1AggrDescription(
                "DHontRoulette", self.rIDs, self.rDescs, aDescDHontRoulette)

        # DHont Roulette3 Portfolio description
        self.pDescDHontRoulette3:APortfolioDescription = Portfolio1AggrDescription(
                "DHontRoulette3", self.rIDs, self.rDescs, aDescDHontRoulette3)


        # DHont Negative Implicit Feedback Portfolio description
        self.pDescNegDHontOStat08HLin1002:Portfolio1AggrDescription = Portfolio1AggrDescription(
                "NegDHontOStat08HLin1002", self.rIDs, self.rDescs, aDescNegDHontOStat08HLin1002)


        # DHont Negative Implicit Feedback Portfolio description
        self.pDescNegDHontOLin0802HLin1002:Portfolio1AggrDescription = Portfolio1AggrDescription(
                "NegDHontOLin0802HLin1002", self.rIDs, self.rDescs, aDescNegDHontOLin0802HLin1002)





