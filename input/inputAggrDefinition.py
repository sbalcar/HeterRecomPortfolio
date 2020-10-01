#!/usr/bin/python3

from typing import List

from pandas import DataFrame

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
from aggregation.aggrDHondtBanditsVotes import AggrDHondtBanditsVotes #class
from aggregation.aggrDHontNegativeImplFeedback import AggrDHontNegativeImplFeedback #class

from evaluationTool.evalToolDHont import EvalToolDHont #class
from evaluationTool.evalToolDHondtBanditVotes import EvalToolDHondtBanditVotes 

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


class InputAggrDefinition:

    @staticmethod
    def exportADescBanditTS():
        aDescBanditTS:AggregationDescription = AggregationDescription(AggrBanditTS,
                                {AggrBanditTS.ARG_SELECTORFNC:(AggrBanditTS.selectorOfRouletteWheelRatedItem,[])})
        return aDescBanditTS

    @staticmethod
    def exportADescDHontFixed():
        aDescDHontFixed:AggregationDescription = AggregationDescription(AggrDHont,
                                {AggrDHont.ARG_SELECTORFNC:(AggrDHont.selectorOfTheMostVotedItem,[])})
        return aDescDHontFixed

    @staticmethod
    def exportADescDHontRoulette():
        aDescDHontRoulette:AggregationDescription = AggregationDescription(AggrDHont,
                                {AggrDHont.ARG_SELECTORFNC:(AggrDHont.selectorOfRouletteWheelRatedItem,[])})
        return aDescDHontRoulette

    @staticmethod
    def exportADescDHontRoulette3():
        aDescDHontRoulette3:AggregationDescription = AggregationDescription(AggrDHont,
                                {AggrDHont.ARG_SELECTORFNC:(AggrDHont.selectorOfRouletteWheelExpRatedItem,[3])})
        return aDescDHontRoulette3

    @staticmethod
    def exportADescDHontBanditVotesRoulette():
        aDescDHontBanditVotesRoulette:AggregationDescription = AggregationDescription(AggrDHondtBanditsVotes,
                                {AggrDHondtBanditsVotes.ARG_SELECTORFNC:(AggrDHondtBanditsVotes.selectorOfRouletteWheelExpRatedItem,[1])})
        return aDescDHontBanditVotesRoulette    
    
    @staticmethod
    def exportADescDHontBanditVotesRoulette3():
        aDescDHontBanditVotesRoulette3:AggregationDescription = AggregationDescription(AggrDHondtBanditsVotes,
                                {AggrDHondtBanditsVotes.ARG_SELECTORFNC:(AggrDHondtBanditsVotes.selectorOfRouletteWheelExpRatedItem,[3])})
        return aDescDHontBanditVotesRoulette3

    @staticmethod
    def exportADescNegDHontFixed(aPenalization:APenalization):
        aDescNegDHontFixed:AggregationDescription = AggregationDescription(AggrDHontNegativeImplFeedback, {
                                AggrDHontNegativeImplFeedback.ARG_SELECTORFNC:(AggrDHont.selectorOfTheMostVotedItem,[]),
                                AggrDHontNegativeImplFeedback.ARG_PENALTY_TOOL:aPenalization})
        return aDescNegDHontFixed

    @staticmethod
    def exportADescNegDHontRoulette(aPenalization:APenalization):
        aDescNegDHontRoulette:AggregationDescription = AggregationDescription(AggrDHontNegativeImplFeedback, {
                                AggrDHontNegativeImplFeedback.ARG_SELECTORFNC:(AggrDHont.selectorOfRouletteWheelRatedItem,[]),
                                AggrDHontNegativeImplFeedback.ARG_PENALTY_TOOL:aPenalization})
        return aDescNegDHontRoulette

    @staticmethod
    def exportADescNegDHontRoulette3(aPenalization:APenalization):
        aDescNegDHontRoulette3:AggregationDescription = AggregationDescription(AggrDHontNegativeImplFeedback, {
                                AggrDHontNegativeImplFeedback.ARG_SELECTORFNC:(AggrDHont.selectorOfRouletteWheelExpRatedItem,[3]),
                                AggrDHontNegativeImplFeedback.ARG_PENALTY_TOOL:aPenalization})
        return aDescNegDHontRoulette3



    @staticmethod
    def exportAPenaltyToolOStat08HLin1002(numberOfAggrItems:int):
        return PenalUsingReduceRelevance(penaltyStatic, [1.0], penaltyLinear, [1.0, 0.2, 100])

    @staticmethod
    def exportAPenaltyToolOLin0802HLin1002(numberOfAggrItems:int):
        return PenalUsingReduceRelevance(penaltyLinear, [0.8, 0.2, numberOfAggrItems], penaltyLinear, [1.0, 0.2, 100])



class ModelDefinition:

    def createDHontModel(recommendersIDs: List[str]):
        modelDHontData:List[List] = [[rIdI, 1] for rIdI in recommendersIDs]
        modelDHontDF:DataFrame = pd.DataFrame(modelDHontData, columns=["methodID", "votes"])
        modelDHontDF.set_index("methodID", inplace=True)
        EvalToolDHont.linearNormalizingPortfolioModelDHont(modelDHontDF)
        return modelDHontDF
    
    def createDHondtBanditsVotesModel(recommendersIDs: List[str]):
        modelDHondtBanditsVotesData:List = [[rIdI, 1.0, 1.0, 1.0, 1.0] for rIdI in recommendersIDs]
        modelDHondtBanditsVotesDF:DataFrame = pd.DataFrame(modelDHondtBanditsVotesData, columns=["methodID", "r", "n", "alpha0", "beta0"])
        modelDHondtBanditsVotesDF.set_index("methodID", inplace=True)
        #EvalToolDHont.linearNormalizingPortfolioModelDHont(modelDHontDF)
        return modelDHondtBanditsVotesDF    

    def createBanditModel(recommendersIDs:List[str]):
        modelBanditTSData:List = [[rIdI, 1, 1, 1, 1] for rIdI in recommendersIDs]
        modelBanditTSDF:DataFrame = pd.DataFrame(modelBanditTSData, columns=["methodID", "r", "n", "alpha0", "beta0"])
        modelBanditTSDF.set_index("methodID", inplace=True)
        return modelBanditTSDF
