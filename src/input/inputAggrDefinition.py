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
from aggregation.aggrDHondt import AggrDHondt #class
from aggregation.aggrDHondtThompsonSampling import AggrDHondtThompsonSampling #class
from aggregation.aggrNegDHondt import AggrNegDHondt #class
from aggregation.aggrNegDHondtThompsonSampling import AggrNegDHondtThompsonSampling #class


from evaluationTool.evalToolDHondt import EvalToolDHondt #class
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

from aggregation.operators.aDHondtSelector import ADHondtSelector #class



class InputAggrDefinition:

    @staticmethod
    def exportADescBanditTS():
        aDescBanditTS:AggregationDescription = AggregationDescription(AggrBanditTS,
                                {AggrBanditTS.ARG_SELECTORFNC:(AggrBanditTS.selectorOfRouletteWheelRatedItem,[])})
        return aDescBanditTS


    @staticmethod
    def exportADescDHont(selector:ADHondtSelector):
        aDescDHontFixed:AggregationDescription = AggregationDescription(AggrDHondt,
                                {AggrDHondt.ARG_SELECTOR:selector})
        return aDescDHontFixed


    @staticmethod
    def exportADescNegDHont(selector:ADHondtSelector, nImplFeedback:APenalization):
        aDescNegDHontFixed:AggregationDescription = AggregationDescription(AggrNegDHondt,
                                {AggrNegDHondt.ARG_SELECTOR:selector,
                                 AggrNegDHondt.ARG_PENALTY_TOOL:nImplFeedback})
        return aDescNegDHontFixed


    @staticmethod
    def exportADescDHontThompsonSampling(nImplFeedback:APenalization):
        aDescDHontBanditVotesRoulette:AggregationDescription = AggregationDescription(AggrDHondtThompsonSampling,
                                {AggrDHondtThompsonSampling.ARG_SELECTOR: nImplFeedback})
        return aDescDHontBanditVotesRoulette



    @staticmethod
    def exportADescNegDHontThompsonSampling(selector:ADHondtSelector, nImplFeedback:APenalization):
        aDescDHontBanditVotesRoulette:AggregationDescription = AggregationDescription(AggrNegDHondtThompsonSampling,
                                {AggrDHondtThompsonSampling.ARG_SELECTOR:selector,
                                 AggrNegDHondt.ARG_PENALTY_TOOL: nImplFeedback})
        return aDescDHontBanditVotesRoulette











    @staticmethod
    def exportAPenaltyToolOStat08HLin1002(numberOfAggrItems:int):
        return PenalUsingReduceRelevance(penaltyStatic, [1.0], penaltyLinear, [1.0, 0.2, 100])

    @staticmethod
    def exportAPenaltyToolOLin0802HLin1002(numberOfAggrItems:int):
        return PenalUsingReduceRelevance(penaltyLinear, [0.8, 0.2, numberOfAggrItems], penaltyLinear, [1.0, 0.2, 100])

    @staticmethod
    def exportAPenaltyToolFiltering():
        return PenalUsingFiltering(1.5, 100)


class ModelDefinition:

    def createDHontModel(recommendersIDs: List[str]):
        modelDHontData:List[List] = [[rIdI, 1] for rIdI in recommendersIDs]
        modelDHontDF:DataFrame = pd.DataFrame(modelDHontData, columns=["methodID", "votes"])
        modelDHontDF.set_index("methodID", inplace=True)
        EvalToolDHondt.linearNormalizingPortfolioModelDHont(modelDHontDF)
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
