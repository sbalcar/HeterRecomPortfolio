#!/usr/bin/python3

from typing import List

from aggregationDescription.aggregationDescription import AggregationDescription #class
from aggregation.aggrBanditTS import AggrBanditTS #class
from aggregation.aggrFuzzyDHondt import AggrFuzzyDHondt #class
from aggregation.aggrDHondtThompsonSampling import AggrDHondtThompsonSampling #class
from aggregation.aggrFuzzyDHondtINF import AggrFuzzyDHondtINF #class
from aggregation.aggrDHondtThompsonSamplingINF import AggrDHondtThompsonSamplingINF #class
from aggregation.aggrFuzzyDHondtDirectOptimize import AggrFuzzyDHondtDirectOptimize #class
from aggregation.aggrFuzzyDHondtDirectOptimizeINF import AggrFuzzyDHondtDirectOptimizeINF #class

from evaluationTool.evalToolDHondt import EvalToolDHondt #class

import pandas as pd

from pandas.core.frame import DataFrame #class

from aggregation.negImplFeedback.aPenalization import APenalization #class
from aggregation.negImplFeedback.penalUsingFiltering import PenalUsingFiltering #class
from aggregation.negImplFeedback.penalUsingReduceRelevance import PenalUsingReduceRelevance #class
from aggregation.negImplFeedback.penalUsingReduceRelevance import penaltyStatic #function
from aggregation.negImplFeedback.penalUsingReduceRelevance import penaltyLinear #function

from aggregation.operators.aDHondtSelector import ADHondtSelector #class



class InputAggrDefinition:

    @staticmethod
    def exportADescBanditTS(selector:ADHondtSelector):
        aDescBanditTS:AggregationDescription = AggregationDescription(AggrBanditTS,
                                                        {AggrBanditTS.ARG_SELECTOR:selector})
        return aDescBanditTS



    @staticmethod
    def exportADescDHont(selector:ADHondtSelector):
        aDescDHontFixed:AggregationDescription = AggregationDescription(AggrFuzzyDHondt,
                                                        {AggrFuzzyDHondt.ARG_SELECTOR:selector})
        return aDescDHontFixed

    @staticmethod
    def exportADescDHontINF(selector:ADHondtSelector, nImplFeedback:APenalization):
        aDescNegDHontFixed:AggregationDescription = AggregationDescription(AggrFuzzyDHondtINF,
                                                        {AggrFuzzyDHondtINF.ARG_SELECTOR:selector,
                                                         AggrFuzzyDHondtINF.ARG_PENALTY_TOOL:nImplFeedback})
        return aDescNegDHontFixed



    @staticmethod
    def exportADescDHontThompsonSampling(nImplFeedback:APenalization):
        aDescDHontBanditVotesRoulette:AggregationDescription = AggregationDescription(AggrDHondtThompsonSampling,
                                                        {AggrDHondtThompsonSampling.ARG_SELECTOR: nImplFeedback})
        return aDescDHontBanditVotesRoulette

    @staticmethod
    def exportADescDHontThompsonSamplingINF(selector:ADHondtSelector, nImplFeedback:APenalization):
        aDescDHontBanditVotesRoulette:AggregationDescription = AggregationDescription(AggrDHondtThompsonSamplingINF,
                                                        {AggrDHondtThompsonSampling.ARG_SELECTOR:selector,
                                                         AggrFuzzyDHondtINF.ARG_PENALTY_TOOL: nImplFeedback})
        return aDescDHontBanditVotesRoulette



    @staticmethod
    def exportADescDHontDirectOptimize(selector:ADHondtSelector):
        aDescFuzzyDHondtDirectOptimize:AggregationDescription = AggregationDescription(AggrFuzzyDHondtDirectOptimize,
                                                        {AggrFuzzyDHondtDirectOptimize.ARG_SELECTOR:selector})
        return aDescFuzzyDHondtDirectOptimize

    @staticmethod
    def exportADescDFuzzyHontDirectOptimizeINF(selector:ADHondtSelector, nImplFeedback:APenalization):
        aDescFuzzyDHondtDirectOptimizeINF:AggregationDescription = AggregationDescription(AggrFuzzyDHondtDirectOptimizeINF,
                                                        {AggrFuzzyDHondtDirectOptimizeINF.ARG_SELECTOR:selector,
                                                         AggrFuzzyDHondtDirectOptimizeINF.ARG_PENALTY_TOOL: nImplFeedback})
        return aDescFuzzyDHondtDirectOptimizeINF







    @staticmethod
    def exportAPenaltyToolOStat08HLin1002(numberOfAggrItems:int):
        return PenalUsingReduceRelevance(penaltyStatic, [1.0], penaltyLinear, [1.0, 0.2, 100], 100)

    @staticmethod
    def exportAPenaltyToolOLin0802HLin1002(numberOfAggrItems:int):
        return PenalUsingReduceRelevance(penaltyLinear, [0.8, 0.2, numberOfAggrItems], penaltyLinear, [1.0, 0.2, 100], 100)

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
