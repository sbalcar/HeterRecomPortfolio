#!/usr/bin/python3

from aggregationDescription.aggregationDescription import AggregationDescription #class
from portfolioDescription.portfolioHierDescription import PortfolioHierDescription #class

from aggregation.aggrWeightedAVG import AggrWeightedAVG #class
from aggregation.aggrMMRWeightedAVG import AggrMMRWeightedAVG #class
from aggregation.aggrFAI import AggrFAI #class
from aggregation.aggrBanditTS import AggrBanditTS #class
from aggregation.aggrFuzzyDHondt import AggrFuzzyDHondt #class
from aggregation.aggrDHondtThompsonSampling import AggrDHondtThompsonSampling #class
from aggregation.aggrFuzzyDHondtINF import AggrFuzzyDHondtINF #class
from aggregation.aggrDHondtThompsonSamplingINF import AggrDHondtThompsonSamplingINF #class
from aggregation.aggrFuzzyDHondtDirectOptimize import AggrFuzzyDHondtDirectOptimize #class
from aggregation.aggrFuzzyDHondtDirectOptimizeINF import AggrFuzzyDHondtDirectOptimizeINF #class
from aggregation.aggrDHondtDirectOptimizeThompsonSamplingINF import AggrDHondtDirectOptimizeThompsonSamplingINF #class
from aggregation.aggrContextFuzzyDHondt import AggrContextFuzzyDHondt #class
from aggregation.aggrContextFuzzyDHondtDirectOptimize import AggrContextFuzzyDHondtDirectOptimize #class
from aggregation.aggrContextFuzzyDHondtINF import AggrContextFuzzyDHondtINF #class
from aggregation.aggrContextFuzzyDHondtDirectOptimize import AggrFuzzyDHondtDirectOptimize #class
from aggregation.aggrContextFuzzyDHondtDirectOptimizeINF import AggrContextFuzzyDHondtDirectOptimizeINF #class
from aggregation.aggrRandomKfromN import AggrRandomKfromN #class
from aggregation.aggrRandomRecsSwitching import AggrRandomRecsSwitching #class
from aggregation.aggrDHondtDirectOptimizeThompsonSampling import AggrDHondtDirectOptimizeThompsonSampling #class
from aggregation.aggrMMRDHondtDirectOptimizeThompsonSampling import AggrMMRDHondtDirectOptimizeThompsonSampling #class

from aggregation.negImplFeedback.aPenalization import APenalization #class
from aggregation.negImplFeedback.penalUsingFiltering import PenalUsingFiltering #class
from aggregation.negImplFeedback.penalUsingReduceRelevance import PenalUsingReduceRelevance #class
from aggregation.negImplFeedback.penalUsingProbability import PenalUsingProbability #class
from aggregation.negImplFeedback.penalUsingReduceRelevance import penaltyStatic #function
from aggregation.negImplFeedback.penalUsingReduceRelevance import penaltyLinear #function

from aggregation.operators.aDHondtSelector import ADHondtSelector #class

from evaluationTool.aEvalTool import AEvalTool #class



class InputAggrDefinition:

    @staticmethod
    def exportADescWeightedAVG():
        return AggregationDescription(AggrWeightedAVG, {
                            })

    @staticmethod
    def exportADescWeightedAVGMMR():
        return AggregationDescription(AggrMMRWeightedAVG, {
                            })

    @staticmethod
    def exportADescFAI():
        return AggregationDescription(AggrFAI, {
                            })

    @staticmethod
    def exportADescBanditTS(selector:ADHondtSelector):
        return AggregationDescription(AggrBanditTS, {
                            AggrBanditTS.ARG_SELECTOR:selector})



    @staticmethod
    def exportADescDHondt(selector:ADHondtSelector):
        return AggregationDescription(AggrFuzzyDHondt, {
                            AggrFuzzyDHondt.ARG_SELECTOR:selector})

    @staticmethod
    def exportADescDHondtINF(selector:ADHondtSelector, nImplFeedback:APenalization):
        return AggregationDescription(AggrFuzzyDHondtINF, {
                            AggrFuzzyDHondtINF.ARG_SELECTOR:selector,
                            AggrFuzzyDHondtINF.ARG_PENALTY_TOOL:nImplFeedback})



    @staticmethod
    def exportADescDHondtThompsonSampling(nImplFeedback:APenalization):
        return AggregationDescription(AggrDHondtThompsonSampling, {
                            AggrDHondtThompsonSampling.ARG_SELECTOR: nImplFeedback})

    @staticmethod
    def exportADescDHondtThompsonSamplingINF(selector:ADHondtSelector, nImplFeedback:APenalization):
        return AggregationDescription(AggrDHondtThompsonSamplingINF, {
                            AggrDHondtThompsonSampling.ARG_SELECTOR:selector,
                            AggrFuzzyDHondtINF.ARG_PENALTY_TOOL: nImplFeedback})


    @staticmethod
    def exportADescDHondtDirectOptimize(selector:ADHondtSelector,
                            discountFactor=AggrDHondtDirectOptimizeThompsonSampling.DISCFACTOR_UNIFORM):
        return AggregationDescription(AggrFuzzyDHondtDirectOptimize, {
                            AggrFuzzyDHondtDirectOptimize.ARG_SELECTOR:selector,
                            AggrDHondtDirectOptimizeThompsonSampling.ARG_DISCOUNT_FACTOR:discountFactor})


    @staticmethod
    def exportADescDFuzzyHondtDirectOptimizeINF(selector:ADHondtSelector, nImplFeedback:APenalization,
                            discountFactor=AggrDHondtDirectOptimizeThompsonSampling.DISCFACTOR_UNIFORM):
        return AggregationDescription(AggrFuzzyDHondtDirectOptimizeINF, {
                            AggrFuzzyDHondtDirectOptimizeINF.ARG_SELECTOR:selector,
                            AggrFuzzyDHondtDirectOptimizeINF.ARG_PENALTY_TOOL: nImplFeedback,
                            AggrDHondtDirectOptimizeThompsonSampling.ARG_DISCOUNT_FACTOR:discountFactor})

    @staticmethod
    def exportADescDHondtDirectOptimizeThompsonSampling(selector:ADHondtSelector,
                            discountFactor=AggrDHondtDirectOptimizeThompsonSampling.DISCFACTOR_UNIFORM):
        return AggregationDescription(AggrDHondtDirectOptimizeThompsonSampling, {
                            AggrDHondtDirectOptimizeThompsonSampling.ARG_SELECTOR:selector,
                            AggrDHondtDirectOptimizeThompsonSampling.ARG_DISCOUNT_FACTOR:discountFactor})

    @staticmethod
    def exportADescDHondtDirectOptimizeThompsonSamplingINF(selector:ADHondtSelector, nImplFeedback:APenalization,
                            discountFactor=AggrDHondtDirectOptimizeThompsonSampling.DISCFACTOR_UNIFORM):
        return AggregationDescription(AggrDHondtDirectOptimizeThompsonSamplingINF, {
                            AggrDHondtThompsonSampling.ARG_SELECTOR:selector,
                            AggrFuzzyDHondtINF.ARG_PENALTY_TOOL: nImplFeedback,
                            AggrDHondtDirectOptimizeThompsonSamplingINF.ARG_DISCOUNT_FACTOR:discountFactor})

    @staticmethod
    def exportADescDHondtDirectOptimizeThompsonSamplingMMR(selector:ADHondtSelector,
                            discountFactor=AggrDHondtDirectOptimizeThompsonSampling.DISCFACTOR_UNIFORM):
        return AggregationDescription(AggrMMRDHondtDirectOptimizeThompsonSampling, {
                            AggrDHondtDirectOptimizeThompsonSampling.ARG_SELECTOR:selector,
                            AggrDHondtDirectOptimizeThompsonSampling.ARG_DISCOUNT_FACTOR:discountFactor,
                            AggrMMRDHondtDirectOptimizeThompsonSampling.ARG_MMR_LAMBDA:0.5})

    @staticmethod
    def exportADescDContextHondt(selector:ADHondtSelector, eTool:AEvalTool):
        return AggregationDescription(AggrContextFuzzyDHondt, {
                            AggrContextFuzzyDHondt.ARG_SELECTOR:selector,
                            AggrContextFuzzyDHondt.ARG_EVAL_TOOL:eTool})

    @staticmethod
    def exportADescDContextHondtINF(selector:ADHondtSelector, nImplFeedback:APenalization, eTool:AEvalTool):
        return AggregationDescription(AggrContextFuzzyDHondtINF, {
                            AggrContextFuzzyDHondtINF.ARG_SELECTOR:selector,
                            AggrContextFuzzyDHondtINF.ARG_EVAL_TOOL:eTool,
                            AggrFuzzyDHondtDirectOptimizeINF.ARG_PENALTY_TOOL: nImplFeedback})

    @staticmethod
    def exportADescContextFuzzyDHondtDirectOptimize(selector:ADHondtSelector, eTool:AEvalTool):
        return AggregationDescription(AggrContextFuzzyDHondtDirectOptimize, {
                            AggrContextFuzzyDHondtDirectOptimize.ARG_SELECTOR:selector,
                            AggrContextFuzzyDHondtDirectOptimize.ARG_EVAL_TOOL:eTool})

    @staticmethod
    def exportADescContextFuzzyDHondtDirectOptimizeINF(selector:ADHondtSelector, nImplFeedback:APenalization, eTool:AEvalTool):
        return AggregationDescription(AggrContextFuzzyDHondtDirectOptimizeINF, {
                            AggrContextFuzzyDHondtINF.ARG_SELECTOR:selector,
                            AggrContextFuzzyDHondtINF.ARG_EVAL_TOOL:eTool,
                            AggrFuzzyDHondtDirectOptimizeINF.ARG_PENALTY_TOOL: nImplFeedback})


    @staticmethod
    def exportADescRandomKfromN(mainMethodID:str):
        return AggregationDescription(AggrRandomKfromN, {
            AggrRandomKfromN.ARG_MAIN_METHOD: mainMethodID,
            AggrRandomKfromN.ARG_MAX_REC_SIZE: 100})


    @staticmethod
    def exportADescRandomRecsSwitching(mainMethodID:str):
        return AggregationDescription(AggrRandomRecsSwitching, {
            AggrRandomRecsSwitching.ARG_MAIN_METHOD: mainMethodID})



class PenalizationToolDefinition:

    @staticmethod
    def exportPenaltyToolOStat08HLin1002(numberOfAggrItems:int):
        return PenalUsingReduceRelevance(penaltyStatic, [1.0], penaltyLinear, [1.0, 0.2, 100], 100)

    @staticmethod
    def exportPenaltyToolOLin0802HLin1002(numberOfAggrItems:int):
        return PenalUsingReduceRelevance(penaltyLinear, [0.8, 0.2, numberOfAggrItems], penaltyLinear, [1.0, 0.2, 100], 100)

    @staticmethod
    def exportProbPenaltyToolOStat08HLin1002(numberOfAggrItems:int):
        return PenalUsingProbability(penaltyStatic, [1.0], penaltyLinear, [1.0, 0.2, 100], 100)

    @staticmethod
    def exportProbPenaltyToolOLin0802HLin1002(numberOfAggrItems:int):
        return PenalUsingProbability(penaltyLinear, [0.8, 0.2, numberOfAggrItems], penaltyLinear, [1.0, 0.2, 100], 100)

    @staticmethod
    def exportPenaltyToolFiltering():
        return PenalUsingFiltering(1.5, 100)


