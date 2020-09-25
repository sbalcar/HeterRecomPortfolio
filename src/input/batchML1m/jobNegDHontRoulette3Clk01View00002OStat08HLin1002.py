#!/usr/bin/python3

from typing import List

from pandas.core.frame import DataFrame #class

from aggregationDescription.aggregationDescription import AggregationDescription #class

from portfolioDescription.portfolio1MethDescription import Portfolio1MethDescription #class
from portfolioDescription.portfolio1AggrDescription import Portfolio1AggrDescription #class

from evaluationTool.aEvalTool import AEvalTool #class
from evaluationTool.evalToolDHont import EvalToolDHont #class

from input.inputAggrDefinition import InputAggrDefinition, ModelDefinition  # class

from input.InputRecomDefinition import InputRecomDefinition #class

from portfolioDescription.aPortfolioDescription import APortfolioDescription #class

from input.batchML1m.aML1MConfig import AML1MConf #function


def jobNegDHontRoulette3Clk01View00002OStat08HLin1002(batchID:str, divisionDatasetPercentualSize:int, uBehaviour:str, repetition:int):

        aConf:AML1MConf = AML1MConf(batchID, divisionDatasetPercentualSize, uBehaviour, repetition)

        rIDs, rDescs = InputRecomDefinition.exportPairOfRecomIdsAndRecomDescrs(aConf.datasetID)

        aDescNegDHontRoulette3OStat08HLin1002:AggregationDescription = InputAggrDefinition.exportADescNegDHontRoulette3(
                InputAggrDefinition.exportAPenaltyToolOStat08HLin1002(AML1MConf.numberOfAggrItems))

        pDescr:Portfolio1AggrDescription = Portfolio1AggrDescription(
                "NegDHontRoulette3Clk01View00002OStat08HLin1002", rIDs, rDescs, aDescNegDHontRoulette3OStat08HLin1002)

        eTool:AEvalTool = EvalToolDHont({EvalToolDHont.ARG_LEARNING_RATE_CLICKS:0.1,
                                         EvalToolDHont.ARG_LEARNING_RATE_VIEWS:0.1 / 500})

        model:DataFrame = ModelDefinition.createDHontModel(pDescr.getRecommendersIDs())

        aConf:AML1MConf = AML1MConf(batchID, divisionDatasetPercentualSize, uBehaviour, repetition)
        aConf.run(pDescr, model, eTool)