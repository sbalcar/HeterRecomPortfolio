#!/usr/bin/python3

from typing import List

from pandas.core.frame import DataFrame #class

from evaluationTool.evalToolDHont import EvalToolDHont #class

from input.inputsML1MDefinition import InputsML1MDefinition #class
from input.inputsML1MDefinition import Tools #class

from portfolioDescription.aPortfolioDescription import APortfolioDescription #class

from input.batchML1m.aConfig import ml1m #function


def jobNegDHontRoulette3OLin0802HLin1002(batchID:str, divisionDatasetPercentualSize:int, uBehaviour:str, repetition:int):

        d = InputsML1MDefinition(divisionDatasetPercentualSize)

        pDescs:List[APortfolioDescription] = [d.pDescNegDHontRoulette3OLin0802HLin1002]
        models:List[DataFrame] = [Tools.createDHontModel(d.pDescNegDHontRoulette3OLin0802HLin1002.getRecommendersIDs())]
        evalTools:List = [EvalToolDHont]

        ml1m(batchID, uBehaviour, repetition, d, pDescs, models, evalTools)