#!/usr/bin/python3

from typing import List

from pandas.core.frame import DataFrame #class

from evaluationTool.evalToolDHont import EvalToolDHont #class

from input.inputsML1MDefinition import InputsML1MDefinition #class
from input.inputsML1MDefinition import Tools #class

from portfolioDescription.aPortfolioDescription import APortfolioDescription #class

from input.batchML1m.aConfig import ml1m #function


def jobDHontRoulette3(batchID:str, divisionDatasetPercentualSize:int, uBehaviour:str, repetition:int):

        d = InputsML1MDefinition

        pDescs:List[APortfolioDescription] = [d.pDescDHontRoulette3]
        models:List[DataFrame] = [Tools.createDHontModel(d.pDescDHontRoulette3.getRecommendersIDs())]
        evalTools:List = [EvalToolDHont]

        ml1m(batchID, divisionDatasetPercentualSize, uBehaviour, repetition, pDescs, models, evalTools)