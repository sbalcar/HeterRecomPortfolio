#!/usr/bin/python3

from typing import List

from pandas.core.frame import DataFrame #class

from evaluationTool.evalToolBanditTS import EvalToolBanditTS #class

from input.inputsML1MDefinition import InputsML1MDefinition #class

from portfolioDescription.aPortfolioDescription import APortfolioDescription #class

from input.batchML1m.aConfig import ml1m #function


def jobBanditTS(batchID:str, divisionDatasetPercentualSize:int, uBehaviour:str, repetition:int):

        d = InputsML1MDefinition

        pDescs:List[APortfolioDescription] = [d.pDescBanditTS]
        models:List[DataFrame] = [d.modelBanditTSDF]
        evalTools:List = [EvalToolBanditTS]

        ml1m(batchID, divisionDatasetPercentualSize, uBehaviour, repetition, pDescs, models, evalTools)