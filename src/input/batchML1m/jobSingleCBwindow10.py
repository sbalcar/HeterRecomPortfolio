#!/usr/bin/python3

from typing import List

from pandas.core.frame import DataFrame #class

from evaluationTool.evalToolSingleMethod import EToolSingleMethod #class

from input.inputsML1MDefinition import InputsML1MDefinition #class

from portfolioDescription.aPortfolioDescription import APortfolioDescription #class

from input.batchML1m.aConfig import ml1m #function


def jobSingleML1mCBwindow10(batchID:str, divisionDatasetPercentualSize:int, uBehaviour:str, repetition:int):

        d = InputsML1MDefinition

        pDescs:List[APortfolioDescription] = [d.pDescCBwindow10]
        models:List[DataFrame] = [d.modelCBwindow10DF]
        evalTools:List = [EToolSingleMethod]

        ml1m(batchID, divisionDatasetPercentualSize, uBehaviour, repetition, pDescs, models, evalTools)