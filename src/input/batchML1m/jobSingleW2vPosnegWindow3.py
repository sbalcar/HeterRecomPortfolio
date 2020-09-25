#!/usr/bin/python3

from typing import List

from pandas.core.frame import DataFrame #class

from portfolioDescription.portfolio1MethDescription import Portfolio1MethDescription #class
from portfolioDescription.portfolio1AggrDescription import Portfolio1AggrDescription #class

from evaluationTool.evalToolSingleMethod import EToolSingleMethod #class

from input.InputRecomDefinition import InputRecomDefinition #class
from input.inputAggrDefinition import ModelDefinition

from portfolioDescription.aPortfolioDescription import APortfolioDescription #class

from input.batchML1m.aML1MConfig import AML1MConf #function

import pandas as pd


def jobSingleW2vPosnegWindow3(batchID:str, divisionDatasetPercentualSize:int, uBehaviour:str, repetition:int):

        aConf:AML1MConf = AML1MConf(batchID, divisionDatasetPercentualSize, uBehaviour, repetition)

        pDescr:APortfolioDescription = Portfolio1MethDescription(
                "W2vPosnegWindow3", "w2vPosnegWindow3", InputRecomDefinition.exportRDescW2vPosnegWindow3(aConf.datasetID))

        model:DataFrame = pd.DataFrame()
        eTool = EToolSingleMethod()

        aConf.run(pDescr, model, eTool)