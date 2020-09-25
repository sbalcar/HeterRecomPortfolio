#!/usr/bin/python3

from typing import List

from pandas.core.frame import DataFrame #class

from portfolioDescription.portfolio1MethDescription import Portfolio1MethDescription #class
from portfolioDescription.portfolio1AggrDescription import Portfolio1AggrDescription #class

from evaluationTool.evalToolSingleMethod import EToolSingleMethod #class

from input.InputRecomDefinition import InputRecomDefinition #class

from portfolioDescription.aPortfolioDescription import APortfolioDescription #class

from input.batchML1m.aML1MConfig import AML1MConf #function

import pandas as pd


def jobSingleML1mCBmax(batchID:str, divisionDatasetPercentualSize:int, uBehaviour:str, repetition:int):

        aConf:AML1MConf = AML1MConf(batchID, divisionDatasetPercentualSize, uBehaviour, repetition)

        pDescr:APortfolioDescription = Portfolio1MethDescription(
                "CosCBmax", "cosCBmax", InputRecomDefinition.exportRDescCBmean(aConf.datasetID))

        model:DataFrame = pd.DataFrame()
        eTool:List = EToolSingleMethod()

        aConf.run(pDescr, model, eTool)