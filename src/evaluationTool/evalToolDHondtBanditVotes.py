#!/usr/bin/python3

from typing import List #class
from typing import Dict #class

from evaluationTool.aEvalTool import AEvalTool #class

from pandas.core.frame import DataFrame #class
from pandas.core.frame import Series #class

import numpy as np

class EvalToolDHondtBanditVotes(AEvalTool):
    def __init__(self, argumentsDict:Dict[str,object]):
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")

    def click(self, userID:int, rItemIDsWithResponsibility:List, clickedItemID:int, portfolioModel:DataFrame, argumentsDict:Dict[str,object]):
        if type(userID) is not int and type(userID) is not np.int64:
            raise ValueError("Argument userID isn't type int.")
        if type(rItemIDsWithResponsibility) is not list:
            raise ValueError("Argument rItemIDsWithResponsibility isn't type list.")
        #if type(clickedItemID) is not int and type(clickedItemID) is not np.int64:
        #    raise ValueError("Argument clickedItemID isn't type int.")
        if not isinstance(portfolioModel, DataFrame):
            raise ValueError("Argument pModelDF isn't type DataFrame.")
        if list(portfolioModel.columns) != ['r', 'n', 'alpha0', 'beta0']:
            raise ValueError("Argument pModelDF doen't contain rights columns.")
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")

        aggrItemIDsWithRespDF:DataFrame = DataFrame(rItemIDsWithResponsibility, columns=["itemId", "responsibility"])
        aggrItemIDsWithRespDF.set_index("itemId", inplace=True)

        #EvalToolDHont.linearNormalizingPortfolioModelDHont(portfolioModel)

        #evaluationDict[AEvalTool.CLICKS] = evaluationDict.get(AEvalTool.CLICKS, 0) + 1

        # responsibilityDict:dict[methodID:str, votes:float]
        responsibilityDict:dict[str, float] = aggrItemIDsWithRespDF.loc[clickedItemID]["responsibility"]

        # increment portfolio model
        for methodIdI in responsibilityDict.keys():
            rowI:Series = portfolioModel.loc[methodIdI]
            rowI['r'] += responsibilityDict[methodIdI]
            #rowI['n'] += 1 - responsibilityDict[methodIdI]
            #relevance_this = responsibilityDict[methodIdI]
            #relevance_others = sumMethodsVotes - relevance_this
            #update_step = self.learningRateClicks * (relevance_this - relevance_others)
            # elif action == "storeViews":
            #    update_step = -1 * learningRateViews * (relevance_this - relevance_others)
            #    pos_step = 0
            #portfolioModel.loc[methodIdI] = portfolioModel.loc[methodIdI] + update_step
            # Apply constraints on maximal and minimal volumes of votes
            #if portfolioModel.loc[methodIdI, 'votes'] < self.minVotesConst:
            #    portfolioModel.loc[methodIdI, 'votes'] = self.minVotesConst
            #elif portfolioModel.loc[methodIdI, 'votes'] > self.maxVotesConst:
            #    portfolioModel.loc[methodIdI, 'votes'] = self.maxVotesConst

         # linearly normalizing to unit sum of votes
        #EvalToolDHont.linearNormalizingPortfolioModelDHont(portfolioModel)

        print("HOP")
        print("clickedItemID: " + str(clickedItemID))
        print(portfolioModel)

    def displayed(self, userID:int, rItemIDsWithResponsibility:List, portfolioModel:DataFrame, argumentsDict:Dict[str,object]):
        if type(userID) is not int and type(userID) is not np.int64:
            raise ValueError("Argument userID isn't type int.")
        if type(rItemIDsWithResponsibility) is not list:
            raise ValueError("Argument rItemIDsWithResponsibility isn't type list.")
        if not isinstance(portfolioModel, DataFrame):
            raise ValueError("Argument pModelDF isn't type DataFrame.")
        if list(portfolioModel.columns) != ['r', 'n', 'alpha0', 'beta0']:
            raise ValueError("Argument pModelDF doen't contain rights columns.")
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")

        aggrItemIDsWithRespDF:DataFrame = DataFrame(rItemIDsWithResponsibility, columns=["itemId", "responsibility"])
        aggrItemIDsWithRespDF.set_index("itemId", inplace=True)

        # responsibilityDict:dict[methodID:str, votes:float]
        # iterate over all recommended items, penalize their methods
        for index, responsibilityI in aggrItemIDsWithRespDF.iterrows():
            responsibilityDict:dict[str,float] = responsibilityI["responsibility"]

            # increment portfolio model
            for methodIdI in responsibilityDict.keys():
                relevance_this:float = responsibilityDict.get(methodIdI)
                portfolioModel.loc[methodIdI]['n'] += relevance_this                
