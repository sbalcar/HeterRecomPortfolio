#!/usr/bin/python3

from typing import List #class
from typing import Dict #class

from evaluationTool.aEvalTool import AEvalTool #class

from pandas.core.frame import DataFrame #class

import numpy as np

from aggregation.tools.aggrResponsibilityDHondt import normalizationOfDHondtResponsibility #function


class EvalToolDHondt(AEvalTool):
    #learningRateClicks:float = 0.1
    #learningRateViews:float = (0.1 / 500)
    #maxVotesConst:float = 0.99
    #minVotesConst:float = 0.01

    ARG_LEARNING_RATE_CLICKS:str = "learningRateClicks"
    ARG_LEARNING_RATE_VIEWS:str = "learningRateViews"
    ARG_NORMALIZATION_OF_RESPONSIBILITY:str = "normalizationOfResponsibility"

    ARG_VERBOSE:str = "verbose"


    def __init__(self, argsDict:dict):
        if type(argsDict) is not dict:
            raise ValueError("Argument argsDict isn't type dict.")

        self.learningRateClicks:float = argsDict[EvalToolDHondt.ARG_LEARNING_RATE_CLICKS]
        self.learningRateViews:float = argsDict[EvalToolDHondt.ARG_LEARNING_RATE_VIEWS]
        self.verbose:float = argsDict.get(EvalToolDHondt.ARG_VERBOSE, True)
        self.maxVotesConst:float = 0.99
        self.minVotesConst:float = 0.01

        self.normalizationOfResponsibility:bool = argsDict.get(self.ARG_NORMALIZATION_OF_RESPONSIBILITY, False)

        #print("learningRateClicks: " + str(self.learningRateClicks))
        #print("learningRateViews: " + str(self.learningRateViews))


    def click(self, userID:int, rItemIDsWithResponsibility:List, clickedItemID:int, portfolioModel:DataFrame, argumentsDict:Dict[str,object]):
        if type(userID) is not int and type(userID) is not np.int64:
            raise ValueError("Argument userID isn't type int.")
        if type(rItemIDsWithResponsibility) is not list:
            print(rItemIDsWithResponsibility)
            raise ValueError("Argument rItemIDsWithResponsibility isn't type list.")
        #if type(clickedItemID) is not int and type(clickedItemID) is not np.int64:
        #    raise ValueError("Argument clickedItemID isn't type int.")
        if not isinstance(portfolioModel, DataFrame):
            raise ValueError("Argument pModelDF isn't type DataFrame.")
        if list(portfolioModel.columns) != ['votes']:
            raise ValueError("Argument pModelDF doen't contain rights columns.")
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")
        if self.verbose:
            print(rItemIDsWithResponsibility)

        if self.normalizationOfResponsibility:
            rItemIDsWithResponsibility = normalizationOfDHondtResponsibility(rItemIDsWithResponsibility)

        aggrItemIDsWithRespDF:DataFrame = DataFrame(rItemIDsWithResponsibility, columns=["itemId", "responsibility"])
        aggrItemIDsWithRespDF.set_index("itemId", inplace=True)

        #EvalToolDHont.linearNormalizingPortfolioModelDHont(portfolioModel)

        # responsibilityDict:dict[methodID:str, votes:float]
        responsibilityDict:dict[str, float] = aggrItemIDsWithRespDF.loc[clickedItemID]["responsibility"]

        # increment portfolio model
        sumMethodsVotes = portfolioModel["votes"].sum()
        for methodIdI in responsibilityDict.keys():

            relevance_this = responsibilityDict[methodIdI]
            relevance_others = sumMethodsVotes - relevance_this
            update_step = self.learningRateClicks * (relevance_this - relevance_others)
            if self.verbose:
                print("update_step: " + str(update_step))
            # elif action == "storeViews":
            #    update_step = -1 * learningRateViews * (relevance_this - relevance_others)
            #    pos_step = 0

            portfolioModel.loc[methodIdI] = portfolioModel.loc[methodIdI] + update_step

            # Apply constraints on maximal and minimal volumes of votes
            if portfolioModel.loc[methodIdI, 'votes'] < self.minVotesConst:
                portfolioModel.loc[methodIdI, 'votes'] = self.minVotesConst
            elif portfolioModel.loc[methodIdI, 'votes'] > self.maxVotesConst:
                portfolioModel.loc[methodIdI, 'votes'] = self.maxVotesConst

         # linearly normalizing to unit sum of votes
        #EvalToolDHondt.linearNormalizingPortfolioModelDHont(portfolioModel)
        portfolioModel.linearNormalizing()

        if self.verbose:
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
        if list(portfolioModel.columns) != ['votes']:
            raise ValueError("Argument pModelDF doen't contain rights columns.")
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")

        if self.normalizationOfResponsibility:
            if self.verbose:
                print("NORM")
            rItemIDsWithResponsibility = normalizationOfDHondtResponsibility(rItemIDsWithResponsibility)

        aggrItemIDsWithRespDF:DataFrame = DataFrame(rItemIDsWithResponsibility, columns=["itemId", "responsibility"])
        aggrItemIDsWithRespDF.set_index("itemId", inplace=True)

        #TODO
        #portfolioModel.linearNormalizing()


        # responsibilityDict:dict[methodID:str, votes:float]
        # iterate over all recommended items, penalize their methods
        for index, responsibilityI in aggrItemIDsWithRespDF.iterrows():
            responsibilityDict:dict[str,float] = responsibilityI["responsibility"]

            # increment portfolio model
            sumMethodsVotes:float = portfolioModel.sum().loc['votes']
            methodIdI:str
            for methodIdI in responsibilityDict.keys():
                relevance_this:float = responsibilityDict.get(methodIdI)
                relevance_others:float = sumMethodsVotes - relevance_this
                update_step:float = self.learningRateViews * (relevance_this - relevance_others)

                portfolioModel.loc[methodIdI, 'votes'] = portfolioModel.loc[methodIdI, 'votes'] - update_step

                # Apply constraints on maximal and minimal volumes of votes
                if portfolioModel.loc[methodIdI, 'votes'] < self.minVotesConst:
                    portfolioModel.loc[methodIdI, 'votes'] = self.minVotesConst
                elif portfolioModel.loc[methodIdI, 'votes'] > self.maxVotesConst:
                    portfolioModel.loc[methodIdI, 'votes'] = self.maxVotesConst

        # linearly normalizing to unit sum of votes
        portfolioModel.linearNormalizing()


