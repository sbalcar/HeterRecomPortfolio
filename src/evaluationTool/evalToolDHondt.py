#!/usr/bin/python3

from typing import List #class
from typing import Dict #class

from evaluationTool.aEvalTool import AEvalTool #class

from pandas.core.frame import DataFrame #class

import numpy as np

class EvalToolDHondt(AEvalTool):
    #learningRateClicks:float = 0.1
    #learningRateViews:float = (0.1 / 500)
    #maxVotesConst:float = 0.99
    #minVotesConst:float = 0.01

    ARG_LEARNING_RATE_CLICKS:str = "learningRateClicks"
    ARG_LEARNING_RATE_VIEWS:str = "learningRateViews"


    def __init__(self, argsDict:dict):
        if type(argsDict) is not dict:
            raise ValueError("Argument argsDict isn't type dict.")

        self.learningRateClicks:float = argsDict[EvalToolDHondt.ARG_LEARNING_RATE_CLICKS]
        self.learningRateViews:float = argsDict[EvalToolDHondt.ARG_LEARNING_RATE_VIEWS]
        self.maxVotesConst:float = 0.99
        self.minVotesConst:float = 0.01

        #print("learningRateClicks: " + str(self.learningRateClicks))
        #print("learningRateViews: " + str(self.learningRateViews))


    def click(self, rItemIDsWithResponsibility:List, clickedItemID:int, portfolioModel:DataFrame, argumentsDict:Dict[str,object]):
        if type(rItemIDsWithResponsibility) is not list:
            print(rItemIDsWithResponsibility)
            raise ValueError("Argument rItemIDsWithResponsibility isn't type list.")
        #if type(clickedItemID) is not int and type(clickedItemID) is not np.int64:
        #    raise ValueError("Argument clickedItemID isn't type int.")
        if type(portfolioModel) is not DataFrame:
            raise ValueError("Argument pModelDF isn't type DataFrame.")
        if list(portfolioModel.columns) != ['votes']:
            raise ValueError("Argument pModelDF doen't contain rights columns.")
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")
        print(rItemIDsWithResponsibility)

        aggrItemIDsWithRespDF:DataFrame = DataFrame(rItemIDsWithResponsibility, columns=["itemId", "responsibility"])
        aggrItemIDsWithRespDF.set_index("itemId", inplace=True)

        #EvalToolDHont.linearNormalizingPortfolioModelDHont(portfolioModel)

        #evaluationDict[AEvalTool.CLICKS] = evaluationDict.get(AEvalTool.CLICKS, 0) + 1

        # responsibilityDict:dict[methodID:str, votes:float]
        responsibilityDict:dict[str, float] = aggrItemIDsWithRespDF.loc[clickedItemID]["responsibility"]

        # increment portfolio model
        sumMethodsVotes = portfolioModel["votes"].sum()
        for methodIdI in responsibilityDict.keys():

            relevance_this = responsibilityDict[methodIdI]
            relevance_others = sumMethodsVotes - relevance_this
            update_step = self.learningRateClicks * (relevance_this - relevance_others)
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
        EvalToolDHondt.linearNormalizingPortfolioModelDHont(portfolioModel)

        print("HOP")
        print("clickedItemID: " + str(clickedItemID))
        print(portfolioModel)

    def displayed(self, rItemIDsWithResponsibility:List, portfolioModel:DataFrame, argumentsDict:Dict[str,object]):
        if type(rItemIDsWithResponsibility) is not list:
            raise ValueError("Argument rItemIDsWithResponsibility isn't type list.")
        if type(portfolioModel) is not DataFrame:
            raise ValueError("Argument pModelDF isn't type DataFrame.")
        if list(portfolioModel.columns) != ['votes']:
            raise ValueError("Argument pModelDF doen't contain rights columns.")
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")

        aggrItemIDsWithRespDF:DataFrame = DataFrame(rItemIDsWithResponsibility, columns=["itemId", "responsibility"])
        aggrItemIDsWithRespDF.set_index("itemId", inplace=True)

        #TODO
        #EvalToolDHondt.linearNormalizingPortfolioModelDHont(portfolioModel)

        # responsibilityDict:dict[methodID:str, votes:float]
        # iterate over all recommended items, penalize their methods
        for index, responsibilityI in aggrItemIDsWithRespDF.iterrows():
            responsibilityDict:dict[str,float] = responsibilityI["responsibility"]

            # increment portfolio model
            sumMethodsVotes:float = portfolioModel.sum()
            methodIdI:str
            for methodIdI in responsibilityDict.keys():
                relevance_this:float = responsibilityDict.get(methodIdI)
                relevance_others:float = sumMethodsVotes - relevance_this
                update_step:float = -1 * self.learningRateViews * (relevance_this - relevance_others)

                portfolioModel.loc[methodIdI] = portfolioModel.loc[methodIdI] + update_step

                # Apply constraints on maximal and minimal volumes of votes
                if portfolioModel.loc[methodIdI, 'votes'] < self.minVotesConst:
                    portfolioModel.loc[methodIdI, 'votes'] = self.minVotesConst
                elif portfolioModel.loc[methodIdI, 'votes'] > self.maxVotesConst:
                    portfolioModel.loc[methodIdI, 'votes'] = self.maxVotesConst

        # linearly normalizing to unit sum of votes
        EvalToolDHondt.linearNormalizingPortfolioModelDHont(portfolioModel)



    @staticmethod
    def linearNormalizingPortfolioModelDHont(portfolioModelDHont:DataFrame):
        # linearly normalizing to unit sum of votes
        sumMethodsVotes:float = portfolioModelDHont["votes"].sum()
        for methodIdI in portfolioModelDHont.index:
            portfolioModelDHont.loc[methodIdI, "votes"] = portfolioModelDHont.loc[methodIdI, "votes"] / sumMethodsVotes