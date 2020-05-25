#!/usr/bin/python3

from typing import List

from evaluationTool.aEvalTool import AEvalTool #class

from pandas.core.frame import DataFrame #class

import numpy as np

class EToolDHontHit1(AEvalTool):
    # TODO: maybe store learning rates to a database?
    learningRateClicks = 0.1
    learningRateViews = (0.1 / 500)
    maxVotesConst = 0.99
    minVotesConst = 0.01

    @staticmethod
    def click(rItemIDsWithResponsibility:List, clickedItemID:int, probability:float, portfolioModel:DataFrame,
              evaluationDict:dict):
        if type(rItemIDsWithResponsibility) is not list:
            raise ValueError("Argument rItemIDsWithResponsibility isn't type list.")
        if type(clickedItemID) is not int and type(clickedItemID) is not np.int64:
            raise ValueError("Argument clickedItemID isn't type int.")
        if type(probability) is not float:
            raise ValueError("Argument probability isn't type float.")
        if type(portfolioModel) is not DataFrame:
            raise ValueError("Argument pModelDF isn't type DataFrame.")
        if list(portfolioModel.columns) != ['votes']:
            raise ValueError("Argument pModelDF doen't contain rights columns.")
        if type(evaluationDict) is not dict:
            raise ValueError("Argument evaluationDict isn't type dict.")

        aggrItemIDsWithRespDF: DataFrame = DataFrame(rItemIDsWithResponsibility, columns=["itemId", "responsibility"])
        aggrItemIDsWithRespDF.set_index("itemId", inplace=True)

        print("HOP")
        print("clickedItemID: " + str(clickedItemID))

        evaluationDict[AEvalTool.CLICKS] = evaluationDict.get(AEvalTool.CLICKS, 0) + 1

        # responsibilityDict:dict[methodID:str, votes:float]
        responsibilityDict: dict[str, float] = aggrItemIDsWithRespDF.loc[clickedItemID]["responsibility"]

        # increment portfolio model
        sumMethodsVotes = portfolioModel.sum()
        for methodIdI in responsibilityDict.keys():

            relevance_this = responsibilityDict[methodIdI]
            relevance_others = sumMethodsVotes - relevance_this
            update_step = EToolDHontHit1.learningRateClicks * (relevance_this - relevance_others)

            # elif action == "storeViews":
            #    update_step = -1 * learningRateViews * (relevance_this - relevance_others)
            #    pos_step = 0

            portfolioModel.loc[methodIdI] = portfolioModel.loc[methodIdI] + update_step

            # Apply constraints on maximal and minimal volumes of votes
            if portfolioModel.loc[methodIdI, 'votes'] < EToolDHontHit1.minVotesConst:
                portfolioModel.loc[methodIdI, 'votes'] = EToolDHontHit1.minVotesConst
            elif portfolioModel.loc[methodIdI, 'votes'] > EToolDHontHit1.maxVotesConst:
                portfolioModel.loc[methodIdI, 'votes'] = EToolDHontHit1.maxVotesConst

         # linearly normalizing to unit sum of votes
        sumMethodsVotes:float = portfolioModel.sum()
        for methodIdI in portfolioModel.index:
            portfolioModel.loc[methodIdI] = portfolioModel.loc[methodIdI] / sumMethodsVotes

    @staticmethod
    def displayed(rItemIDsWithResponsibility:List, portfolioModel:DataFrame, evaluationDict:dict):
        if type(rItemIDsWithResponsibility) is not list:
            raise ValueError("Argument rItemIDsWithResponsibility isn't type list.")
        if type(portfolioModel) is not DataFrame:
            raise ValueError("Argument pModelDF isn't type DataFrame.")
        if list(portfolioModel.columns) != ['votes']:
            raise ValueError("Argument pModelDF doen't contain rights columns.")
        if type(evaluationDict) is not dict:
            raise ValueError("Argument evaluationDict isn't type dict.")

        aggrItemIDsWithRespDF:DataFrame = DataFrame(rItemIDsWithResponsibility, columns=["itemId", "responsibility"])
        aggrItemIDsWithRespDF.set_index("itemId", inplace=True)

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
                update_step:float = -1 * EToolDHontHit1.learningRateViews * (relevance_this - relevance_others)

                portfolioModel.loc[methodIdI] = portfolioModel.loc[methodIdI] + update_step

                # Apply constraints on maximal and minimal volumes of votes
                if portfolioModel.loc[methodIdI, 'votes'] < EToolDHontHit1.minVotesConst:
                    portfolioModel.loc[methodIdI, 'votes'] = EToolDHontHit1.minVotesConst
                elif portfolioModel.loc[methodIdI, 'votes'] > EToolDHontHit1.maxVotesConst:
                    portfolioModel.loc[methodIdI, 'votes'] = EToolDHontHit1.maxVotesConst

         # linearly normalizing to unit sum of votes
        sumMethodsVotes:float = portfolioModel.sum()
        for methodIdI in portfolioModel.index:
            portfolioModel.loc[methodIdI] = portfolioModel.loc[methodIdI] / sumMethodsVotes