#!/usr/bin/python3

from recommender.aRecommender import ARecommender #class

from recommenderDescription.recommenderDescription import RecommenderDescription #class

from portfolio.portfolio1Meth import Portfolio1Meth #class

from portfolioDescription.aPortfolioDescription import APortfolioDescription #class

from history.aHistory import AHistory #class

from userBehaviourDescription.userBehaviourDescription import UserBehaviourDescription #class


class Portfolio1MethDescription(APortfolioDescription):

    def __init__(self, portfolioID:str, recommID:str, recommDescr:RecommenderDescription):
        if type(portfolioID) is not str:
            raise ValueError("Type of portfolioID is not str.")
        if type(recommID) is not str:
            raise ValueError("Type of argument recommID isn't str.")
        if type(recommDescr) is not RecommenderDescription:
            raise ValueError("Type of argument recommDescr isn't RecommenderDescription.")

        self._portfolioID:str = portfolioID
        self._recommID:str = recommID
        self._recommDescr:RecommenderDescription = recommDescr

    def getPortfolioID(self):
        return self._recommID

    def exportPortfolio(self, batchID:str, history:AHistory):
        if type(batchID) is not str:
            raise ValueError("Type of argument batchID isn't str.")
        if not isinstance(history, AHistory):
            raise ValueError("Type of argument history isn't AHistory.")

        recommender:ARecommender = self._recommDescr.exportRecommender(batchID)
        return Portfolio1Meth(batchID, self.getPortfolioID(), recommender, self._recommID, self._recommDescr)