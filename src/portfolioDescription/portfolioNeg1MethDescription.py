#!/usr/bin/python3

from recommender.aRecommender import ARecommender #class

from recommenderDescription.recommenderDescription import RecommenderDescription #class

from portfolio.portfolioNeg1Meth import PortfolioNeg1Meth #class

from portfolioDescription.aPortfolioDescription import APortfolioDescription #class

from history.aHistory import AHistory #class

from userBehaviourDescription.userBehaviourDescription import UserBehaviourDescription #class

from aggregation.negImplFeedback.aPenalization import APenalization #class


class PortfolioNeg1MethDescription(APortfolioDescription):

    def __init__(self, portfolioID:str, recommID:str, recommDescr:RecommenderDescription, nImplFeedback:APenalization):
        if type(portfolioID) is not str:
            raise ValueError("Type of portfolioID is not str.")
        if type(recommID) is not str:
            raise ValueError("Type of argument recommID isn't str.")
        if type(recommDescr) is not RecommenderDescription:
            raise ValueError("Type of argument recommDescr isn't RecommenderDescription.")
        if not isinstance(nImplFeedback, APenalization):
            raise ValueError("Type of argument nImplFeedback isn't APenalization.")

        self._portfolioID:str = portfolioID
        self._recommID:str = recommID
        self._recommDescr:RecommenderDescription = recommDescr
        self._nImplFeedback = nImplFeedback

    def getPortfolioID(self):
        return self._portfolioID

    def exportPortfolio(self, jobID:str, history:AHistory):
        if type(jobID) is not str:
            raise ValueError("Type of argument jobID isn't str.")
        if not isinstance(history, AHistory):
            raise ValueError("Type of argument history isn't AHistory.")

        recommender:ARecommender = self._recommDescr.exportRecommender(jobID)
        return PortfolioNeg1Meth(recommender, self._portfolioID, self._recommDescr, self._nImplFeedback)
