#!/usr/bin/python3

from recommender.aRecommender import ARecommender #class

from recommenderDescription.recommenderDescription import RecommenderDescription #class

from portfolio.portfolioNeg1Meth import PortfolioNeg1Meth #class
from portfolio.portfolioHier import PortfolioHier #class

from portfolioDescription.aPortfolioDescription import APortfolioDescription #class
from portfolioDescription.portfolio1AggrDescription import Portfolio1AggrDescription #class

from history.aHistory import AHistory #class

from userBehaviourDescription.userBehaviourDescription import UserBehaviourDescription #class

from aggregation.negImplFeedback.aPenalization import APenalization #class

from aggregationDescription.aggregationDescription import AggregationDescription #class

from aggregation.aggrD21 import AggrD21 #class


class PortfolioHierDescription(APortfolioDescription):

    def __init__(self, portfolioID:str, recommID:str, recommDescr:RecommenderDescription,
                 p1AggrID:str, p1AggrDescr:Portfolio1AggrDescription, pHierDescr:AggregationDescription,
                 nImplFeedback:APenalization):
        if type(portfolioID) is not str:
            raise ValueError("Type of portfolioID is not str.")
        if type(recommID) is not str:
            raise ValueError("Type of argument recommID isn't str.")
        if type(recommDescr) is not RecommenderDescription:
            raise ValueError("Type of argument recommDescr isn't RecommenderDescription.")
        if type(p1AggrID) is not str:
            raise ValueError("Type of argument p1AggrID isn't str.")
        if type(p1AggrDescr) is not Portfolio1AggrDescription:
            raise ValueError("Type of argument p1AggrDescr isn't Portfolio1AggrDescription.")
        if not isinstance(nImplFeedback, APenalization):
            raise ValueError("Type of argument nImplFeedback isn't APenalization.")
        if type(pHierDescr) is not AggregationDescription:
            raise ValueError("Type of argument pHierDescr isn't AggregationDescription.")

        self._portfolioID:str = portfolioID
        self._recommID:str = recommID
        self._recommDescr:RecommenderDescription = recommDescr
        self._p1AggrID:str = p1AggrID
        self._p1AggrDescr:Portfolio1AggrDescription = p1AggrDescr
        self._pHierDescr:AggregationDescription = pHierDescr
        self._nImplFeedback = nImplFeedback

    def getPortfolioID(self):
        return self._portfolioID

    def exportPortfolio(self, batchID:str, history:AHistory):
        if type(batchID) is not str:
            raise ValueError("Type of argument batchID isn't str.")
        if type(batchID) is not str:
            raise ValueError("Type of argument jobID isn't str.")
        if not isinstance(history, AHistory):
            raise ValueError("Type of argument history isn't AHistory.")

        recommender:ARecommender = self._recommDescr.exportRecommender(batchID)

        p1Aggr = self._p1AggrDescr.exportPortfolio(batchID, history)

        pHier = self._pHierDescr.exportAggregation(history)

        return PortfolioHier(batchID, self.getPortfolioID(), recommender, self._recommID, self._recommDescr,
                             p1Aggr, pHier, self._nImplFeedback)
