#!/usr/bin/python3

from input.input import Input #class

from portfolio.portfolio1Aggr import Portfolio1Aggr #class
from portfolioDescription.portfolio1AggrDescription import Portfolio1AggrDescription #class

from evaluationOfRecommender.evaluationOfRecommenders import EvaluationOfRecommenders #class


def recommendation():

  # number of recommended items
  numberOfItems:int = 20
  print(numberOfItems)

  portfolioDescr:Portfolio1AggrDescription
  evaluationOfRecommenders:EvaluationOfRecommenders
  portfolioDescr, evaluationOfRecommenders = Input.input01()
  portfolioDescr, evaluationOfRecommenders = Input.input02()

  portfolio:Portfolio1Aggr = Portfolio1Aggr(portfolioDescr)

  itemIDs:list[int] = portfolio.run(evaluationOfRecommenders, numberOfItems)
  print(itemIDs)
