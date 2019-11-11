#!/usr/bin/python3

import numpy as np
import pandas as pd

from configuration.arguments import Arguments #class
from configuration.argument import Argument #class

from input.input import Input #class

from portfolio.portfolio import Portfolio #class

from aggregation.aggregationDescription import AggregationDescription #class

from aggregation.aggrElections import AggrElections #class

from recommendation.resultOfRecommendation import ResultOfRecommendation #class
from recommendation.resultsOfRecommendations import ResultsOfRecommendations #class

from evaluationOfRecommender.evaluationOfRecommenders import EvaluationOfRecommenders #class


def recommendation():

  # number of recommended items
  numberOfItems = 20
  print(numberOfItems)

  # (PortfolioDescription, EvaluationOfRecommenders)
  portfolioDescr, evaluationOfRecommenders = Input.input01()
  portfolioDescr, evaluationOfRecommenders = Input.input02()

  #portfolio:Portfolio
  portfolio = Portfolio(portfolioDescr)

  # itemIDs:list<int>
  itemIDs = portfolio.run(evaluationOfRecommenders, numberOfItems)
  print(itemIDs)

