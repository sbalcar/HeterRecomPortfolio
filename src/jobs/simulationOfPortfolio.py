#!/usr/bin/python3

from typing import List

from pandas.core.series import Series #class

from recommender.description.recommenderDescription import RecommenderDescription #class

from recommender.aRecommender import ARecommender #class

from recommender.recommenderTheMostPopular import RecommenderTheMostPopular #class
from recommender.dummy.recommenderDummyRedirector import RecommenderDummyRedirector #class

from configuration.arguments import Arguments #class
from configuration.argument import Argument #class

from datasets.ratings import Ratings #class
from datasets.rating import Rating #class

from datasets.users import Users #class

from datasets.items import Items #class

from portfolio.portfolioDescription import PortfolioDescription #class
from portfolio.portfolio import Portfolio #class

from aggregation.aggregationDescription import AggregationDescription #class
from aggregation.aggrDHont import AggrDHont #class

from recommendation.resultOfRecommendation import ResultOfRecommendation #class

import numpy as np
import pandas as pd

from pandas.core.frame import DataFrame #class

from simulation.simulationOfPersonalisedPortfolio import SimulationOfPersonalisedPortfolio #class
from simulation.simulationOfNonPersonalisedPortfolio import SimulationOfNonPersonalisedPortfolio #class


def simulationOfPortfolio():

    # dataset reading
    ratingsDF: DataFrame = Ratings.readFromFileMl100k()
    usersDF: DataFrame = Users.readFromFileMl100k()
    itemsDF: DataFrame = Items.readFromFileMl100k()

    #ratingsDF: DataFrame = Ratings.readFromFileMl1m()
    #usersDF: DataFrame = Users.readFromFileMl1m()
    #itemsDF: DataFrame = Items.readFromFileMl1m()

    #ratingsDF: DataFrame = Ratings.readFromFileMl10M100K()
    #usersDF: DataFrame = Users.readFromFileMl10M100K()
    #itemsDF: DataFrame = Items.readFromFileMl10M100K()

    # portfolio definiton
    rDescTheMostPopular: RecommenderDescription = RecommenderDescription(RecommenderTheMostPopular, Arguments([]))
    rDescDummyRedirector: RecommenderDescription = RecommenderDescription(RecommenderDummyRedirector,
                            Arguments([Argument(RecommenderDummyRedirector.ARG_RESULT, ResultOfRecommendation(list(range(1, 21)),[0.05] * 20))]))

    aggregationDesc: AggregationDescription = AggregationDescription(AggrDHont, Arguments([]))

    portfolioDesc: PortfolioDescription = PortfolioDescription(
            ["RecommenderTheMostPopular", "RecommenderDummyRedirector"],
            [rDescTheMostPopular, rDescDummyRedirector],
            aggregationDesc)


    # simulation of portfolio
    #simulation:SimulationOfPersonalisedPortfolio = SimulationOfPersonalisedPortfolio(
    simulation:SimulationOfNonPersonalisedPortfolio = SimulationOfNonPersonalisedPortfolio(
            ratingsDF, usersDF, itemsDF, repetitionOfRecommendation=1, numberOfItems=20)
    simulation.run(portfolioDesc)