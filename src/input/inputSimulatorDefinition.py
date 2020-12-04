#!/usr/bin/python3

from typing import List

from recommenderDescription.recommenderDescription import RecommenderDescription #class

from recommender.recommenderTheMostPopular import RecommenderTheMostPopular #class
from recommender.recommenderCosineCB import RecommenderCosineCB #class
from recommender.recommenderW2V import RecommenderW2V #class

from datasets.ratings import Ratings #class
from datasets.users import Users #class
from datasets.items import Items #class
from datasets.behaviours import Behaviours #class

from configuration.configuration import Configuration #class

from pandas.core.frame import DataFrame #class

from portfolioDescription.aPortfolioDescription import APortfolioDescription #class

from aggregation.negImplFeedback.aPenalization import APenalization #class
from aggregation.negImplFeedback.penalUsingReduceRelevance import PenalUsingReduceRelevance #class
from aggregation.negImplFeedback.penalUsingReduceRelevance import penaltyStatic #function
from aggregation.negImplFeedback.penalUsingReduceRelevance import penaltyLinear #function

from simulation.simulationOfPortfoliosRecommToUser import SimulationPortfolioToUser #class

from simulator.simulator import Simulator #class



class InputSimulatorDefinition:

    @staticmethod
    def exportSimulatorML1M(batchID:str, divisionDatasetPercentualSize:int, uBehaviourID:str, repetition:int):

        argsSimulationDict:dict = {SimulationPortfolioToUser.ARG_WINDOW_SIZE: 5,
                                   SimulationPortfolioToUser.ARG_REPETITION_OF_RECOMMENDATION: repetition,
                                   SimulationPortfolioToUser.ARG_NUMBER_OF_RECOMM_ITEMS: 100,
                                   SimulationPortfolioToUser.ARG_NUMBER_OF_AGGR_ITEMS: 20,
                                   SimulationPortfolioToUser.ARG_DIV_DATASET_PERC_SIZE: divisionDatasetPercentualSize}

        # dataset reading
        ratingsDF:DataFrame = Ratings.readFromFileMl1m()
        usersDF:DataFrame = Users.readFromFileMl1m()
        itemsDF:DataFrame = Items.readFromFileMl1m()

        behaviourFile:str = Behaviours.getFile(uBehaviourID)
        behavioursDF:DataFrame = Behaviours.readFromFileMl1m(behaviourFile)

        # simulation of portfolio
        simulator:Simulator = Simulator(batchID, SimulationPortfolioToUser, argsSimulationDict,
                                        ratingsDF, usersDF, itemsDF, behavioursDF)

        return simulator