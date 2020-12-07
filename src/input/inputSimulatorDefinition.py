#!/usr/bin/python3

from datasets.ml.ratings import Ratings #class
from datasets.ml.users import Users #class
from datasets.ml.items import Items #class
from datasets.ml.behaviours import Behaviours #class

from pandas.core.frame import DataFrame #class

from simulation.simulationOfPortfoliosRecommToUser import SimulationPortfolioToUser #class

from simulator.simulator import Simulator #class



class InputSimulatorDefinition:
    numberOfAggrItems:int = 20

    @staticmethod
    def exportSimulatorML1M(batchID:str, divisionDatasetPercentualSize:int, uBehaviourID:str, repetition:int):

        argsSimulationDict:dict = {SimulationPortfolioToUser.ARG_WINDOW_SIZE: 5,
                                   SimulationPortfolioToUser.ARG_REPETITION_OF_RECOMMENDATION: repetition,
                                   SimulationPortfolioToUser.ARG_NUMBER_OF_RECOMM_ITEMS: 100,
                                   SimulationPortfolioToUser.ARG_NUMBER_OF_AGGR_ITEMS: InputSimulatorDefinition.numberOfAggrItems,
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