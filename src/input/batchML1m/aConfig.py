#!/usr/bin/python3

from typing import List

from pandas.core.frame import DataFrame #class

from simulator.simulator import Simulator #class

from simulation.recommendToUser.simulatorOfPortfoliosRecommToUser import SimulationPortfolioToUser #class

from evaluationTool.evalToolSingleMethod import EToolSingleMethod #class
from evaluationTool.evalToolDHont import EvalToolDHont #class
from evaluationTool.evalToolBanditTS import EvalToolBanditTS #class

from input.inputsML1MDefinition import InputsML1MDefinition #class

from portfolioDescription.aPortfolioDescription import APortfolioDescription #class

from history.aHistory import AHistory #class
from history.historyHierDF import HistoryHierDF #class


def ml1m(prefix:str, divisionDatasetPercentualSize:int, repetition:int,
         pDescs:List[APortfolioDescription], models:List[DataFrame], evalTools:List):

        d = InputsML1MDefinition
        batchID:str = prefix + "ml1mDiv" + str(divisionDatasetPercentualSize) + "R" + str(repetition)

        argsSimulationDict:dict = {SimulationPortfolioToUser.ARG_WINDOW_SIZE: 5,
                                   SimulationPortfolioToUser.ARG_REPETITION_OF_RECOMMENDATION: repetition,
                                   SimulationPortfolioToUser.ARG_NUMBER_OF_RECOMM_ITEMS: d.numberOfRecommItems,
                                   SimulationPortfolioToUser.ARG_NUMBER_OF_AGGR_ITEMS: d.numberOfAggrItems,
                                   SimulationPortfolioToUser.ARG_DIV_DATASET_PERC_SIZE: divisionDatasetPercentualSize}

        # simulation of portfolio
        simulator:Simulator = Simulator(batchID, SimulationPortfolioToUser, argsSimulationDict,
                                        d.ratingsDF, d.usersDF, d.itemsDF, d.behavioursDF)

        evaluations:List[dict] = simulator.simulate(pDescs, models, evalTools, HistoryHierDF)
