#!/usr/bin/python3

from typing import List

from pandas.core.frame import DataFrame #class

from simulator.simulator import Simulator #class

from simulation.recommendToUser.simulatorOfPortfoliosRecommToUser import SimulationPortfolioToUser #class

from evaluationTool.evalToolSingleMethod import EToolSingleMethod #class
from evaluationTool.evalToolDHont import EvalToolDHont #class
from evaluationTool.evalToolBanditTS import EvalToolBanditTS #class

from portfolioDescription.aPortfolioDescription import APortfolioDescription #class

from history.aHistory import AHistory #class
from history.historyHierDF import HistoryHierDF #class

from datasets.ratings import Ratings #class
from datasets.users import Users #class
from datasets.items import Items #class
from datasets.behaviours import Behaviours #class


class AML1MConf:

    windowSize:int = 5
    numberOfAggrItems:int = 20
    numberOfRecommItems:int = 100

    def __init__(self, batchID:str, divisionDatasetPercentualSize:int, uBehaviour:str, repetition:int):
        self.batchID = batchID
        self.divisionDatasetPercentualSize = divisionDatasetPercentualSize
        self.uBehaviour = uBehaviour
        self.repetition = repetition

        self.datasetID:str = "ml1m" + "Div" + str(divisionDatasetPercentualSize)


    def run(self, pDesc:APortfolioDescription, model:DataFrame, evalTool):

        argsSimulationDict:dict = {SimulationPortfolioToUser.ARG_WINDOW_SIZE: AML1MConf.windowSize,
                                   SimulationPortfolioToUser.ARG_REPETITION_OF_RECOMMENDATION: self.repetition,
                                   SimulationPortfolioToUser.ARG_NUMBER_OF_RECOMM_ITEMS: AML1MConf.numberOfRecommItems,
                                   SimulationPortfolioToUser.ARG_NUMBER_OF_AGGR_ITEMS: AML1MConf.numberOfAggrItems,
                                   SimulationPortfolioToUser.ARG_DIV_DATASET_PERC_SIZE: self.divisionDatasetPercentualSize,
                                   SimulationPortfolioToUser.AGR_USER_BEHAVIOUR_DFINDEX: self.uBehaviour}

        # dataset reading
        ratingsDF:DataFrame = Ratings.readFromFileMl1m()
        usersDF:DataFrame = Users.readFromFileMl1m()
        itemsDF:DataFrame = Items.readFromFileMl1m()
        behavioursDF:DataFrame = Behaviours.readFromFileMl1m()

        # simulation of portfolio
        simulator:Simulator = Simulator(self.batchID, SimulationPortfolioToUser, argsSimulationDict,
                                        ratingsDF, usersDF, itemsDF, behavioursDF)

        pDescs:List[APortfolioDescription] = [pDesc]
        models:List[DataFrame] = [model]
        evalTools:List = [evalTool]

        evaluations:List[dict] = simulator.simulate(pDescs, models, evalTools, HistoryHierDF)
