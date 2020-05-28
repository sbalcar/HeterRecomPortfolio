#!/usr/bin/python3

from typing import List

from simulator.simulator import Simulator #class

from simulation.recommendToUser.simulatorOfPortfoliosRecommToUser import SimulationPortfolioToUser #class

from evaluationTool.evalToolDHont import EvalToolDHont #class
from evaluationTool.evalToolBanditTS import EvalToolBanditTS #class

from input.inputsML1MDefinition import InputsML1MDefinition #class


def ml1mDiv50():
    divisionDatasetPercentualSize:int = 50

    __ml1m(divisionDatasetPercentualSize)

def ml1mDiv60():
    divisionDatasetPercentualSize:int = 60

    __ml1m(divisionDatasetPercentualSize)

def ml1mDiv70():
    divisionDatasetPercentualSize:int = 70

    __ml1m(divisionDatasetPercentualSize)

def ml1mDiv80():
    divisionDatasetPercentualSize:int = 80

    __ml1m(divisionDatasetPercentualSize)

def ml1mDiv90():
    divisionDatasetPercentualSize:int = 90

    __ml1m(divisionDatasetPercentualSize)



def __ml1m(divisionDatasetPercentualSize:int):

        d = InputsML1MDefinition
        jobID:str = "ml1mDiv" + str(divisionDatasetPercentualSize)

        argsSimulationDict:dict = {SimulationPortfolioToUser.ARG_WINDOW_SIZE: 5,
                                   SimulationPortfolioToUser.ARG_REPETITION_OF_RECOMMENDATION: 1,
                                   SimulationPortfolioToUser.ARG_NUMBER_OF_RECOMM_ITEMS: d.numberOfRecommItems,
                                   SimulationPortfolioToUser.ARG_NUMBER_OF_AGGR_ITEMS: d.numberOfAggrItems,
                                   SimulationPortfolioToUser.ARG_DIV_DATASET_PERC_SIZE: divisionDatasetPercentualSize}

        # simulation of portfolio
        simulator:Simulator = Simulator(jobID, SimulationPortfolioToUser, argsSimulationDict, d.ratingsDF, d.usersDF,
                                         d.itemsDF, d.uBehaviourDesc)

        #evaluations:List[dict] = simulator.simulate([d.pDescTheMostPopular], [d.modelTheMostPopularDF], [EToolSingleMethod], [HistoryHierDF("historyTheMostPopular")])
        #evaluations:List[dict] = simulator.simulate([d.pDescCCB], [d.modelCCBDF], [EToolSingleMethod], [d.historyCCB])
        #evaluations:List[dict] = simulator.simulate([pDescW2V], [modelW2VDF], [EToolSingleMethod], [historyW2V])
        #evaluations:List[dict] = simulator.simulate([pDescBanditTS], [modelBanditTSDF], [EToolBanditTSHit1], [historyBanditTS])
        #evaluations:List[dict] = simulator.simulate([d.pDescDHont], [d.modelDHontDF], [EvalToolDHont], [d.historyDHont])
        #evaluations:List[dict] = simulator.simulate([d.pDescDHontNF], [d.modelDHontNFDF], [EToolDHontHit1], [d.historyDHontNF])
        #evaluations: List[dict] = simulator.simulate([d.pDescBanditTS, d.pDescDHont],
        #                                            [d.modelBanditTSDF, d.modelDHontDF],
        #                                            [EToolBanditTSHit1, EToolDHontHit1],
        #                                            [d.historyBanditTS, d.historyDHont])

        evaluations: List[dict] = simulator.simulate([d.pDescBanditTS, d.pDescDHont, d.pDescDHontNF],
                                                     [d.modelBanditTSDF, d.modelDHontDF, d.modelDHontNFDF],
                                                     [EvalToolBanditTS, EvalToolDHont, EvalToolDHont],
                                                     [d.historyBanditTS, d.historyDHont, d.historyDHontNF])
