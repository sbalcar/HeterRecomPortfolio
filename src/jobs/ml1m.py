#!/usr/bin/python3

from typing import List

from simulator.simulator import Simulator #class

from simulation.recommendToUser.simulatorOfPortfoliosRecommToUser import SimulationPortfolioToUser #class

from evaluationTool.dHont.eToolDHontHit1 import EToolDHontHit1 #class
from evaluationTool.banditTS.eToolBanditTSHit1 import EToolBanditTSHit1 #class

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

        argsSimulationDict:dict = {SimulationPortfolioToUser.ARG_ID: "ml1mDiv50",
                                   SimulationPortfolioToUser.ARG_WINDOW_SIZE: 3,
                                   SimulationPortfolioToUser.ARG_REPETITION_OF_RECOMMENDATION: 3,
                                   SimulationPortfolioToUser.ARG_NUMBER_OF_ITEMS: 20,
                                   SimulationPortfolioToUser.ARG_DIV_DATASET_PERC_SIZE: divisionDatasetPercentualSize}

        # simulation of portfolio
        # simulator:Simulator = Simulator(SimulationPortfoliosRecomToItemSeparatedUsers, ratingsDF, usersDF, itemsDF, uBehaviourDesc
        simulator: Simulator = Simulator(SimulationPortfolioToUser, argsSimulationDict, d.ratingsDF, d.usersDF,
                                         d.itemsDF, d.uBehaviourDesc)

        # evaluations:List[dict] = simulator.simulate([pDescCCB], [modelCCBDF], [EToolSingleMethod], [historyCCB])
        # evaluations:List[dict] = simulator.simulate([pDescW2V], [modelW2VDF], [EToolSingleMethod], [historyW2V])
        # evaluations:List[dict] = simulator.simulate([pDescTheMostPopular], [modelTheMostPopularDF], [EToolSingleMethod], [historyTheMostPopular])
        # evaluations:List[dict] = simulator.simulate([pDescDHont], [modelDHontDF], [EToolDHontHit1], [historyDHont])
        # evaluations:List[dict] = simulator.simulate([pDescBanditTS], [modelBanditTSDF], [EToolBanditTSHit1], [historyBanditTS])
        # evaluations: List[dict] = simulator.simulate([d.pDescBanditTS, d.pDescDHont],
        #                                             [d.modelBanditTSDF, d.modelDHontDF],
        #                                             [EToolBanditTSHit1, EToolDHontHit1],
        #                                             [d.historyBanditTS, d.historyDHont])

        evaluations: List[dict] = simulator.simulate([d.pDescBanditTS, d.pDescDHont, d.pDescDHontNF],
                                                     [d.modelBanditTSDF, d.modelDHontDF, d.modelDHontNFDF],
                                                     [EToolBanditTSHit1, EToolDHontHit1, EToolDHontHit1],
                                                     [d.historyBanditTS, d.historyDHont, d.historyDHontNF])
