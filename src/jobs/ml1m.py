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


def ml1m(prefix:str, divisionDatasetPercentualSize:int, repetition:int):

        d = InputsML1MDefinition
        jobID:str = prefix + "ml1mDiv" + str(divisionDatasetPercentualSize) + "R" + str(repetition)

        argsSimulationDict:dict = {SimulationPortfolioToUser.ARG_WINDOW_SIZE: 5,
                                   SimulationPortfolioToUser.ARG_REPETITION_OF_RECOMMENDATION: repetition,
                                   SimulationPortfolioToUser.ARG_NUMBER_OF_RECOMM_ITEMS: d.numberOfRecommItems,
                                   SimulationPortfolioToUser.ARG_NUMBER_OF_AGGR_ITEMS: d.numberOfAggrItems,
                                   SimulationPortfolioToUser.ARG_DIV_DATASET_PERC_SIZE: divisionDatasetPercentualSize}

        pDescs:List[APortfolioDescription] = []
        models:List[DataFrame] = []
        evalTools:List = []
        histories:List = []

        # single CB portfolios
        pDescs += [d.pDescTheMostPopular, d.pDescCBmax, d.pDescCBwindow10]
        models += [d.modelTheMostPopularDF, d.modelCBmaxDF, d.modelCBwindow10DF]
        evalTools += [EToolSingleMethod, EToolSingleMethod, EToolSingleMethod]
        histories += [HistoryHierDF("TMPopular"), HistoryHierDF("CBmax"), HistoryHierDF("CBwindow10")]

        # single W2V portfolios
        pDescs += [d.pDescW2vPosnegMean, d.pDescW2vPosnegWindow3]
        models += [d.modelW2vPosnegMeanDF, d.modelW2vPosnegWindow3DF]
        evalTools += [EToolSingleMethod, EToolSingleMethod]
        histories += [HistoryHierDF("W2vPosnegMax"), HistoryHierDF("CBwindow10")]


        # BanditTS portfolios
        pDescs += [d.pDescBanditTS]
        models += [d.modelBanditTSDF]
        evalTools += [EvalToolBanditTS]
        histories += [HistoryHierDF("BanditTS")]

        # DHont portfolios
        pDescs += [d.pDescDHontFixed, d.pDescDHontRoulette, d.pDescDHontRoulette3]
        models += [d.modelDHontFixedDF, d.modelDHontRouletteDF, d.modelDHontRoulette3DF]
        evalTools += [EvalToolDHont, EvalToolDHont, EvalToolDHont]
        histories += [HistoryHierDF("DHontFixed"), HistoryHierDF("DHontRoulette"), HistoryHierDF("DHontRoulette3")]

        # NegDHontNF portfolios1
        pDescs += [d.pDescNegDHontOStat08HLin1002]
        models += [d.modelNegDHontOStat08HLin1002DF]
        evalTools += [EvalToolDHont]
        histories += [HistoryHierDF("NegDHontOStat08HLin1002")]

        # NegDHontNF portfolios2
        pDescs += [d.pDescNegDHontOLin0802HLin1002]
        models += [d.modelNegDHontOLin0802HLin1002DF]
        evalTools += [EvalToolDHont]
        histories += [HistoryHierDF("NegDHontOLin0802HLin1002")]

        # simulation of portfolio
        simulator:Simulator = Simulator(jobID, SimulationPortfolioToUser, argsSimulationDict, d.ratingsDF, d.usersDF,
                                         d.itemsDF, d.uBehaviourDesc)

        evaluations:List[dict] = simulator.simulate(pDescs, models, evalTools, histories)
