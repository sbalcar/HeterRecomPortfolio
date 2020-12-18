#!/usr/bin/python3

from typing import List

from pandas.core.frame import DataFrame #class

from simulation.simulationML import SimulationML #class

from datasets.aDataset import ADataset #class

from history.aHistory import AHistory #class

from portfolioDescription.aPortfolioDescription import APortfolioDescription #class

from evaluationTool.aEvalTool import AEvalTool #class

from simulation.aSequentialSimulation import ASequentialSimulation #class


class Simulator:

    def __init__(self, jobID:str, simulatorClass, argumentsDict:dict, dataset:ADataset, behavioursDF:DataFrame):
        if type(jobID) is not str:
            raise ValueError("Argument jobID isn't type str.")
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")
        if not isinstance(dataset, ADataset):
            raise ValueError("Argument dataset isn't type ADataset.")
        if type(behavioursDF) is not DataFrame:
            raise ValueError("Argument behavioursDF isn't type DataFrame.")

        self._simulation:ASequentialSimulation = simulatorClass(
            jobID, dataset, behavioursDF, argumentsDict)


    def simulate(self, pDescs:List[APortfolioDescription], portModels:List[DataFrame], eTools:List[AEvalTool], historyClass):

        if type(pDescs) is not list:
            raise ValueError("Argument histories isn't type list.")
        for pDescI in pDescs:
            if not isinstance(pDescI, APortfolioDescription):
               raise ValueError("Argument pDescs don't contain APortfolioDescription.")

        if type(portModels) is not list:
            raise ValueError("Argument portModels isn't type list.")
        for portModI in portModels:
            if type(portModI) is not DataFrame:
               raise ValueError("Argument portModels don't contain DataFrame.")

        if type(eTools) is not list:
            raise ValueError("Argument etools isn't type list.")

        #if type(histories) is not list:
        #    raise ValueError("Argument histories isn't type list.")
        #for historyI in histories:
        #    if not isinstance(historyI, AHistory):
        #       raise ValueError("Argument histories don't contain AHistory.")

        histories:List[AHistory] = []
        for i in range(len(pDescs)):

            pDescI:APortfolioDescription = pDescs[i]
            portfolioIdI:str = pDescI.getPortfolioID()

            historyI:AHistory = historyClass(portfolioIdI)
            histories.append(historyI)


        evaluations:List[dict] = self._simulation.run(pDescs, portModels, eTools, histories)


        i:int
        for i in range(len(pDescs)):

            pDescI:APortfolioDescription = pDescs[i]

            print()
            portfolioIdI:str = pDescI.getPortfolioID()
            print("PortfolioIdI: " + portfolioIdI)

            print()
            eToolClassI:str = eTools[i]
            print("EToolClass " + portfolioIdI)
            print(eToolClassI)

            print()
            portModelI:DataFrame = portModels[i]
            print("Model of " + portfolioIdI)
            print(portModelI)

            print()
            historyI:AHistory = histories[i]
            print("History of " + portfolioIdI)
            historyI.print()

        ids:List[str] = [pDescI.getPortfolioID() for pDescI in pDescs]

        print()
        print("ids: " + str(ids))
        print("Evaluations: " + str(evaluations))
        return evaluations