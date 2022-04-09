#!/usr/bin/python3

from datasets.aDataset import ADataset #class
from datasets.datasetML import DatasetML #class
from datasets.datasetRetailRocket import DatasetRetailRocket #class
from datasets.datasetST import DatasetST #class

from datasets.ml.ratings import Ratings #class
from datasets.ml.users import Users #class
from datasets.ml.items import Items #class
from datasets.ml.behavioursML import BehavioursML #class
from datasets.retailrocket.behavioursRR import BehavioursRR #class
from datasets.slantour.behavioursST import BehavioursST #class

from pandas.core.frame import DataFrame #class

from simulation.simulationML import SimulationML #class
from simulation.simulationRR import SimulationRR #class
from simulation.simulationST import SimulationST #class

from simulator.simulator import Simulator #class



class InputSimulatorDefinition:

    numberOfAggrItems:int = 20

    def exportSimulatorML1M(self, batchID:str, divisionDatasetPercentualSize:int, uBehaviourID:str, repetition:int):

        argsSimulationDict:dict = {SimulationML.ARG_WINDOW_SIZE: 5,
                                   SimulationML.ARG_RECOM_REPETITION_COUNT: repetition,
                                   SimulationML.ARG_NUMBER_OF_RECOMM_ITEMS: 100,
                                   SimulationML.ARG_NUMBER_OF_AGGR_ITEMS: self.numberOfAggrItems,
                                   SimulationML.ARG_DIV_DATASET_PERC_SIZE: divisionDatasetPercentualSize,
                                   SimulationML.ARG_HISTORY_LENGTH: 10}

        # dataset reading
        dataset:ADataset = DatasetML.readDatasets()

        behaviourFile:str = BehavioursML.getFile(uBehaviourID)
        behavioursDF:DataFrame = BehavioursML.readFromFileMl1m(behaviourFile)

        # simulation of portfolio
        simulator:Simulator = Simulator(batchID, SimulationML, argsSimulationDict, dataset, behavioursDF)

        return simulator


    @staticmethod
    def exportSimulatorRetailRocket(self, batchID:str, divisionDatasetPercentualSize:int, uBehaviourID:str, repetition:int):

        argsSimulationDict:dict = {SimulationML.ARG_WINDOW_SIZE: 5,
                                   SimulationML.ARG_RECOM_REPETITION_COUNT: repetition,
                                   SimulationML.ARG_NUMBER_OF_RECOMM_ITEMS: 100,
                                   SimulationML.ARG_NUMBER_OF_AGGR_ITEMS: self.numberOfAggrItems,
                                   SimulationML.ARG_DIV_DATASET_PERC_SIZE: divisionDatasetPercentualSize,
                                   SimulationML.ARG_HISTORY_LENGTH: 10}

        # dataset reading
        dataset:ADataset = DatasetRetailRocket.readDatasetsWithFilter(minEventCount=50)

        behaviourFile:str = BehavioursRR.getFile(uBehaviourID)
        behavioursDF:DataFrame = BehavioursRR.readFromFileRR(behaviourFile)

        # simulation of portfolio
        simulator:Simulator = Simulator(batchID, SimulationRR, argsSimulationDict, dataset, behavioursDF)

        return simulator


    @staticmethod
    def exportSimulatorSlantour(self, batchID:str, divisionDatasetPercentualSize:int, uBehaviourID:str, repetition:int):

        argsSimulationDict:dict = {SimulationML.ARG_WINDOW_SIZE: 5,
                                   SimulationML.ARG_RECOM_REPETITION_COUNT: repetition,
                                   SimulationML.ARG_NUMBER_OF_RECOMM_ITEMS: 100,
                                   SimulationML.ARG_NUMBER_OF_AGGR_ITEMS: self.numberOfAggrItems,
                                   SimulationML.ARG_DIV_DATASET_PERC_SIZE: divisionDatasetPercentualSize,
                                   SimulationML.ARG_HISTORY_LENGTH: 10}

        # dataset reading
        dataset:ADataset = DatasetST.readDatasets()

        behaviourFile:str = BehavioursST.getFile(uBehaviourID)
        behavioursDF:DataFrame = BehavioursST.readFromFileST(behaviourFile)

        # simulation of portfolio
        simulator:Simulator = Simulator(batchID, SimulationST, argsSimulationDict, dataset, behavioursDF)

        return simulator
