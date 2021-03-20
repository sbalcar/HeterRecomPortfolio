#!/usr/bin/python3

import os
from typing import List
from typing import Dict

from datasets.aDataset import ADataset #class
from datasets.datasetML import DatasetML #class
from datasets.datasetRetailrocket import DatasetRetailRocket #class
from datasets.datasetST import DatasetST #class

from datasets.ml.ratings import Ratings #class
from datasets.ml.behavioursML import BehavioursML #class
from datasets.retailrocket.behavioursRR import BehavioursRR #class
from datasets.slantour.behavioursST import BehavioursST #class

from pandas.core.frame import DataFrame #class

from simulator.simulator import Simulator #class

from simulation.tools.modelOfIndexes import ModelOfIndexes #class
from simulation.simulationML import SimulationML #class
from simulation.simulationRR import SimulationRR #class
from simulation.simulationST import SimulationST #class

from pandas.core.frame import DataFrame #class

from portfolioDescription.portfolio1MethDescription import Portfolio1MethDescription #class

from input.inputRecomMLDefinition import InputRecomMLDefinition #class
from input.inputRecomSTDefinition import InputRecomSTDefinition #class

from portfolioDescription.aPortfolioDescription import APortfolioDescription #class

from evaluationTool.evalToolDoNothing import EToolDoNothing #class

from recommenderDescription.recommenderDescription import RecommenderDescription #class

from input.inputSimulatorDefinition import InputSimulatorDefinition #class

from history.historyHierDF import HistoryHierDF #class

import pandas as pd
from input.inputABatchDefinition import InputABatchDefinition
from input.aBatchDefinitionML import ABatchDefinitionML #class

from userBehaviourDescription.userBehaviourDescription import observationalPowerLawFnc #function
from userBehaviourDescription.userBehaviourDescription import observationalLinearProbabilityFnc #function

from recommender.tools.toolMMR import ToolMMR #class

def test01():
    print("")

    dataset:DatasetML = DatasetML.readDatasets()

    toolMMR:ToolMMR = ToolMMR()
    toolMMR.init(dataset)

    inputRecommendation:Dict[int,float] = {
        2390: 0.585729,
        1858: 0.513299,
        2169: 0.471460,
        3125: 0.376679,
        3624: 0.369205
    }

    lambda_:float = 0.5
    numberOfItems:int = 20
    r = toolMMR.mmr_sorted(lambda_, pd.Series(inputRecommendation), numberOfItems)
    print(r)

if __name__ == "__main__":
    os.chdir("..")

    test01()