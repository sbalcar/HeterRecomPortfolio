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

def test01():
    print("")

    numberOfItems:int = 5

    u1:List[bool] = observationalLinearProbabilityFnc(0.1, 0.9, numberOfItems)
    print(u1)

    # return 0.54 * np.power(x, -0.48)
    u2:List[bool] = observationalPowerLawFnc(0.54, -0.48, numberOfItems)
    print(u2)


if __name__ == "__main__":
    os.chdir("..")

    test01()