#!/usr/bin/python3

import os
from typing import List

from datasets.aDataset import ADataset #class
from datasets.datasetML import DatasetML #class
from datasets.datasetRetailrocket import DatasetRetailRocket #class

from datasets.ml.ratings import Ratings #class
from datasets.ml.behaviours import Behaviours #class
from datasets.retailrocket.behavioursRR import BehavioursRR #class
from datasets.slantour.behavioursST import BehavioursST #class


from datasets.datasetST import DatasetST #class

from pandas.core.frame import DataFrame #class

from simulator.simulator import Simulator #class

from simulation.tools.modelOfIndexes import ModelOfIndexes #class
from simulation.simulationML import SimulationML #class
from simulation.simulationRR import SimulationRR #class
from simulation.simulationST import SimulationST #class

from pandas.core.frame import DataFrame #class

from portfolioDescription.portfolio1MethDescription import Portfolio1MethDescription #class

from input.inputRecomMLDefinition import InputRecomMLDefinition #class

from portfolioDescription.aPortfolioDescription import APortfolioDescription #class

from evaluationTool.evalToolSingleMethod import EToolSingleMethod #class

from recommenderDescription.recommenderDescription import RecommenderDescription #class

from input.inputSimulatorDefinition import InputSimulatorDefinition #class

from history.historyHierDF import HistoryHierDF #class

import pandas as pd
from input.aBatch import BatchParameters #class
from input.aBatchML import ABatchML #class



def test01():

    df:DataFrame = DataFrame({'$a':[101,101,101,102, 102], '$b':[9990,9991,9992,9993,9994], '$c':[2,4,5,4,1]}, index=[31,32,33,34,35])
    df.columns = [Ratings.COL_USERID, Ratings.COL_MOVIEID, Ratings.COL_RATING]
    print(df)
    print("--------------")
    print()

    relevantDFIndexes:List[int] = [31,32]
    relevantDFIndexes:List[int] = list(df.index)
    print(relevantDFIndexes)
    m:ModelOfIndexes = ModelOfIndexes(df, relevantDFIndexes, Ratings)

    print("Get next index for index 31 (101, 9990):")
    a1:int = m.getNextDFIndexes(31, 3)
    print(a1)

    print("Get next index for 32 (101, 9991):")
    a2:int = m.getNextDFIndexes(32, 1)
    print(a2)

    print("Get next index for 33 (101, 9992):")
    a3:int = m.getNextDFIndexes(33, 1)
    print(a3)

    print("Get next index for 34 (102, 9993):")
    a4:int = m.getNextDFIndexes(34, 1)
    print(a4)
    print("--------------")

    print("Get next relevant itemIDS for 31 (102, 9993):")
    a5:int = m.getNextRelevantItemIDs(31, 3)
    print(a5)

    print("Get next relevant itemIDS for 34 (102, 9994):")
    a6:int = m.getNextRelevantItemIDs(34, 1)
    print(a6)


if __name__ == "__main__":
    os.chdir("..")

    test01()
