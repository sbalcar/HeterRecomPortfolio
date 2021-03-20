#!/usr/bin/python3

import os
import os.path
from typing import List #class
from typing import Dict #class

from configuration.configuration import Configuration #class
from pandas.core.frame import DataFrame #class

from evaluationTool.aEvalTool import AEvalTool #class

from portfolioDescription.aPortfolioDescription import APortfolioDescription #class

from simulator.simulator import Simulator #class

from history.aHistory import AHistory #class



class Batch:

    def __init__(self, batchDefinitionClass, batchID:str, jobID:str):
        self.batchDefinitionClass = batchDefinitionClass
        self.batchID:str = batchID
        self.jobID = jobID

    def run(self):
        self.batchDefinitionClass.run(self.batchID, self.jobID)


    def exists(self):
        job:str = str(self.batchDefinitionClass) + self.jobID

        fname:str = Configuration.resultsDirectory + os.sep + self.batchID + os.sep + job + ".txt"
        return os.path.isfile(fname)



if __name__ == "__main__":
    os.chdir("..")
    os.chdir("..")
    print(os.getcwd())

