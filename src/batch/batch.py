#!/usr/bin/python3

import os
import os.path
from typing import List #class
from typing import Dict #class

from configuration.configuration import Configuration #class


class Batch:

    def __init__(self, batchDefinitionClass, batchID:str, jobID:str):
        self.batchDefinitionClass = batchDefinitionClass
        self.batchID:str = batchID
        self.jobID = jobID

    def run(self):
        self.batchDefinitionClass.run(self.batchID, self.jobID)


    def exists(self):
        #print(str(self.batchDefinitionClass))
        from execute.generateBatches import getBatchInstance  # class
        aa = getBatchInstance(self.batchDefinitionClass)
        job:str = str(aa.getBatchName()) + self.jobID

        fname:str = Configuration.resultsDirectory + os.sep + self.batchID + os.sep + "computation-" + job + ".txt"
        #print(fname)

        return os.path.isfile(fname)



if __name__ == "__main__":
    os.chdir("..")
    os.chdir("..")
    print(os.getcwd())

