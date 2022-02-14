#!/usr/bin/python3

import os
from typing import List

import pandas as pd
from pandas import DataFrame

from evaluationTool.evalToolDHondt import EvalToolDHondt
from numpy.random import beta


class PModelBandit(pd.DataFrame):

    COL_METHOD_ID:str = "methodID"
    COL_R:str = "r"
    COL_N:str = "n"
    COL_ALPHA:str = "alpha0"
    COL_BETA:str = "beta0"

    def __init__(self, recommendersIDs:List[str]):
        #if type(recommendersIDs) is pd.DataFrame:
        #    super(PModelBandit, self).__init__(recommendersIDs, columns=PModelBandit.getColumns())
        #    return

        if type(recommendersIDs) is not list:
            raise ValueError("Argument recommendersIDs isn't type list.")

        modelBanditTSData:List = [[rIdI, 1, 1, 1, 1] for rIdI in recommendersIDs]

        super(PModelBandit, self).__init__(modelBanditTSData, columns=PModelBandit.getColumns())
        self.set_index(PModelBandit.COL_METHOD_ID, inplace=True)

    def get(self):
#        for mIndex in methodsParamsDFI.index:
#            # print("mIndexI: ", mIndex)
#            methodI = methodsParamsDFI.loc[methodsParamsDFI.index == mIndex]  # .iloc[0]
#            # alpha + number of successes, beta + number of failures
#            pI = beta(methodI.alpha0 + methodI.r, methodI.beta0 + (methodI.n - methodI.r), size=1)[0]
#            methodProbabilitiesDicI[mIndex] = pI
        return 0



    @staticmethod
    def getColumns():
        columns = [
            PModelBandit.COL_METHOD_ID,
            PModelBandit.COL_R,
            PModelBandit.COL_N,
            PModelBandit.COL_ALPHA,
            PModelBandit.COL_BETA]
        return columns


if __name__ == "__main__":
    os.chdir("..")

    a1 = DataFrame(columns=['a', 'b', 'c'])
    a1.loc[0] = ['name0', 0, 0]

    a2 = a1.copy()
    a2.loc[1] = ['name1', 1, 1]

    print(a1.head())
    print()
    print(a2.head())
    print()

    for mI in a1:
        print(mI)