#!/usr/bin/python3

import os
from typing import List

import pandas as pd
from pandas import DataFrame

from evaluationTool.evalToolDHondt import EvalToolDHondt


class PModelDHondtBanditsVotes(pd.DataFrame):

    def __init__(self, recommendersIDs:List[str]):
        modelDHondtBanditsVotesData:List = [[rIdI, 1.0, 1.0, 1.0, 1.0] for rIdI in recommendersIDs]

        super(PModelDHondtBanditsVotes, self).__init__(modelDHondtBanditsVotesData, columns=["methodID", "r", "n", "alpha0", "beta0"])
        self.set_index("methodID", inplace=True)
        #self.linearNormalizingPortfolioModelDHont(modelDHontDF)




if __name__ == "__main__":
    os.chdir("..")
    m = PModelDHondtBanditsVotes(["r1", "r2"])
    print(m.head(10))