from typing import List

import pandas as pd
from pandas import DataFrame

from evaluationTool.evalToolDHondt import EvalToolDHondt


class ModelDefinition:

    def createDHontModel(recommendersIDs: List[str]):
        modelDHontData:List[List] = [[rIdI, 1] for rIdI in recommendersIDs]
        modelDHontDF:DataFrame = pd.DataFrame(modelDHontData, columns=["methodID", "votes"])
        modelDHontDF.set_index("methodID", inplace=True)
        EvalToolDHondt.linearNormalizingPortfolioModelDHont(modelDHontDF)
        return modelDHontDF

    def createDHondtBanditsVotesModel(recommendersIDs: List[str]):
        modelDHondtBanditsVotesData:List = [[rIdI, 1.0, 1.0, 1.0, 1.0] for rIdI in recommendersIDs]
        modelDHondtBanditsVotesDF:DataFrame = pd.DataFrame(modelDHondtBanditsVotesData, columns=["methodID", "r", "n", "alpha0", "beta0"])
        modelDHondtBanditsVotesDF.set_index("methodID", inplace=True)
        #EvalToolDHont.linearNormalizingPortfolioModelDHont(modelDHontDF)
        return modelDHondtBanditsVotesDF

    def createBanditModel(recommendersIDs:List[str]):
        modelBanditTSData:List = [[rIdI, 1, 1, 1, 1] for rIdI in recommendersIDs]
        modelBanditTSDF:DataFrame = pd.DataFrame(modelBanditTSData, columns=["methodID", "r", "n", "alpha0", "beta0"])
        modelBanditTSDF.set_index("methodID", inplace=True)
        return modelBanditTSDF