#!/usr/bin/python3

import os

from typing import List #class
from typing import Dict #class

from pandas.core.frame import DataFrame #class

from portfolioDescription.portfolio1AggrDescription import Portfolio1AggrDescription #class

from evaluationTool.aEvalTool import AEvalTool #class
from evaluationTool.evalToolDHondt import EvalToolDHondt #class

from aggregationDescription.aggregationDescription import AggregationDescription #class

from input.inputAggrDefinition import InputAggrDefinition  #class
from input.modelDefinition import ModelDefinition

from input.inputRecomMLDefinition import InputRecomMLDefinition #class

from input.batchesML1m.batchMLFuzzyDHondt import BatchMLFuzzyDHondt #class
from input.batchesML1m.batchMLFuzzyDHondtINF import BatchMLFuzzyDHondtINF #class

from aggregation.negImplFeedback.aPenalization import APenalization #class

from aggregation.operators.aDHondtSelector import ADHondtSelector #class

from input.aBatch import BatchParameters #class
from input.aBatchML import ABatchML #class

from evaluationTool.evalToolDHondtBanditVotes import EvalToolDHondtBanditVotes #class

from input.inputSimulatorDefinition import InputSimulatorDefinition #class

from simulator.simulator import Simulator #class

from history.historyHierDF import HistoryHierDF #class

from aggregation.negImplFeedback.aPenalization import APenalization #class
from aggregation.negImplFeedback.penalUsingFiltering import PenalUsingFiltering #class
from aggregation.negImplFeedback.penalUsingReduceRelevance import PenalUsingReduceRelevance #class
from aggregation.negImplFeedback.penalUsingProbability import PenalUsingProbability #class
from aggregation.negImplFeedback.penalUsingReduceRelevance import penaltyStatic #function
from aggregation.negImplFeedback.penalUsingReduceRelevance import penaltyLinear #function



class BatchMLFuzzyDHondtThompsonSamplingINF(ABatchML):

    @classmethod
    def getParamsForLinerFncs(cls):
        a:List[int,int] = [(1.0, i) for i in [0.75, 0.5, 0.25, 0.0]]
        b:List[int, int] = [(0.75, i) for i in [0.5, 0.25, 0.0]]
        c:List[int, int] = [(0.5, i) for i in [0.25, 0.0]]
        d:List[int, int] = [(0.25, i) for i in [0.0]]

        functions:List[tuple] =  a + b + c + d
        return [[aI,bI] for aI, bI in functions]

    @classmethod
    def getParamsForStaticFncs(cls):
        return [0.75, 0.5, 0.25]

    @classmethod
    def getStaticLinearFncs(cls):
        functionsDict = {}
        for penalClassI in [PenalUsingReduceRelevance, PenalUsingProbability]:
            for staticFncI in cls.getParamsForStaticFncs():
                for linerFncI in cls.getParamsForLinerFncs():
                    rI:str = ""
                    if penalClassI == PenalUsingReduceRelevance:
                        rI = "Reduce"
                    elif penalClassI == PenalUsingProbability:
                        rI = "ReduceProb"
                    else:
                        raise ValueError("That is not a positive!")

                    pFncI = penalClassI(penaltyStatic, [staticFncI], penaltyLinear, [linerFncI[0], linerFncI[1], 100], 100)
                    keyI: str = rI + "OStat" + str(staticFncI).replace(".", "") + "HLin" + str(linerFncI[0]).replace(".", "") + str(linerFncI[1]).replace(".", "")
                    #print(keyI)

                    functionsDict[keyI] = pFncI
                    aI = str(penalClassI.__name__) + "(penaltyStatic, [" + str(staticFncI) + "], penaltyLinear, [" + str(linerFncI[0]) + ", " + str(linerFncI[1]) + ", 100], 100)"
                    #print(aI)
        return functionsDict

    @classmethod
    def getLinearLinerFncs(cls):
        functionsDict:Dict[str,object] = {}
        for penalClassI in [PenalUsingReduceRelevance, PenalUsingProbability]:
            for linerFncI in cls.getParamsForLinerFncs():
                for linerFncJ in cls.getParamsForLinerFncs():
                    rI:str = ""
                    if penalClassI == PenalUsingReduceRelevance:
                        rI = "Reduce"
                    elif penalClassI == PenalUsingProbability:
                        rI = "ReduceProb"
                    else:
                        raise ValueError("That is not a positive!")

                    pFncI = penalClassI(penaltyLinear, [linerFncI[0], linerFncI[1], 20], penaltyLinear, [linerFncJ[0], linerFncJ[1], 100], 100)
                    keyI:str = rI + "OLin" + str(linerFncI[0]).replace(".","") + str(linerFncI[1]).replace(".","") + "HLin" + str(linerFncJ[0]).replace(".","") + str(linerFncJ[1]).replace(".","")
                    #print(keyI)
                    functionsDict[keyI] = pFncI
                    aI = str(penalClassI.__name__) + "(penaltyLinear, [" + str(linerFncI[0]) + ", " + str(linerFncI[1]) + ", 20], penaltyLinear, [" + str(linerFncI[0]) + ", " + str(linerFncI[1]) + ", 100], 100)"
                    #print(aI)
        return functionsDict


    def getPenalFncs(cls):
        d1 = cls.getStaticLinearFncs()
        d2 = cls.getLinearLinerFncs()

        dall:Dict[str.objecr] = {}
        dall.update(d1)
        dall.update(d2)

        return dall

    @staticmethod
    def getParameters():
        selectorIDs:List[str] = BatchMLFuzzyDHondt().getSelectorParameters().keys()
        negativeImplFeedback:List[str] = BatchMLFuzzyDHondtThompsonSamplingINF().getPenalFncs().keys()

        aDict:dict = {}
        for selectorIDH in selectorIDs:
            for nImplFeedbackI in negativeImplFeedback:
                keyIJ:str = str(selectorIDH) + nImplFeedbackI

                nImplFeedback:APenalization = BatchMLFuzzyDHondtThompsonSamplingINF().getPenalFncs()[nImplFeedbackI]
                selectorH:ADHondtSelector = BatchMLFuzzyDHondt().getSelectorParameters()[selectorIDH]

                aDict[keyIJ] = (selectorH, nImplFeedback)
        return aDict


    @staticmethod
    def getParameters2():
        selectorIDs:List[str] = BatchMLFuzzyDHondt().getSelectorParameters().keys()
        negativeImplFeedback:List[str] = BatchMLFuzzyDHondtINF().getNegativeImplFeedbackParameters().keys()

        aDict:dict = {}
        for selectorIDH in selectorIDs:
            for nImplFeedbackI in negativeImplFeedback:
                keyIJ:str = str(selectorIDH) + nImplFeedbackI

                nImplFeedback:APenalization = BatchMLFuzzyDHondtINF().getNegativeImplFeedbackParameters()[nImplFeedbackI]
                selectorH:ADHondtSelector = BatchMLFuzzyDHondt().getSelectorParameters()[selectorIDH]

                aDict[keyIJ] = (selectorH, nImplFeedback)
        return aDict


    def run(self, batchID:str, jobID:str):

        divisionDatasetPercentualSize:int
        uBehaviour:str
        repetition:int
        divisionDatasetPercentualSize, uBehaviour, repetition = BatchParameters.getBatchParameters(self.datasetID)[batchID]

        selector, nImplFeedback = self.getParameters()[jobID]

        eTool:AEvalTool = EvalToolDHondtBanditVotes({})

        rIDs, rDescs = InputRecomMLDefinition.exportPairOfRecomIdsAndRecomDescrs()

        aDescNegDHontThompsonSamplingI:AggregationDescription = InputAggrDefinition.exportADescDHondtThompsonSamplingINF(selector, nImplFeedback)

        pDescr:Portfolio1AggrDescription = Portfolio1AggrDescription(
            "FuzzyDHondtThompsonSamplingINF" + jobID, rIDs, rDescs, aDescNegDHontThompsonSamplingI)

        model:DataFrame = ModelDefinition.createDHondtBanditsVotesModel(pDescr.getRecommendersIDs())

        simulator:Simulator = InputSimulatorDefinition.exportSimulatorML1M(
                batchID, divisionDatasetPercentualSize, uBehaviour, repetition)
        simulator.simulate([pDescr], [model], [eTool], [HistoryHierDF(pDescr.getPortfolioID())])




if __name__ == "__main__":
    os.chdir("..")
    os.chdir("..")

    BatchMLFuzzyDHondtThompsonSamplingINF.generateBatches()