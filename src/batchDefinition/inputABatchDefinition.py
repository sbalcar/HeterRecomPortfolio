from abc import abstractmethod

from typing import List #class
from typing import Dict #class

from datasets.ml.behavioursML import BehavioursML


class InputABatchDefinition:

    divisionsDatasetPercentualSize:List[int] = [80, 90]
    uBehaviours:List[str] = [BehavioursML.BHVR_LINEAR0109,
                              BehavioursML.BHVR_STATIC08,
                              # BehavioursML.BHVR_STATIC06,
                              # BehavioursML.BHVR_STATIC04,
                              # BehavioursML.BHVR_STATIC02,
                              BehavioursML.BHVR_POWERLAW054MIN048]
    repetitions:List[int] = [1, 2, 3]

    @abstractmethod
    def getBatchParameters(self, datasetID:str):

        aDict:Dict[str,object] = {}

        for divisionDatasetPercentualSizeI in self.divisionsDatasetPercentualSize:
            for uBehaviourJ in self.uBehaviours:
                for repetitionK in self.repetitions:
                    batchID:str = datasetID + "Div" + str(divisionDatasetPercentualSizeI) + "U" + uBehaviourJ + "R" + str(repetitionK)

                    aDict[batchID] = (divisionDatasetPercentualSizeI, uBehaviourJ, repetitionK)

        return aDict