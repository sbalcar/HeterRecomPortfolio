#!/usr/bin/python3

from typing import List
from typing import Dict

from datasets.aDataset import ADataset #class
from datasets.datasetST import DatasetST #class

from datasets.slantour.events import Events #class
from datasets.slantour.serials import Serials #class

from pandas.core.frame import DataFrame #class

from portfolioDescription.aPortfolioDescription import APortfolioDescription #class

from simulation.aSequentialSimulation import ASequentialSimulation #class

from history.aHistory import AHistory #class
from evaluationTool.aEvalTool import AEvalTool #class

from datasets.slantour.behavioursST import BehavioursST #class

from recommender.aRecommender import ARecommender #class

from portfolio.aPortfolio import APortfolio #class

from simulation.tools.modelOfIndexes import ModelOfIndexes #class


class SimulationST(ASequentialSimulation):

    _ratingClass = Events
    _behaviourClass = BehavioursST


    @staticmethod
    def divideDataset(dataset:ADataset, behaviourDF:DataFrame,
                      divisionDatasetPercentualSize:int, testDatasetPercentualSize:int,
                      recomRepetitionCount:int):

        eventsDF:DataFrame = dataset.eventsDF
        serialsDF:DataFrame = dataset.serialsDF


        # create train Dataset
        eventsSortedDF:DataFrame = eventsDF #.sort_values(by=Events.COL_TIME_STAMP)
        numberOfEvents:int = eventsSortedDF.shape[0]

        trainSize:int = (int)(numberOfEvents * divisionDatasetPercentualSize / 100)
        #print("trainSize: " + str(trainSize))
        trainEventsDF:DataFrame = eventsSortedDF[0:trainSize]

        datasetID:str = "st" + "Div" + str(divisionDatasetPercentualSize)
        trainDataset:ADataset = DatasetST(datasetID, trainEventsDF, serialsDF)


        # create test Event DataFrame
        testSize:int = (int)(numberOfEvents * testDatasetPercentualSize / 100)
        #print("testSize: " + str(testSize))
        testEventsPartDF:DataFrame = eventsSortedDF[trainSize:(trainSize + testSize)]
        testEventsDF:DataFrame = testEventsPartDF.loc[testEventsPartDF[Events.COL_OBJECT_ID] != 0]


        # create test relevant Event DataFrame
        testRelevantEventsDF:DataFrame = testEventsDF


        # create behaviour dictionary of DataFrame indexed by recomRepetition
        recomRepetitionCountInDataset:int = behaviourDF[BehavioursST.COL_REPETITION].max() +1

        testRepeatedBehaviourDict:dict = {}
        bIndexes:List[int] = list([recomRepetitionCountInDataset*i for i in testEventsDF.index])
        for repetitionI in range(recomRepetitionCount):
            # indexes of behaviour
            indexes:List[int] = [vI+repetitionI for vI in bIndexes]
            behaviourDFI:DataFrame = DataFrame(behaviourDF.take(indexes).values.tolist(),
                        index=testEventsDF.index, columns=behaviourDF.keys())
            testRepeatedBehaviourDict[repetitionI] = behaviourDFI


        return (trainDataset, testEventsDF, testRelevantEventsDF, testRepeatedBehaviourDict)



    @staticmethod
    def initEvaluationModel(dataset:ADataset):
        eventsDF:DataFrame = dataset.eventsDF

        # model with clicked ItemIDs for users
        clickedItems:dict[List[int]] = {}
        for indexI, rowI in eventsDF.iterrows():
            clickedItems[rowI[Events.COL_USER_ID]] = []
        return clickedItems


    def iterateOverDataset(self, portfolios:List[APortfolio], portfolioDescs:List[APortfolioDescription],
                             portFolioModels:List[DataFrame], evaluatonTools:List[AEvalTool],
                             histories:List[AHistory], testRatingsDF:DataFrame, testRelevantRatingsDF:DataFrame,
                             testBehaviourDict:Dict[int, DataFrame]):

        model:ModelOfIndexes = ModelOfIndexes(testRatingsDF, list(testRelevantRatingsDF.index), Events)

        portIds:List[str] = [portDescI.getPortfolioID() for portDescI in portfolioDescs]

        evaluations:List[dict] = [{} for i in range(len(portfolios))]

        counterI:int = 0

        currentDFIndexI:int
        nextIndexDFI:int
        for currentDFIndexI in list(testRatingsDF.index):

            counterI += 1
            if counterI  % 100 == 0:
                print("EventI: " + str(counterI) + " / " + str(testRatingsDF.shape[0]))
                print("PortfolioIds: " + str(portIds))
                print("Evaluations: " + str(evaluations))

                self.computationFile.write("RatingI: " + str(counterI) + " / " + str(testRatingsDF.shape[0]) + "\n")
                self.computationFile.write("PortfolioIds: " + str(portIds) + "\n")
                self.computationFile.write("Evaluations: " + str(evaluations) + "\n")
                self.computationFile.flush()

            currentItemIdI:int = testRatingsDF.loc[currentDFIndexI][Events.COL_OBJECT_ID]
            currentUserIdI:int = testRatingsDF.loc[currentDFIndexI][Events.COL_USER_ID]


            windowOfItemIDsI:int = model.getNextRelevantItemIDsExceptItemIDs(currentDFIndexI,
                                                                             self._clickedItems[currentUserIdI], self._windowSize)
            portfolioI:APortfolio
            for portfolioI in portfolios:

                dfI:DataFrame = DataFrame([testRatingsDF.loc[currentDFIndexI]], columns=testRatingsDF.keys())
                portfolioI.update(ARecommender.UPDT_VIEW, dfI)

            repetitionI:int
            for repetitionI in range(self._recomRepetitionCount):
                self.simulateRecommendations(portfolios, portfolioDescs, portFolioModels, evaluatonTools,
                                             histories, evaluations, currentDFIndexI, currentUserIdI, repetitionI,
                                             testRatingsDF, testBehaviourDict, windowOfItemIDsI)

        return evaluations

