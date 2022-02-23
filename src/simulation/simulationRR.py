#!/usr/bin/python3
import os
from configuration.configuration import Configuration #class

from typing import List
from typing import Dict

from datasets.aDataset import ADataset #class
from datasets.datasetRetailRocket import DatasetRetailRocket #class

from datasets.retailrocket.events import Events #class

from pandas.core.frame import DataFrame #class

from recommender.aRecommender import ARecommender #class

from portfolioDescription.aPortfolioDescription import APortfolioDescription #class

from simulation.aSequentialSimulation import ASequentialSimulation #class
from simulation.simulationML import SimulationML #class

from history.aHistory import AHistory #class
from evaluationTool.aEvalTool import AEvalTool #class

from datasets.retailrocket.behavioursRR import BehavioursRR #class

from portfolio.aPortfolio import APortfolio #class

from simulation.tools.modelOfIndexes import ModelOfIndexes #class


class SimulationRR(ASequentialSimulation):

    _ratingClass = Events
    _behaviourClass = BehavioursRR


    @staticmethod
    def divideDataset(dataset:ADataset, behaviourDF:DataFrame,
                      divisionDatasetPercentualSize:int, testDatasetPercentualSize:int,
                      recomRepetitionCount:int):

        eventsDF:DataFrame = dataset.eventsDF
        categoryTreeDF:DataFrame = dataset.categoryTreeDF
        itemPropertiesDF:DataFrame = dataset.itemPropertiesDF

        # create train Dataset
        eventsSortedDF:DataFrame = eventsDF.sort_values(by=Events.COL_TIME_STAMP)
        numberOfEvents:int = eventsSortedDF.shape[0]

        trainSize:int = (int)(numberOfEvents * divisionDatasetPercentualSize / 100)
        #print("trainSize: " + str(trainSize))
        trainRatingsDF:DataFrame = eventsSortedDF[0:trainSize]

        datasetID:str = "rr" + "Div" + str(divisionDatasetPercentualSize)
        trainDataset:ADataset = DatasetRetailRocket(datasetID, trainRatingsDF, categoryTreeDF, itemPropertiesDF)


        # create test Event DataFrame
        testSize:int = (int)(numberOfEvents * testDatasetPercentualSize / 100)
        #print("testSize: " + str(testSize))
        testEventsPartDF:DataFrame = eventsSortedDF[trainSize:(trainSize + testSize)]
        testEventsDF:DataFrame = testEventsPartDF.loc[testEventsPartDF[Events.COL_EVENT] == "view"]


        # create test relevant Event DataFrame
        testRelevantEventsDF:DataFrame = testEventsDF


        # create behaviour dictionary of DataFrame indexed by recomRepetition
        recomRepetitionCountInDataset:int = behaviourDF[BehavioursRR.COL_REPETITION].max() +1
        #print("count: " + str(recomRepetitionCountInDataset))
        #print(behaviourDF.head(10))
        #print(behaviourDF.tail(10))
        #print("events:      " + str(len(eventsDF)))
        #print("behaviours: " + str(len(behaviourDF)))
        #print("max behaviours: " + str(max(behaviourDF.index)))
        #print("recomRepetitionCount: " + str(recomRepetitionCount))
        #print("range: " + str(range(recomRepetitionCount)))

        testRepeatedBehaviourDict:dict = {}
        bIndexes:List[int] = list([recomRepetitionCountInDataset*i for i in testEventsDF.index])
        for repetitionI in range(recomRepetitionCount):
            # indexes of behaviour
            indexes:List[int] = [vI+repetitionI for vI in bIndexes]

            # indexes
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
            clickedItems[rowI[Events.COL_VISITOR_ID]] = []
        return clickedItems


    def iterateOverDataset(self, portfolios:List[APortfolio], portfolioDescs:List[APortfolioDescription],
                             portFolioModels:List[DataFrame], evaluatonTools:List[AEvalTool],
                             histories:List[AHistory], testRatingsDF:DataFrame, testRelevantRatingsDF:DataFrame,
                             testBehaviourDict:Dict[int, DataFrame]):

        model:ModelOfIndexes = ModelOfIndexes(testRatingsDF, list(testRelevantRatingsDF.index), Events)

        portIds:List[str] = [portDescI.getPortfolioID() for portDescI in portfolioDescs]

        evaluations:List[dict] = [{} for i in range(len(portfolios))]

        # opening stat file
        dir:str = Configuration.resultsDirectory + os.sep + self._batchID
        fileNameStat:str = dir + os.sep + "status-" + portfolioDescs[0].getPortfolioID() + ".txt"
        if os.path.exists(fileNameStat):
            os.remove(fileNameStat)
        fileStat = open(fileNameStat, "a")

        counterI:int = 0

        currentDFIndexI:int
        nextIndexDFI:int
        for currentDFIndexI in list(testRatingsDF.index):

            counterI += 1
            if counterI  % 100 == 0:
                print("EventI: " + str(counterI) + " / " + str(testRatingsDF.shape[0]))
                fileStat.write("status: " + str(round(counterI / testRatingsDF.shape[0] * 100,2)) + " procent\n")
                fileStat.flush()

                print("PortfolioIds: " + str(portIds))
                print("Evaluations: " + str(evaluations))

                self.computationFile.write("RatingI: " + str(counterI) + " / " + str(testRatingsDF.shape[0]) + "\n")
                self.computationFile.write("PortfolioIds: " + str(portIds) + "\n")
                self.computationFile.write("Evaluations: " + str(evaluations) + "\n")
                self.computationFile.flush()

            currentItemIdI:int = testRatingsDF.loc[currentDFIndexI][Events.COL_ITEM_ID]
            currentRatingI:int = testRatingsDF.loc[currentDFIndexI][Events.COL_EVENT]
            currentUserIdI:int = testRatingsDF.loc[currentDFIndexI][Events.COL_VISITOR_ID]
            currentSessionIdI:int = None
            currentPageTypeI:object = None

            windowOfItemIDsI:List[int] = model.getNextRelevantItemIDsExceptItemIDs(currentDFIndexI,
                                                                             self._clickedItems[currentUserIdI], self._windowSize)
            windowOfItemIDsI:List[int] = [int(itemIdI) for itemIdI in windowOfItemIDsI]

            portfolioI:APortfolio
            for portfolioI in portfolios:

                dfI:DataFrame = DataFrame([testRatingsDF.loc[currentDFIndexI]], columns=testRatingsDF.keys())
                portfolioI.update(dfI, {})

            repetitionI:int
            for repetitionI in range(self._recomRepetitionCount):
                self.simulateRecommendations(portfolios, portfolioDescs, portFolioModels, evaluatonTools,
                                             histories, evaluations, currentDFIndexI, counterI, testRatingsDF.shape[0],
                                             currentUserIdI, currentSessionIdI, repetitionI,
                                             testRatingsDF, testBehaviourDict, windowOfItemIDsI, currentPageTypeI)

        return evaluations

