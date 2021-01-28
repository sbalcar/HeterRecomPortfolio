#!/usr/bin/python3

import os
import _thread
import time
from httpServer.httpServer import HeterRecomHTTPHandler #class

from http.server import BaseHTTPRequestHandler, HTTPServer

from typing import Dict
from typing import List
from pandas import DataFrame
from pandas.core.series import Series #class

from history.aHistory import AHistory #class
from history.historyDF import HistoryDF #class

from datasets.aDataset import ADataset #class
from datasets.datasetML import DatasetML #class
from datasets.datasetST import DatasetST #class
from datasets.slantour.events import Events #class

from recommender.aRecommender import ARecommender #class
from recommender.recommenderTheMostPopular import RecommenderTheMostPopular #class
from recommender.recommenderCosineCB import RecommenderCosineCB #class
from recommender.recommenderW2V import RecommenderW2V #class
from recommender.recommenderItemBasedKNN import RecommenderItemBasedKNN #class
from recommender.recommenderBPRMF import RecommenderBPRMF #class

from portfolio.aPortfolio import APortfolio #class

from portfolioDescription.portfolio1MethDescription import Portfolio1MethDescription #class

from recommenderDescription.recommenderDescription import RecommenderDescription #class

from evaluationTool.aEvalTool import AEvalTool #class
from evaluationTool.evalToolDoNothing import EToolDoNothing #class

from execute.startHttpServer import getTheMostPopular #function
from execute.startHttpServer import getFuzzyDHont #function



def startS():
    print("Starting Http server")
    server = HTTPServer(('', 8080), HeterRecomHTTPHandler)
    server.serve_forever()


def test01():
    print("Test 01")

    #port1, model1, evalTool1 = getTheMostPopular()
    port1, model1, evalTool1 = getFuzzyDHont()

    portfolioDict:Dict[str, APortfolio] = {HeterRecomHTTPHandler.VARIANT_1: port1}
    modelsDict:Dict[str, int] = {HeterRecomHTTPHandler.VARIANT_1: model1}
    evalToolsDict:Dict[str, AEvalTool] = {HeterRecomHTTPHandler.VARIANT_1: evalTool1}
    evaluationDict:Dict[str,object] = {}

    HeterRecomHTTPHandler.portfolioDict = portfolioDict
    HeterRecomHTTPHandler.modelsDict = modelsDict
    HeterRecomHTTPHandler.evalToolsDict = evalToolsDict
    HeterRecomHTTPHandler.evaluation = evaluationDict
    HeterRecomHTTPHandler.datasetClass = DatasetST

    # Run HTTP in separate thread
    try:
        _thread.start_new_thread(startS, ())
    except:
        print("Error: unable to start HTTP Server thread")

    time.sleep(3)

    itemID:int = 555
    userID:int = 1

    rItemIDsWithResponsibility:List = [(7, {'metoda1': 0, 'metoda2': 24.0, 'metoda3': 18.0}), (1, {'metoda1': 30.0, 'metoda2': 8.0, 'metoda3': 0}), (32, {'metoda1': 20.0, 'metoda2': 16.0, 'metoda3': 0}), (8, {'metoda1': 30.0, 'metoda2': 0, 'metoda3': 0}), (6, {'metoda1': 0, 'metoda2': 24.0, 'metoda3': 0}), (64, {'metoda1': 0, 'metoda2': 0, 'metoda3': 18.0}), (2, {'metoda1': 10.0, 'metoda2': 0, 'metoda3': 6.0}), (77, {'metoda1': 0, 'metoda2': 0, 'metoda3': 12.0}), (4, {'metoda1': 10.0, 'metoda2': 0, 'metoda3': 0}), (5, {'metoda1': 0, 'metoda2': 8.0, 'metoda3': 0}), (12, {'metoda1': 0, 'metoda2': 0, 'metoda3': 6.0})]
    #rItemIDsWithResponsibility = "\[1,2,3,\{\}\]"
    rItemIDsWithResponsibility = "\[(555,\{\"RecomKnn\":1.0\})\]"


    command0:str = "curl -sS 'http://127.0.0.1:8080/?" + \
                   HeterRecomHTTPHandler.ARG_ACTIONID + "=" + HeterRecomHTTPHandler.ACTION_VISIT + "&" + \
                   HeterRecomHTTPHandler.ARG_VARIANTID + "=" + HeterRecomHTTPHandler.VARIANT_1 + "&" + \
                   HeterRecomHTTPHandler.ARG_USERID + "=" + str(userID) + "&" + \
                   HeterRecomHTTPHandler.ARG_ITEMID + "=" + str(itemID) + "&" + \
                   HeterRecomHTTPHandler.ARG_NUMBER_OF_ITEMS + "=20" + "&" + \
                   HeterRecomHTTPHandler.ARG_ALLOWED_ITEMIDS + "=None" + "'"

    print("IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII")
    print(command0)
    print("IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII")
    cmdResult:str = os.popen(command0).read()
    print("OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
    print(cmdResult)
    print("OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")


    command1:str = "curl -sS 'http://127.0.0.1:8080/?" + \
                   HeterRecomHTTPHandler.ARG_ACTIONID + "=" + HeterRecomHTTPHandler.ACTION_CLICK + "&" + \
                   HeterRecomHTTPHandler.ARG_VARIANTID + "=" + HeterRecomHTTPHandler.VARIANT_1 + "&" + \
                   HeterRecomHTTPHandler.ARG_UPDU_TYPE + "=" + HeterRecomHTTPHandler.UPDT_CLICK + "&" + \
                   HeterRecomHTTPHandler.ARG_ITEMID + "=" + str(itemID) + "&" + \
                   HeterRecomHTTPHandler.ARG_USERID + "=" + str(userID) + "&" + \
                   HeterRecomHTTPHandler.ARG_RITEMIDS_WITH_RESP + "=" + str(rItemIDsWithResponsibility) + "'"

    print("IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII")
    print(command1)
    print("IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII")
    cmdResult:str = os.popen(command1).read()
    print("OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
    print(cmdResult)
    print("OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")

    print("evaluationDict: " + str(evaluationDict))


    time.sleep(3)


if __name__ == '__main__':
    os.chdir("..")

    test01()