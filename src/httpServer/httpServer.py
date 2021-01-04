#!/usr/bin/python3

from typing import Dict
from pandas import DataFrame

from portfolio.aPortfolio import APortfolio #class

from http.server import BaseHTTPRequestHandler, HTTPServer

from urllib.parse import parse_qs, urlparse

from datasets.slantour.events import Events #class

from evaluationTool.aEvalTool import AEvalTool #class
from evaluationTool.evalToolSingleMethod import EToolSingleMethod #class

import pandas as pd

class HeterRecomHTTPHandler(BaseHTTPRequestHandler):

    #ACTION_INIT:str = "init"
    ACTION_UPDATE:str = "update"
    ACTION_RECOMMEND:str = "recommend"

    ARG_VARIANTID:str = "variantID"
    ARG_ACTIONID:str = "actionID"
    ARG_UPDU_TYPE:str = "updtType"
    ARG_USERID:str = "userID"
    ARG_ITEMID:str = "itemID"
    ARG_NUMBER_OF_ITEMS:str = "numberOfItems"
    ARG_RITEMIDS_WITH_RESP:str = "rItemIDsWithResponsibility"

    VARIANT_A:str = "a"
    VARIANT_B:str = "b"

    UPDT_CLICK:str = "click"
    UPDT_VIEW:str = "view"

    #portfolioDict:Dict[]
    #modelsDict:Dict[]
    #evalToolsDict:Dict[]

    def do_GET(self):
        query = urlparse(self.path).query
        #print(query)
        params: Dict[str, str] = dict(qc.split("=") for qc in query.split("&"))
        #print(params)

        if not self.ARG_VARIANTID in params:
            self.send_error(404)
            return
        if not self.ARG_ACTIONID in params:
            self.send_error(404)
            return
        if not self.ARG_USERID in params:
            self.send_error(404)
            return

        variant:str = params[self.ARG_VARIANTID]
        print("variant: " + variant)

        action:str = params[self.ARG_ACTIONID]
        print("action: " + action)

        userID:int = int(params[self.ARG_USERID])
        print("userID: " + str(userID))

        if action == self.ACTION_UPDATE:
            if not self.ARG_ITEMID in params:
                self.send_error(404)
                return
            if not self.ARG_UPDU_TYPE in params:
                self.send_error(404)
                return
            if not self.ARG_RITEMIDS_WITH_RESP in params:
                self.send_error(404)
                return
            itemID:int = int(params[self.ARG_ITEMID])
            print("itemID: " + str(itemID))
            updtType:str = str(params[self.ARG_UPDU_TYPE])
            print("updtType: " + str(updtType))
            a:str = params[self.ARG_RITEMIDS_WITH_RESP]
            print(a)
            print("a: " + str(a))
            if not updtType in [self.UPDT_CLICK, self.UPDT_VIEW]:
                print("Error: incorect updtType :" + str(updtType))
                self.send_error(404)
                return
            self.__update(variant, updtType, userID, itemID)
            return

        elif action == self.ACTION_RECOMMEND:
            if not self.ARG_NUMBER_OF_ITEMS in params:
                self.send_error(404)
                return
            numberOfItems:int = int(params[self.ARG_NUMBER_OF_ITEMS])
            print("numberOfItems: " + str(numberOfItems))
            self.__recommend(variant, userID, numberOfItems)
            return

        else:
            self.send_error(404)


    def __update(self, variant:str, updtType:str, userID:int, itemID:int):
        print("update")
        if not variant in self.portfolioDict:
            self.send_error(404)
            return
        if not updtType in [self.UPDT_CLICK, self.UPDT_VIEW]:
            self.send_error(404)
            return

        portfolio:APortfolio = self.portfolioDict[variant]

        df:DataFrame = DataFrame([[userID, itemID]], columns=[Events.COL_USER_ID, Events.COL_OBJECT_ID])
        portfolio.update(updtType, df)

        evaluation:Dict = {}
        rItemIDsWithResponsibility = pd.Series()
        evalTool:AEvalTool = self.evalToolsDict[variant]

        if updtType == self.UPDT_CLICK:
            evalTool.click(rItemIDsWithResponsibility, itemID, self.modelsDict[variant], evaluation)
        elif updtType == self.UPDT_VIEW:
            evalTool.displayed(rItemIDsWithResponsibility, self.modelsDict[variant], evaluation)

        self.send_response(200)
        self.send_header('Content-Type', 'text/plain; charset=utf-8')
        self.end_headers()
        message:str = "updated: userID=" + str(userID) + ", itemID=" + str(itemID)
        self.wfile.write(message.encode('utf-8'))
        self.wfile.write(b'\n')


    def __recommend(self, variant:str, userID:int, numberOfItems:int):
        print("recommend")
        if not variant in self.portfolioDict:
            self.send_error(404)
            return

        portfolio:APortfolio = self.portfolioDict[variant]
        model:DataFrame = self.modelsDict[variant]

        rec, resp = portfolio.recommend(userID, model, {APortfolio.ARG_NUMBER_OF_AGGR_ITEMS:numberOfItems})
        #print(rec)
        #print(resp)

        self.send_response(200)
        self.send_header('Content-Type', 'text/plain; charset=utf-8')
        self.end_headers()
        message:str = "recommendation: userID=" + str(userID) + ", r=" + str(rec)
        self.wfile.write(message.encode('utf-8'))
        self.wfile.write(b'\n')


if __name__ == '__main__':

    server = HTTPServer(('', 8080), HeterRecomHTTPHandler)
    server.serve_forever()
