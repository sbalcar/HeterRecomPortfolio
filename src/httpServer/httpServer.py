#!/usr/bin/python3

from typing import Dict
from pandas import DataFrame

from portfolio.aPortfolio import APortfolio #class

from http.server import BaseHTTPRequestHandler, HTTPServer

from urllib.parse import parse_qs, urlparse

from datasets.slantour.events import Events #class

class HeterRecomHTTPHandler(BaseHTTPRequestHandler):

    VARIANT_A:str = "a"
    VARIANT_B:str = "b"

    #ACTION_INIT:str = "init"
    ACTION_UPDATE:str = "update"
    ACTION_RECOMMEND:str = "recommend"

    ARG_VARIANTID:str = "variantID"
    ARG_ACTIONID:str = "actionID"
    ARG_USERID:str = "userID"
    ARG_ITEMID:str = "itemID"
    ARG_NUMBER_OF_ITEMS:str = "numberOfItems"

    #portfolioDict:Dict[]
    #modelsDict:Dict[]

    def do_GET(self):
        query = urlparse(self.path).query
        #print(query)
        params: Dict[str, str] = dict(qc.split("=") for qc in query.split("&"))
        #print(params)

        if not self.ARG_ACTIONID in params:
            self.send_error(404)
            return

        variant:str = params[self.ARG_VARIANTID]
        print("variant: " + variant)

        action:str = params[self.ARG_ACTIONID]
        print("action: " + action)

        userID:int = int(params[self.ARG_USERID])
        print("userID: " + str(userID))

        if action == self.ACTION_UPDATE:
            itemID:int = int(params[self.ARG_ITEMID])
            print("itemID: " + str(itemID))
            return self.__update(variant, userID, itemID)

        elif action == self.ACTION_RECOMMEND:
            numberOfItems:int = int(params[self.ARG_NUMBER_OF_ITEMS])
            print("numberOfItems: " + str(numberOfItems))
            return self.__recommend(variant, userID, numberOfItems)

        else:
            self.send_error(404)


    def __update(self, variant:str, userID:int, itemID:int):
        print("update")

        portfolio:APortfolio = self.portfolioDict[variant]

        df:DataFrame = DataFrame([[userID, itemID]], columns=[Events.COL_USER_ID, Events.COL_OBJECT_ID])
        portfolio.update(df)

        self.send_response(200)
        self.send_header('Content-Type', 'text/plain; charset=utf-8')
        self.end_headers()
        message:str = "updated: userID=" + str(userID) + ", itemID=" + str(itemID)
        self.wfile.write(message.encode('utf-8'))
        self.wfile.write(b'\n')


    def __recommend(self, variant:str, userID:int, numberOfItems:int):
        print("recommend")
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
