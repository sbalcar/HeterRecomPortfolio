#!/usr/bin/python3
import ast

from typing import Dict
from typing import List
from pandas import DataFrame
from pandas.core.series import Series #class

from portfolio.aPortfolio import APortfolio #class

from http.server import BaseHTTPRequestHandler, HTTPServer

from urllib.parse import parse_qs, urlparse

from datasets.aDataset import ADataset #class
from datasets.datasetML import DatasetML #class
from datasets.datasetRetailrocket import DatasetRetailRocket #class
from datasets.datasetST import DatasetST #class

from datasets.slantour.events import Events #class
from datasets.ml.ratings import Ratings #class

from evaluationTool.aEvalTool import AEvalTool #class
from evaluationTool.evalToolSingleMethod import EToolSingleMethod #class

from recommender.aRecommender import ARecommender #class

import pandas as pd

class HeterRecomHTTPHandler(BaseHTTPRequestHandler):

    #ACTION_INIT:str = "init"
    ACTION_CLICK:str = "click"
    ACTION_VISIT:str = "visit"

    ARG_VARIANTID:str = "variantID"
    ARG_ACTIONID:str = "actionID"
    ARG_UPDU_TYPE:str = "updtType"
    ARG_USERID:str = "userID"
    ARG_ITEMID:str = "itemID"
    ARG_NUMBER_OF_ITEMS:str = "numberOfItems"
    ARG_RITEMIDS_WITH_RESP:str = "rItemIDsWithResponsibility"
    ARG_ALLOWED_ITEMIDS:str = ARecommender.ARG_ALLOWED_ITEMIDS

    VARIANT_A:str = "a"
    VARIANT_B:str = "b"

    UPDT_CLICK:str = "click"
    UPDT_VIEW:str = "view"

    #portfolioDict:Dict[]
    #modelsDict:Dict[]
    #evalToolsDict:Dict[]
    #evaluation
    #datasetClass

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
        if not self.ARG_ITEMID in params:
            self.send_error(404)
            return

        variant:str = params[self.ARG_VARIANTID]
        print("variant: " + variant)

        action:str = params[self.ARG_ACTIONID]
        print("action: " + action)

        userID:int = int(params[self.ARG_USERID])
        print("userID: " + str(userID))

        itemID: int = int(params[self.ARG_ITEMID])
        print("itemID: " + str(itemID))

        if action == self.ACTION_CLICK:
            if not self.ARG_RITEMIDS_WITH_RESP in params:
                self.send_error(404)
                return
            rItemIdsWithRespStr:str = params[self.ARG_RITEMIDS_WITH_RESP]
            print("rItemIdsWithResp: " + str(rItemIdsWithRespStr))
            rItemIdsWithResp = ast.literal_eval(rItemIdsWithRespStr)

            self.__click(variant, userID, itemID, rItemIdsWithResp)
            return

        elif action == self.ACTION_VISIT:
            if not self.ARG_NUMBER_OF_ITEMS in params:
                self.send_error(404)
                return
            numberOfItems:int = int(params[self.ARG_NUMBER_OF_ITEMS])
            allowedItemIDsStr:str = params[self.ARG_ALLOWED_ITEMIDS]
            allowedItemIDs:List[int] = ast.literal_eval(allowedItemIDsStr)

            print("numberOfItems: " + str(numberOfItems))
            self.__visit(variant, userID, itemID, numberOfItems, allowedItemIDs)
            return

        else:
            self.send_error(404)


    def __click(self, variant:str, userID:int, itemID:int, rItemIDsWithResponsibility:List[tuple]):
        print("click")
        if not variant in self.portfolioDict:
            self.send_error(404)
            return

        evalTool:AEvalTool = self.evalToolsDict[variant]
        evalTool.click(rItemIDsWithResponsibility, itemID, self.modelsDict[variant], self.evaluation)

        #print("evaluation: ", str(self.evaluation))

        self.send_response(200)
        self.send_header('Content-Type', 'text/plain; charset=utf-8')
        self.end_headers()
        message:str = "click: userID=" + str(userID) + ", itemID=" + str(itemID)
        self.wfile.write(message.encode('utf-8'))
        self.wfile.write(b'\n')


    def __visit(self, variant:str, userID:int, itemID:int, numberOfItems:int, allowedItemIDs:List[int]):
        print("visit")
        if not variant in self.portfolioDict:
            self.send_error(404)
            return

        portfolio:APortfolio = self.portfolioDict[variant]

        if self.datasetClass is DatasetML:
            COL_USERID:str = Ratings.COL_USERID
            COL_ITEMID:str = Ratings.COL_MOVIEID
        elif self.datasetClass is DatasetRetailRocket:
            from datasets.retailrocket.events import Events  # class
            COL_USERID:str = Events.COL_VISITOR_ID
            COL_ITEMID:str = Events.COL_ITEM_ID
        elif self.datasetClass is DatasetST:
            from datasets.slantour.events import Events  # class
            COL_USERID:str = Events.COL_USER_ID
            COL_ITEMID:str = Events.COL_OBJECT_ID

        updateDF:DataFrame = DataFrame([[userID, itemID]], columns=[COL_USERID, COL_ITEMID])
        portfolio.update(updateDF, {})

        model:DataFrame = self.modelsDict[variant]
        rItemIDs, rItemIDsWithtResp = portfolio.recommend(userID, model, {
                        APortfolio.ARG_NUMBER_OF_AGGR_ITEMS:numberOfItems,
                        APortfolio.ARG_NUMBER_OF_RECOMM_ITEMS:100,
                        ARecommender.ARG_ALLOWED_ITEMIDS:allowedItemIDs})
        print(rItemIDs)
        print(rItemIDsWithtResp)

        evalTool:AEvalTool = self.evalToolsDict[variant]
        evalTool.displayed(rItemIDsWithtResp, self.modelsDict[variant], self.evaluation)

        self.send_response(200)
        self.send_header('Content-Type', 'text/plain; charset=utf-8')
        self.end_headers()
        message:str = "recommendation: userID=" + str(userID) + ", rItemIDs=" + str(rItemIDs) + ", rItemIDsWithtResp=" + str(rItemIDsWithtResp)
        self.wfile.write(message.encode('utf-8'))
        self.wfile.write(b'\n')


if __name__ == '__main__':

    server = HTTPServer(('', 8080), HeterRecomHTTPHandler)
    server.serve_forever()
