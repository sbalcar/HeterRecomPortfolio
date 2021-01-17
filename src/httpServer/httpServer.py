#!/usr/bin/python3
import ast
import time
import os
from datetime import datetime

from typing import Dict
from typing import List
from pandas import DataFrame
from pandas.core.series import Series #class

from portfolio.aPortfolio import APortfolio #class

from http.server import BaseHTTPRequestHandler, HTTPServer

from urllib.parse import parse_qs, urlparse, unquote

from datasets.aDataset import ADataset #class
from datasets.datasetML import DatasetML #class
from datasets.datasetRetailrocket import DatasetRetailRocket #class
from datasets.datasetST import DatasetST #class

from datasets.slantour.events import Events #class
from datasets.ml.ratings import Ratings #class

from evaluationTool.aEvalTool import AEvalTool #class
from evaluationTool.evalToolSingleMethod import EToolSingleMethod #class
from evaluationTool.evalToolContext import EvalToolContext #class
from recommender.aRecommender import ARecommender #class

from history.aHistory import AHistory #class

import pandas as pd
import numpy as np
import json


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)




class HeterRecomHTTPHandler(BaseHTTPRequestHandler):

    ACTION_CLICK:str = "click"
    ACTION_VISIT:str = "visit"
    ACTION_RECOMMEND:str = "recommend"

    ARG_VARIANTID:str = "variantID"
    ARG_ACTIONID:str = "actionID"
    ARG_UPDU_TYPE:str = "updtType"
    ARG_USERID:str = "userID"
    ARG_ITEMID:str = "itemID"
    ARG_SESSIONID:str = "session"
    ARG_PAGETYPE:str = "pageType"
    ARG_NUMBER_OF_ITEMS:str = "numberOfItems"
    ARG_RITEMIDS_WITH_RESP:str = "rItemIDsWithResponsibility"
    ARG_ALLOWED_ITEMIDS:str = ARecommender.ARG_ALLOWED_ITEMIDS


    VARIANT_1:str = "1"
    VARIANT_2:str = "2"
    VARIANT_3:str = "3"
    VARIANT_4:str = "4"
    VARIANT_5:str = "5"

    #portfolioDict:Dict[]
    #modelsDict:Dict[]
    #evalToolsDict:Dict[]
    #evaluation

    #computationFileDict:Dict[str,file]
    #portModelTimeEvolutionFilesDict:Dict[str,file]
    #historyOfRecommendationFilesDict:Dict[str,file]

    #datasetClass


    def initialization(cls, portfolio:List[APortfolio], models:List[DataFrame], evalTools:List[AEvalTool],
                       histories:List[AHistory], datasetClass):
        print("Initialization")

        types:List[str] = [cls.VARIANT_1, cls.VARIANT_2, cls.VARIANT_3, cls.VARIANT_4, cls.VARIANT_5]
        types = types[:len(portfolio)]

        cls.portfolioDict = dict(zip(types,portfolio))
        cls.modelsDict = dict(zip(types,models))
        cls.evalToolsDict = dict(zip(types,evalTools))
        cls.historiesDict = dict(zip(types,histories))

        #  HeterRecomHTTPHandler.evaluation:Dict = {}
        cls.datasetClass = datasetClass

        for typeIdI, portfolioI in zip(types, portfolio):
            fileNameI:str = dir + os.sep + "computation-" + portfolioI.getPortfolioID() + ".txt"
            cls.computationFileDict[typeIdI] = open(fileNameI, "a")

        for typeIdI, portfolioI in zip(types, portfolio):
            fileNameI:str = dir + os.sep + "portfModelTimeEvolution-" + portfolioI.getPortfolioID() + ".txt"
            cls.portModelTimeEvolutionFilesDict[typeIdI] = open(fileNameI, "a")

        for typeIdI, portfolioI in zip(types, portfolio):
            fileNameI:str = dir + os.sep + "historyOfRecommendation-" + portfolioI.getPortfolioID() + ".txt"
            cls.historyOfRecommendationFilesDict[typeIdI] = open(fileNameI, "a")



    def do_POST(self):
        print("doPOST")
        query = urlparse(self.path).query
        print(query)
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
        variant = "2"
        print("variant: " + variant)

        action:str = params[self.ARG_ACTIONID]
        print("action: " + action)

        userID:int = int(params[self.ARG_USERID])
        print("userID: " + str(userID))

        itemID: int = int(params.get(self.ARG_ITEMID, 0))
        print("itemID: " + str(itemID))

        if action == self.ACTION_CLICK:
 
            content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
            post_data = unquote(self.rfile.read(content_length).decode("utf-8")) # <--- Gets the data itself
            post_data = post_data.replace("&quot;","\"")
             
            #print("post data") 
            #print(post_data)  
            data = dict(qc.split("=") for qc in post_data.split("&"))  
                      
            sessionID:int = int(params[self.ARG_SESSIONID])
            print("sessionID: " + str(sessionID))
            
            pageType:str = params[self.ARG_PAGETYPE] 
            print("pageType: " + str(pageType))
            
            rItemIdsWithRespStr:str = data.get(self.ARG_RITEMIDS_WITH_RESP)
            print(rItemIdsWithRespStr)  
            if rItemIdsWithRespStr is not None:                  
                rItemIds = json.loads(rItemIdsWithRespStr)
                #print(rItemIds)
                
            else:
                self.send_error(404)
                return
                     
            rItemIdsWithResp = [[itemID, rItemIds]]
            print("rItemIdsWithResp: " + str(rItemIdsWithResp))

            self.__click(variant, userID, itemID, sessionID, pageType, rItemIdsWithResp)
            return
            
        elif action == self.ACTION_RECOMMEND:
            if not self.ARG_NUMBER_OF_ITEMS in params:
                params[self.ARG_NUMBER_OF_ITEMS] = 20
                
            if not self.ARG_SESSIONID in params:
                print("error in "+self.ARG_SESSIONID )
                self.send_error(404)
                return
            if not self.ARG_PAGETYPE in params:
                print("error in "+self.ARG_PAGETYPE )
                self.send_error(404)
                return    
                
            sessionID:int = int(params[self.ARG_SESSIONID])
            print("sessionID: " + str(sessionID))
            
            pageType:str = params[self.ARG_PAGETYPE] 
            print("pageType: " + str(pageType)) 
              
            content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
            post_data = unquote(self.rfile.read(content_length).decode("utf-8")) # <--- Gets the data itself 
            data = dict(qc.split("=") for qc in post_data.split("&"))  
            #print("post data") 
            #print(data)            
                
            numberOfItems:int = int(params[self.ARG_NUMBER_OF_ITEMS])
            #TODO: rewrite this to collect it from content
            
            allowedItemIDsStr:str = data.get(self.ARG_ALLOWED_ITEMIDS)
            if allowedItemIDsStr is not None:                  
                allowedItemIDs:List[int] = json.loads("["+allowedItemIDsStr+"]")
                print(allowedItemIDs[0:5])
            else:
                allowedItemIDs = None

            print("numberOfItems: " + str(numberOfItems))
            self.__recommend(variant, userID, itemID, sessionID, pageType,  numberOfItems, allowedItemIDs)
            return
            
        else:
            print("error in action type")
            self.send_error(404)
    
    def do_GET(self):
        print("doGET")
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
        variant = "2"
        print("variant: " + variant)

        action:str = params[self.ARG_ACTIONID]
        print("action: " + action)

        userID:int = int(params[self.ARG_USERID])
        print("userID: " + str(userID))

        itemID: int = int(params[self.ARG_ITEMID])
        print("itemID: " + str(itemID))

        if action == self.ACTION_VISIT:
        
            if not self.ARG_SESSIONID in params:
                self.send_error(404)
                return
            if not self.ARG_PAGETYPE in params:
                self.send_error(404)
                return
            sessionID:int = int(params[self.ARG_SESSIONID])
            pageType:str = params[self.ARG_PAGETYPE]
             
            self.__visit(variant, userID, itemID, sessionID, pageType)
            return

        else:
            self.send_error(404)


    def __click(self, variant:str, userID:int, itemID:int, sessionID:int, pageType:str, rItemIDsWithResponsibility:List[tuple]):
        print("click")
        if not variant in self.portfolioDict:
            self.send_error(404)
            return

        evalTool:AEvalTool = self.evalToolsDict[variant]
        evalTool.click(rItemIDsWithResponsibility, itemID, self.modelsDict[variant], {EvalToolContext.ARG_USER_ID: userID,
                        EvalToolContext.ARG_ITEM_ID: itemID,
                        EvalToolContext.ARG_SENIORITY: sessionID,
                        EvalToolContext.ARG_PAGE_TYPE: pageType
                        })

        print("evaluation: ", str(self.evaluation))

        self.portModelTimeEvolutionFilesDict[variant].write("currentItemID: " + str(itemID) + "\n")
        self.portModelTimeEvolutionFilesDict[variant].write("userID: " + str(userID) + "\n")
        self.portModelTimeEvolutionFilesDict[variant].write(str(self.modelsDict[variant]) + "\n\n")
        self.portModelTimeEvolutionFilesDict[variant].flush()

        self.computationFileDict[variant].write("RatingI: " + str(datetime.now().strftime("%H:%M:%S")) + "\n")
        self.computationFileDict[variant].write("PortfolioIds: " + str(variant) + "\n")
        #self.computationFileDict[variant].write("Evaluations: " + str(evaluations) + "\n")
        self.computationFileDict[variant].flush()

        self.send_response(200)
        self.send_header('Content-Type', 'text/plain; charset=utf-8')
        self.end_headers()
        message:str = "click: userID=" + str(userID) + ", itemID=" + str(itemID)
        self.wfile.write(message.encode('utf-8'))
        self.wfile.write(b'\n')


    def __visit(self, variant:str, userID:int, itemID:int, sessionID:int, pageType:str):
        print("visit")
        if not variant in self.portfolioDict:
            self.send_error(404)
            return

        portfolio:APortfolio = self.portfolioDict[variant]

        if self.datasetClass is DatasetML:
            COL_USERID:str = Ratings.COL_USERID
            COL_ITEMID:str = Ratings.COL_MOVIEID
            COL_START_DATE_TIME:str = Ratings.COL_TIMESTAMP
            updateDF:DataFrame = DataFrame([[userID, itemID, time.time()]], columns=[COL_USERID, COL_ITEMID, COL_START_DATE_TIME])
        elif self.datasetClass is DatasetRetailRocket:
            from datasets.retailrocket.events import Events  # class
            COL_USERID:str = Events.COL_VISITOR_ID
            COL_ITEMID:str = Events.COL_ITEM_ID
            COL_START_DATE_TIME:str = Events.COL_TIME_STAMP
            updateDF:DataFrame = DataFrame([[userID, itemID, time.time()]], columns=[COL_USERID, COL_ITEMID, COL_START_DATE_TIME])
        elif self.datasetClass is DatasetST:
            from datasets.slantour.events import Events  # class
            COL_USERID:str = Events.COL_USER_ID
            COL_ITEMID:str = Events.COL_OBJECT_ID
            COL_SESSION:str = Events.COL_SESSION_ID
            COL_PAGETYPE:str = Events.COL_PAGE_TYPE
            COL_START_DATE_TIME:str = Events.COL_START_DATE_TIME
                        
            updateDF:DataFrame = DataFrame([[userID, itemID, sessionID, pageType, time.time()]], columns=[COL_USERID, COL_ITEMID, COL_SESSION, COL_PAGETYPE, COL_START_DATE_TIME])


        portfolio.update(updateDF, {})

        self.send_response(200)
        self.send_header('Content-Type', 'text/plain; charset=utf-8')
        self.end_headers()
        message:str = ""
        self.wfile.write(message.encode('utf-8'))
        self.wfile.write(b'\n')


    def __recommend(self, variant:str, userID:int, itemID:int, sessionID:int, pageType:str, numberOfItems:int, allowedItemIDs):
        print("recommend")
        if not variant in self.portfolioDict:
            self.send_error(404)
            return

        portfolio:APortfolio = self.portfolioDict[variant]


        model:DataFrame = self.modelsDict[variant]
        rItemIDs, rItemIDsWithtResp = portfolio.recommend(userID, model, {
                        APortfolio.ARG_NUMBER_OF_AGGR_ITEMS:numberOfItems,
                        APortfolio.ARG_NUMBER_OF_RECOMM_ITEMS:100,
                        ARecommender.ARG_ALLOWED_ITEMIDS:allowedItemIDs,
                        EvalToolContext.ARG_USER_ID: userID,
                        EvalToolContext.ARG_ITEM_ID: itemID,
                        EvalToolContext.ARG_SENIORITY: sessionID,
                        EvalToolContext.ARG_PAGE_TYPE: pageType
                        })
        #print(rItemIDs)
        #print(rItemIDsWithtResp)

        evalTool:AEvalTool = self.evalToolsDict[variant]
        evalTool.displayed(rItemIDsWithtResp, self.modelsDict[variant], {
                        APortfolio.ARG_NUMBER_OF_AGGR_ITEMS:numberOfItems,
                        APortfolio.ARG_NUMBER_OF_RECOMM_ITEMS:100,
                        ARecommender.ARG_ALLOWED_ITEMIDS:allowedItemIDs,
                        EvalToolContext.ARG_USER_ID: userID,
                        EvalToolContext.ARG_ITEM_ID: itemID,
                        EvalToolContext.ARG_SENIORITY: sessionID,
                        EvalToolContext.ARG_PAGE_TYPE: pageType
                        })#self.evaluation)


        self.historiesDict[variant].insertRecomAndClickedItemIDs(userID, rItemIDs, [])
        # delete log of history
        lengthOfHistory:int = 100 # * self._recomRepetitionCount * self._numberOfAggrItems
        #print("lengthOfHistory: " + str(lengthOfHistory))
        self.historiesDict[variant].deletePreviousRecomOfUser(userID, lengthOfHistory)


        self.historyOfRecommendationFilesDict[variant].write("userID: " + str(userID) + "\n")
        self.historyOfRecommendationFilesDict[variant].write("currentItemID: " + str(itemID) + "\n")
        self.historyOfRecommendationFilesDict[variant].write("sessionID: " + str(sessionID) + "\n")
        self.historyOfRecommendationFilesDict[variant].write("pageType: " + str(pageType) + "\n\n")
        self.historyOfRecommendationFilesDict[variant].write("rItemIDs: " + str(rItemIDs) + "\n")
        self.historyOfRecommendationFilesDict[variant].flush()

        self.send_response(200)
        self.send_header('Content-Type', 'text/plain; charset=utf-8')
        self.end_headers()
        #message:str = "{'rItemIDs':" + json.dumps(rItemIDs) + ", 'rItemIDsWithtResp':" + json.dumps(rItemIDsWithtResp) +"}"
        message:str = json.dumps(rItemIDsWithtResp, cls=NpEncoder)
        self.wfile.write(message.encode('utf-8'))
        #self.wfile.write(b'\n')


if __name__ == '__main__':

    server = HTTPServer(('', 8080), HeterRecomHTTPHandler)
    server.serve_forever()
