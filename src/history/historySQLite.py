#!/usr/bin/python3

import sqlite3
import datetime
import random
import string
import os

from typing import List
from pandas.core.series import Series #class

import pandas as pd

from history.aHistory import AHistory #class
from userBehaviourDescription.userBehaviourDescription import UserBehaviourDescription #class


class HistorySQLite(AHistory):
    def __init__(self, dbName:str):
        self.databaseFileName = ".." + os.sep + ".." + os.sep + os.sep + "database" + os.sep + dbName + ".db"

        self.connection = sqlite3.connect(self.databaseFileName)
        cursor = self.connection.cursor()

        cursor.execute('CREATE TABLE IF NOT EXISTS Users(UserID INTEGER PRIMARY KEY, Name TEXT)')
        cursor.execute('CREATE TABLE IF NOT EXISTS Items(ItemID INTEGER PRIMARY KEY, Name TEXT)')
        cursor.execute('CREATE TABLE IF NOT EXISTS Recommenders(RecommenderID INTEGER PRIMARY KEY, Name TEXT)')
        # cursor.execute("""
        #  CREATE TABLE IF NOT EXISTS RecommendedItems(
        #  RecommendationID integer PRIMARY KEY,
        #  UserID INTEGER REFERENCES Users(UserID),
        #  ItemID INTEGER REFERENCES Items(ItemID),
        #  RecommenderID INTEGER REFERENCES Recommenders(RecommenderID),
        #  Timestamp TIMESTAMP,
        #  Position INTEGER,
        #  Clicked BOOLEAN)
        #  """)

        # simplified for the off-line evaluations
        cursor.execute("""
          CREATE TABLE IF NOT EXISTS RecommendedItems(
          RecommendationID integer PRIMARY KEY,
          UserID INTEGER,
          ItemID INTEGER,
          Position INTEGER,
          Observation REAL,
          Clicked BOOLEAN,
          Timestamp TIMESTAMP)
          """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS uid 
            ON RecommendedItems(UserID);
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS tmstmp 
            ON RecommendedItems(Timestamp);
        """)

        self.connection.commit()

    def insertUser(self, username):
        # connection = sqlite3.connect(self.databaseName)
        cursor = self.connection.cursor()

        cursor.execute("INSERT INTO Users (Name) VALUES (?)", (username,))

        self.connection.commit()
        # connection.close()

    def insertItem(self, name):
        # connection = sqlite3.connect(self.databaseName)
        cursor = self.connection.cursor()

        cursor.execute("INSERT INTO Items (Name) VALUES (?)", (name,))

        self.connection.commit()
        # connection.close()

    def insertRecommendation(self, userID:int, itemID:int, position:int, uObservation:float, clicked:bool, timestamp=datetime.datetime.now()):
        # connection = sqlite3.connect(self.databaseName)
        cursor = self.connection.cursor()

        cursor.execute(
            "INSERT INTO RecommendedItems (UserID, ItemID, Position, Observation, Clicked, Timestamp) VALUES (?,?,?,?,?,?)",
            (userID, itemID, position, uObservation, clicked, timestamp))

        self.connection.commit()
        # connection.close()

    # currently only recommeder 1
    def getPreviousRecomOfUser(self, userID, limit=100):
        # connection = sqlite3.connect(self.databaseName)
        cursor = self.connection.cursor()

        cursor.execute("SELECT * FROM RecommendedItems WHERE UserID=? ORDER BY Timestamp desc LIMIT ?", (userID, limit))
        previousRecommendations = cursor.fetchall()

        self.connection.commit()
        # connection.close()

        # SQLite does not have a separate Boolean storage class. Instead, Boolean values are stored as integers 0 (false) and 1 (true).
        result:List[tuple] = []
        for rI in previousRecommendations:
            rListI = list(rI)
            if rListI[5] == 0:
                rListI[5] = False
            elif rListI[5] == 1:
                rListI[5] = True
            else:
                print("Error")
            result.append(tuple(rListI))
        return result

    def getPreviousRecomOfUserAndItem(self, userID:int, itemID:int, limit:int=100):
        # connection = sqlite3.connect(self.databaseName)
        cursor = self.connection.cursor()

        cursor.execute("SELECT * FROM RecommendedItems WHERE UserID=? AND ItemID=? ORDER BY Timestamp desc LIMIT ?", (userID, itemID, limit))
        previousRecommendations = cursor.fetchall()

        self.connection.commit()
        # connection.close()

        # SQLite does not have a separate Boolean storage class. Instead, Boolean values are stored as integers 0 (false) and 1 (true).
        result:List[tuple] = []
        for rI in previousRecommendations:
            rListI = list(rI)
            if rListI[5] == 0:
                rListI[5] = False
            elif rListI[5] == 1:
                rListI[5] = True
            else:
                print("Error")
            result.append(tuple(rListI))
        return result


    def getInteractionCount(self, userID, limit):
        # connection = sqlite3.connect(self.databaseName)
        cursor = self.connection.cursor()

        cursor.execute("SELECT * FROM RecommendedItems WHERE  UserID=? ORDER BY Timestamp desc LIMIT ?",
                       (userID, limit))
        previousRecommendations = cursor.fetchall()

        self.connection.commit()
        # connection.close()

        return len(previousRecommendations)

    def print(selfs):
        pass
