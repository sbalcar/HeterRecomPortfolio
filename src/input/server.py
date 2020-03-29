import sqlite3
import datetime

class Database:
  def __init__(self, dbName):
    self.databaseName = dbName

    connection = sqlite3.connect(self.databaseName)
    cursor = connection.cursor()

    cursor.execute('CREATE TABLE IF NOT EXISTS Users(UserID INTEGER PRIMARY KEY, Name TEXT)')
    cursor.execute('CREATE TABLE IF NOT EXISTS Items(ItemID INTEGER PRIMARY KEY, Name TEXT)')
    cursor.execute('CREATE TABLE IF NOT EXISTS Recommenders(RecommenderID INTEGER PRIMARY KEY, Name TEXT)')
    cursor.execute("""
      CREATE TABLE IF NOT EXISTS RecommendedItems(
      RecommendationID integer PRIMARY KEY,
      UserID INTEGER REFERENCES Users(UserID),
      ItemID INTEGER REFERENCES Items(ItemID),
      RecommenderID INTEGER REFERENCES Recommenders(RecommenderID),
      Timestamp TIMESTAMP,
      Clicked BOOLEAN)
      """)

    connection.commit()
    connection.close()


  def insertUser(self, username):
    connection = sqlite3.connect(self.databaseName)
    cursor = connection.cursor()

    cursor.execute("INSERT INTO Users (Name) VALUES (?)", (username,))

    connection.commit()
    connection.close()


  def insertItem(self, name):
    connection = sqlite3.connect(self.databaseName)
    cursor = connection.cursor()

    cursor.execute("INSERT INTO Items (Name) VALUES (?)", (name,))

    connection.commit()
    connection.close()


  def insertRecommendation(self, userID, itemID, clicked, timestamp = datetime.datetime.now()):
    connection = sqlite3.connect(self.databaseName)
    cursor = connection.cursor()

    cursor.execute("INSERT INTO RecommendedItems (UserID, ItemID, Timestamp, Clicked) VALUES (?,?,?,?)", (userID, itemID, timestamp, clicked))

    connection.commit()
    connection.close()


  def getPreviousRecommendations(self, userID):
    connection = sqlite3.connect(self.databaseName)
    cursor = connection.cursor()

    cursor.execute("SELECT * FROM RecommendedItems WHERE UserID=?",(userID,))
    previousRecommendations = cursor.fetchall()

    connection.commit()
    connection.close()

    return previousRecommendations

  def generateRandomData(self, weekCount, userCount, itemCount, recommenderCount, sessionsPerWeek, recommendationsPerSession, recommenderOverlap):
    raise NotImplementedError()