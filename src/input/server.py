import sqlite3
import datetime
import random
import string


def randomDatetime(start, end):
  '''
  Generates a random time (precise to seconds) between start and end
  :param start: datetime.datetime object
  :param end: datetime.datetime object
  :return: datetime.datetime object with time between start and end
  '''

  if type(start) is not datetime.datetime:
    raise ValueError("Argument start is not type datetime.datetime")
  if type(end) is not datetime.datetime:
    raise ValueError("Argument end is not type datetime.datetime")


  timeFrame = end - start
  seconds = (timeFrame.days * 24 * 60 * 60) + timeFrame.seconds
  randomSecond = random.randrange(seconds)
  return start + datetime.timedelta(seconds=randomSecond)


def penalizeIfRecentlyRecommended(database, userID, itemID, score, maxFinalPenalty, minTimeDiff, maxTimeDiff, minSinglePenalty, maxSinglePenalty):
  '''
  reduces given score based on how many times has the item already been recommended to the user in the last week.
  :param database: database with all previous recommendations
  :param userID: id of the user we're recommending to
  :param itemID: item id
  :param score: the score to adjust
  :param maxFinalPenalty: maximal penalty given to an item
  :param minTimeDiff: single recommendation penalty starts to decrease after this age (seconds)
  :param maxTimeDiff: single recommendation penalty remains minimal after this age (seconds)
  :param minSinglePenalty: minimal single recommendation penalty
  :param maxSinglePenalty: maximal single recommendation penalty
  :return: penalized score
  '''

  recommendationCount = database.getInteractionCount(userID) # not used in this version
  previousRecommendations = database.getPreviousRecommendations(userID, itemID)
  penalty = 0
  for recommendation in previousRecommendations:
    penalty += getPenaltyLinear(recommendation[6], minTimeDiff, maxTimeDiff, minSinglePenalty, maxSinglePenalty)

  return score - score * min(penalty, maxFinalPenalty)


def getPenaltyLinear(timeDiff, minTimeDiff, maxTimeDiff, minPenalty, maxPenalty):
  '''
  computes linear penalty based on a line equation
  y = m*x + c   ~   penalty = m*timeDiff + c

  the line is defined by two points:
  (minTimeDiff, maxPenalty), (maxTimeDiff, minPenalty)

  :param timeDiff: time since the recommendation (in seconds)
  :param minTimeDiff: penalty starts to decrease after this time
  :param maxTimeDiff: penalty remains minimal after this time
  :param minPenalty: minimal penalty given (for timeDiff >= maxTimeDiff)
  :param maxPenalty: maximal penalty given (for timeDiff <= minTimeDiff)
  :return: computed penalty
  '''
  m = (minPenalty - maxPenalty) / (maxTimeDiff - minTimeDiff)
  c = maxPenalty - (m * maxTimeDiff)

  return (m * timeDiff) + c


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


  def insertRecommendation(self, userID, itemID, RecommenderID,clicked, timestamp = datetime.datetime.now()):
    connection = sqlite3.connect(self.databaseName)
    cursor = connection.cursor()

    cursor.execute("INSERT INTO RecommendedItems (UserID, ItemID, RecommenderID, Timestamp, Clicked) VALUES (?,?,?,?,?)", (userID, itemID, RecommenderID, timestamp, clicked))

    connection.commit()
    connection.close()

  # currently only recommeder 3
  def getPreviousRecommendations(self, userID, itemID):
    connection = sqlite3.connect(self.databaseName)
    cursor = connection.cursor()

    cursor.execute("SELECT *, round((julianday('now') - julianday(Timestamp)) * 86400) FROM RecommendedItems WHERE UserID=? AND ItemID=? AND RecommenderID = 3",(userID, itemID))
    previousRecommendations = cursor.fetchall()

    connection.commit()
    connection.close()

    return previousRecommendations

  def getInteractionCount(self, userID):
    connection = sqlite3.connect(self.databaseName)
    cursor = connection.cursor()

    cursor.execute("SELECT * FROM RecommendedItems WHERE UserID=? AND Timestamp >= date('now','-7 day') GROUP BY Timestamp",
                   (userID,))
    previousRecommendations = cursor.fetchall()

    connection.commit()
    connection.close()

    return len(previousRecommendations)

  
  
  def generateRandomData(self, userCount, itemCount, recommenderCount, sessionsPerWeek, recommendationsPerSession, recommenderOverlap, itemsPerRecommendation):
    connection = sqlite3.connect(self.databaseName)
    cursor = connection.cursor()

    # generate users
    for i in range(userCount):
      username = "U" + "".join(random.choice(string.ascii_lowercase) for j in range(8))
      cursor.execute("INSERT INTO Users (Name) VALUES (?)", (username,))

    # generate items
    for i in range(itemCount):
      name = "I" + "".join(random.choice(string.ascii_lowercase) for j in range(8))
      cursor.execute("INSERT INTO Items (Name) VALUES (?)", (name,))

    # generate recommenders
    for i in range(recommenderCount):
      cursor.execute("INSERT INTO Recommenders (Name) VALUES (?)", (i,))


    cursor.execute("SELECT * FROM Users")
    users = cursor.fetchall()

    cursor.execute("SELECT * FROM Items")
    items = cursor.fetchall()

    cursor.execute("SELECT * FROM Recommenders")
    recommenders = cursor.fetchall()

    for j in range(sessionsPerWeek):
      user = random.choice(users)
      for k in range(recommendationsPerSession):
        time = randomDatetime(datetime.datetime(2020, 3, 23, 0, 0, 0), datetime.datetime(2020, 3, 30, 8, 59, 59))
        for l in range(recommenderOverlap):
          item = random.choice(items)
          for recommender in recommenders:
            cursor.execute(
              "INSERT INTO RecommendedItems (UserID, ItemID, RecommenderID, Timestamp, Clicked) VALUES (?,?,?,?,?)",
              (user[0], item[0], recommender[0], time + l * datetime.timedelta(seconds=120), False))
        for l in range(itemsPerRecommendation - recommenderOverlap):
          item = random.choice(items)
          recommender = random.choice(recommenders)
          cursor.execute(
            "INSERT INTO RecommendedItems (UserID, ItemID, RecommenderID, Timestamp, Clicked) VALUES (?,?,?,?,?)",
            (user[0], item[0], recommender[0], time + l * datetime.timedelta(seconds=120), False))

    connection.commit()
    connection.close()




db = Database("test.db")

print(penalizeIfRecentlyRecommended(db, 71, 4443, 85, 0.8, 24*60*60, 15*24*60*60, 0.01, 0.12))