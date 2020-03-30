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


def penalizeIfRecentlyRecommended(database, userID, itemID, score, previouslyRecommendedPenalty = 0.12):
  '''
  reduces given score based on how many times has the item already been recommended to the user in the last week.
  :param database: database with all previous recommendations
  :param userID: id of the user we're recommending to
  :param itemID: item id
  :param score: the score to adjust
  :param previouslyRecommendedPenalty: the base penalty that gets added for every previous recommendation (0.1 = 10%)
  :return: penalized score
  '''

  recommendationCount = database.getInteractionCount(userID) # not used in this version
  previousRecommendations = database.getPreviousRecommendations(userID, itemID)
  penalty = 0
  penaltyDecay = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2] # indexed by age in days
  for recommendation in previousRecommendations:
    assert (recommendation[6] <= 8)
    penalty += previouslyRecommendedPenalty * penaltyDecay[int(recommendation[6])] # previousRecommendations[6] = recommendation age

  return score - score * min(penalty, 0.8)


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

    cursor.execute("SELECT *, round(julianday('now') - julianday(Timestamp)) FROM RecommendedItems WHERE UserID=? AND ItemID=? AND RecommenderID = 3 AND Timestamp >= date('now','-7 day')",(userID, itemID))
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

print(penalizeIfRecentlyRecommended(db, 71, 4443, 85, 0.12))