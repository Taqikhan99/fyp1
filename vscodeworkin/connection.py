import pyodbc

# database connection class
class DbConnection:
    def __init__(self) -> None:
        self.cursor=None
    def getCursor(self):
        return self.cursor
    def connectToDb(self):

        # creating connection
        connection = pyodbc.connect('DRIVER={SQL Server};SERVER=TAQILAPTOP\SQLEXPRESS2019;DATABASE=TaqiComputers_DB;Trusted_Connection=yes;')
        self.cursor=connection.cursor()
        print("Connection successful")
        # # creating cursor to execute commands
        # cursor=connection.cursor()
        # cursor.execute("Select * from tbPerson")
        # personData= cursor.fetchall()

        
