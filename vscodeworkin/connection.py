import pyodbc


# creating connection
connection = pyodbc.connect('DRIVER={SQL Server};SERVER=TAQILAPTOP\SQLEXPRESS2019;DATABASE=TaqiComputers_DB;Trusted_Connection=yes;')

# creating cursor to execute commands
cursor=connection.cursor()
cursor.execute("Select * from tbPerson")
personData= cursor.fetchall()

# print(personData[0][0])
