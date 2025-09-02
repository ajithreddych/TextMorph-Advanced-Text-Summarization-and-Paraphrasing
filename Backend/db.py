import mysql.connector
from mysql.connector import Error

def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",         # change to your MySQL username
            password="Ajith@2005", # change to your MySQL password
            database="text_summarization"
        )
        return connection
    except Error as e:
        print("Error while connecting to MySQL", e)
        return None
