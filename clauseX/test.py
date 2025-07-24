import os
import glob
import logging
import pandas as pd
import mysql.connector
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load .env variables
load_dotenv()
host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
dbname = os.getenv("dbname")

class EmbeddingDatabaseBuilder():
    def __init__(self, directory=None):
        self.files = directory
        self.db = self.initialize_db()

    def initialize_db(self):
        documents_metadata_only = []   # For MySQL: only metadata (no content)
        
        logger.info("Processing files from: %s", self.files)
        
        for csv_file in glob.glob(os.path.join(self.files, "*.csv")):
            df = pd.read_csv(csv_file)
            for idx, row in df.iterrows():
                if pd.notnull(row['clause_text']) and pd.notnull(row['clause_type']):
                    documents_metadata_only.append({
                        "clause_type": row['clause_type'],
                        "source": csv_file,
                        "row": idx
                    })

        return documents_metadata_only

    def insert_into_mysql(self):
        try:
            conn = mysql.connector.connect(
                host=host,
                user=user,
                password=password,
                database=dbname
            )
            cursor = conn.cursor()  # Create a cursor object to execute SQL queries using the connection
            logger.info("Database connection established.")

            # Clear existing data before inserting new rows
            cursor.execute("DELETE FROM clause_metadata")
            conn.commit()  # Commit the deletion
            logger.info("Old data cleared from clause_metadata table.")

            logger.info("Starting insertion of metadata into clause_metadata table...")

            for item in self.db[:20]:
                insert_query = """
                INSERT INTO clause_metadata (clause_type, source, row_number)
                VALUES (%s, %s, %s)
                """
                values = (item['clause_type'], item['source'], item['row'])
                cursor.execute(insert_query, values)

            conn.commit()
            logger.info("Data inserted successfully into clause_metadata table.")

        except mysql.connector.Error as err:
            logger.error(" MySQL Error: %s", err)

        finally:
            if cursor:
                cursor.close()  # Close the cursor to free up memory and DB resources
                logger.info("Cursor closed.")
            if conn:
                conn.close()    # Close the connection to avoid connection leaks
                logger.info("Database connection closed.")


if __name__ == "__main__":
    rag_instance = EmbeddingDatabaseBuilder(
        directory="/home/ubaid-ur-rehman/Downloads/legal_clause/"
    )

    rag_instance.insert_into_mysql()  # Insert metadata into MySQL
