import os
import logging
import pandas as pd
import mysql.connector
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
dbname = os.getenv("dbname")

class EmbeddingDatabaseBuilder:
    def __init__(self, embedding_model="models/embedding-001"):
        self.embedding = GoogleGenerativeAIEmbeddings(model=embedding_model)

    def sql_connection(self):
        return mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=dbname
        )


    def process_dataframe(self, df: pd.DataFrame, source: str):
        documents = []
        documents_metadata_only = []

        for idx, row in df.iterrows():
            documents.append(
                Document(
                    page_content=row['clause_text'],
                    metadata={"clause_type": row['clause_type']}
                )
            )
            documents_metadata_only.append({
                "clause_type": row['clause_type'],
                "source": source,
                "row": idx
            })

        logger.info("Adding new file to FAISS index: %s", source)
        logger.info("Loaded %d valid documents from this file.", len(documents))

        faiss_path = "faiss_index"

        logger.info("FAISS index found. Loading and appending new documents...")
        db = FAISS.load_local(
            folder_path=faiss_path,
            embeddings=self.embedding,
            allow_dangerous_deserialization=True
        )

        db.add_documents(documents)
        logger.info("Appended %d documents from '%s' to FAISS index.", len(documents), source)

        db.save_local(faiss_path)
        logger.info("FAISS index saved to '%s'.", faiss_path)

        return documents_metadata_only


    def insert_into_mysql(self, metadata_list):
        try:
            conn = self.sql_connection()
            cursor = conn.cursor()
            logger.info("MySQL connection established.")

            for item in metadata_list:
                cursor.execute(
                    "INSERT INTO clause_metadata (clause_type, source, row_number) VALUES (%s, %s, %s)",
                    (item['clause_type'], item['source'], item['row'])
                )
            conn.commit()
            logger.info("Metadata inserted into MySQL.")

        except mysql.connector.Error as err:
            logger.error("MySQL Error: %s", err)

        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
                logger.info("MySQL connection closed.")

    
    def store_followup(self, question: str, response: str, session_id=None):
        try:
            conn = self.sql_connection()
            cursor = conn.cursor()

            logger.info("Storing follow-up QnA")

            cursor.execute("""
                INSERT INTO follow_up_questions (session_id, question, response)
                VALUES (%s, %s, %s)
            """, (session_id, question, response))

            conn.commit()
            logger.info("Follow-up QnA stored successfully.")

            # return session_id

        except mysql.connector.Error as err:
            logger.error("MySQL Error in store_followup(): %s", err)

        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
                logger.info("MySQL connection closed after storing follow-up.")


    def get_session_messages(self, session_id):
     
        try:
            conn = self.sql_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT question, response
                FROM follow_up_questions
                WHERE session_id = %s
                ORDER BY created_at ASC
            """, (session_id,))
            rows = cursor.fetchall()
            return rows

        except mysql.connector.Error as err:
            logger.error("MySQL Error in get_session_messages(): %s", err)
            return []

        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()


if __name__ == "__main__":

    path = "/home/ubaid-ur-rehman/Downloads/legal_clause/term.csv"
    df = pd.read_csv(path)  

    udb = EmbeddingDatabaseBuilder()
    metadata = udb.process_dataframe(df=df, source= path)
    udb.insert_into_mysql(metadata_list=metadata)