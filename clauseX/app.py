import io
import uuid # uuid module to generate unique session IDs for tracking user sessions
import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File,Path, BackgroundTasks
from query_router import QueryRouter
from update_db import EmbeddingDatabaseBuilder


app = FastAPI()

class QueryInput(BaseModel):
    text: str


router = QueryRouter()
builder = EmbeddingDatabaseBuilder()
create_session_id = str(uuid.uuid4())

@app.get("/")
def home():
    return {"message": "hi there"}

@app.post("/predict")
def predict(user_query: QueryInput):

    previous_qna = builder.get_session_messages(create_session_id)  # returns list of tuples (question, response)

    chat_history = ""
    for q, r in previous_qna:
        chat_history += f"User: {q}\nAssistant: {r}\n"

    query_with_context = chat_history + f"User: {user_query.text}\nAssistant:"
    result = router.run(query=query_with_context)

    builder.store_followup(
        question=user_query.text,
        response=result,
        session_id=create_session_id
    )

    return {
        "response": result,
        "session_id": create_session_id
    }


@app.post("/oldchat/{created_session_id}")
def chat(user_query: QueryInput, created_session_id: str = Path(..., description="Add previous session_id", example="27afee60-c4b9-47a3-aa71-7f229b9e3a77")):
    
    previous_qna = builder.get_session_messages(created_session_id)
    chat_history = ""
    for q, r in previous_qna:
        chat_history += f"User: {q}\nAssistant: {r}\n"

    query_with_context = chat_history + f"User: {user_query.text}\nAssistant:"
    result = router.run(query=query_with_context)

    builder.store_followup(
        question=user_query.text,
        response=result,
        session_id=created_session_id
    )

    return {
        "response": result,
        "previous session_id": created_session_id
    }


@app.post("/upload_csv/")
async def upload_csv(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    if not file.filename.endswith(".csv"):
        return {"error": "Only CSV files are supported."}
    
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    # Start vector DB and MySQL logic in background
    metadata = builder.process_dataframe(df, source=file.filename)
    background_tasks.add_task(builder.insert_into_mysql, metadata)

    return {
        "message": "CSV processed successfully (in background)",
        "filename": file.filename,
        "rows": len(df),
        "inserted_metadata": len(metadata)
    }