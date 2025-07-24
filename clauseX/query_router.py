import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.schema.output_parser import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnableBranch

load_dotenv()

class QueryRouter:
    def __init__(self, db_path="faiss_index", llm_name="models/gemini-2.0-flash", embedding_model="models/text-embedding-004"):

        google_api_key = os.getenv("GOOGLE_API_KEY")

        self.embedding = GoogleGenerativeAIEmbeddings(model=embedding_model)
    
        self.llm = ChatGoogleGenerativeAI(
        model=llm_name,
        temperature=0,
        max_tokens=None,
        google_api_key=google_api_key
    )

        self.parser = StrOutputParser()
        self.retriever = FAISS.load_local(
            db_path,
            self.embedding,
            allow_dangerous_deserialization=True
        ).as_retriever(
            search_type="mmr",
            search_kwargs={"k": 15, "fetch_k": 50}
        )

        self.definition_prompt = PromptTemplate(
            template="""
            You are a legal assistant. Define the legal term in the query using only the provided context.

            Respond directly — do not say "Based on the context" or use passive or indirect phrasing.

            Query: {user_query}

            Context:
            {context}
            """,
                input_variables=["user_query", "context"]
        )


        self.clause_prompt = PromptTemplate(
        template="""
            You are a legal assistant. Extract and explain the clause mentioned in the query using only the text from the provided context.

            Do NOT begin your response with phrases like "Based on the context" or "It appears that". Respond clearly and directly.

            Query: {user_query}

            Context:
            {context}
            """,
                input_variables=["user_query", "context"]
        )


        self.comparison_prompt = PromptTemplate(
         template="""
            You are a legal assistant. Compare the legal clauses mentioned in the query using only the content from the provided context.

            Write a clear comparison. Do NOT use hedging phrases like "Based on the provided context". If any clause is missing, state it plainly.

            Query: {user_query}

            Context:
            {context}
            """,
                input_variables=["user_query", "context"]
        )
        self.unknown_intent_prompt = PromptTemplate(
            input_variables=["user_query"],
            template="""
        The query you asked cannot be classified into a known legal intent.
        Query: "{user_query}"

        We support only:
        1. DefinitionQuery (ask for meaning)
        2. ClauseRetrieval (ask for a clause)
        3. ComparativeAnalysis (compare legal clauses)

        Please rephrase your question accordingly.
        """)


        self.branch_chain = RunnableBranch(
            (lambda x: x["intent"] == "DefinitionQuery", self.definition_prompt | self.llm | self.parser),
            (lambda x: x["intent"] == "ClauseRetrieval", self.clause_prompt | self.llm | self.parser),
            (lambda x: x["intent"] == "ComparativeAnalysis", self.comparison_prompt | self.llm | self.parser),
            self.unknown_intent_prompt | self.llm | self.parser
        )
        
        self.final_chain = RunnableLambda(
        lambda x: {
            "user_query": x["user_query"],
            "intent": self.ClassifyQuery(x["user_query"]),
            "context": "\n\n".join(
                [doc.page_content for doc in self.retriever.invoke(x["user_query"])]
            ),
        }
) | self.branch_chain
        
    def ClassifyQuery(self, query):

        prompt = PromptTemplate(
        input_variables=["user_query"],
        template="""
            You are a legal AI assistant that classifies user queries based on their intent.

            Your task is to analyze any kind of user query — whether short, long, or complex — and classify it into **exactly one** of the following three categories:

            1. DefinitionQuery: If the user is asking for the meaning, explanation, or definition of a legal term or concept.
            2. ClauseRetrieval: If the user is asking to find, show, or understand a specific clause from a contract or document.
            3. ComparativeAnalysis: If the user is asking to compare clauses or legal elements between two or more documents.

            Return only one of these three words exactly: `DefinitionQuery`, `ClauseRetrieval`, or `ComparativeAnalysis`.  
            Do not provide any explanation.

            ---

            User Query: "{user_query}"

            Answer:

    """
    )
        
        intent_classifier = prompt | self.llm | self.parser
        return intent_classifier.invoke({"user_query": query})
        
    def run(self, query: str):
        return self.final_chain.invoke({"user_query": query})
    
if __name__ == "__main__":
    router = QueryRouter()

    message = "Show me the confidentiality clause from the agreement."
    response = router.run(query= message)
    print(response)

