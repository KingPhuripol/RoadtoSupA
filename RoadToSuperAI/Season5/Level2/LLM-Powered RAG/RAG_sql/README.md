# 📘 Detailed Explanation of the RAG_sql.ipynb Notebook

# Build a Question/Answering system over SQL data

![RAG with Structured Data](https://python.langchain.com/v0.2/assets/images/sql_usecase-d432701261f05ab69b38576093718cf3.png)

## Setup

```python
## install Library
!pip -q install pypdf sentence-transformers langchain langchain_community jq faiss-cpu tiktoken
!pip -q install groq
```

```python
# prompt: download folder with gdown

!gdown --folder 1MGp7wWcJcduMm4nNmjzvN4egxxqKVq1N
```

### install postgresql

```python
!apt-get -y install postgresql postgresql-contrib
```

### Start PostgreSQL service & create uset & database

### register postgresql cloud free for 5 GB


https://console.aiven.io

```python
# Start PostgreSQL service
!service postgresql start

# Set password for the 'postgres' user
!sudo -u postgres psql -c "ALTER USER postgres PASSWORD '1234';"

# Create a new database
!sudo -u postgres createdb mydatabase

```

```python
!pip -q install psycopg2-binary sqlalchemy

```

### Transform csv to DB

```python
import pandas as pd
from sqlalchemy import create_engine
from google.colab import userdata

# Load CSV files into pandas DataFrames
financial_statements_df = pd.read_csv('/content/dataset/CSV/Financial Statements.csv')
online_shopping_df = pd.read_csv('/content/dataset/CSV/TBL3-Online-Shopping-Dataset.csv')
customer_support_tickets_df = pd.read_csv('/content/dataset/CSV/customer_support_tickets.csv')
spotify_data_df = pd.read_csv('/content/dataset/CSV/spotify-2023.csv', encoding='ISO-8859-1')

# Database connection parameters for local PostgreSQL
db_name = 'mydatabase'           # Your database name
db_user = 'postgres'             # PostgreSQL default superuser
db_password = '1234'             # The password you set above
db_host = 'localhost'            # Localhost for local connection
db_port = '5432'                 # Default PostgreSQL port

# Create a connection string
connection_string = f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
# connection_string =  userdata.get('AIVEN_URL')
# Create a SQLAlchemy engine
engine = create_engine(connection_string)

# Insert DataFrames into PostgreSQL tables
financial_statements_df.to_sql('financial_statements', engine, if_exists='replace', index=False)
online_shopping_df.to_sql('online_shopping', engine, if_exists='replace', index=False)
customer_support_tickets_df.to_sql('customer_support_tickets', engine, if_exists='replace', index=False)
spotify_data_df.to_sql('spotify_data', engine, if_exists='replace', index=False)

```

```python
## list Table in DB
pd.read_sql("""
            SELECT schemaname, tablename, tableowner
            FROM pg_catalog.pg_tables
            WHERE schemaname != 'pg_catalog' AND schemaname!='tiger' AND
            schemaname != 'information_schema';
            """,
            engine
            )
```

## langchain

### Download Database with SQLDatabase (langchain)

```python
from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri(connection_string)
print(db.dialect)
print(db.get_usable_table_names())

```

```python
print(db.get_table_info())
```

```python
print(db.get_table_info(['online_shopping']))
```

```python
online_shopping_df
```

test sql query

```python
db.run("SELECT * FROM online_shopping LIMIT 5;")
```

### Using groq via Langchain

**login**
https://console.groq.com/home

**model**
https://console.groq.com/docs/models

```python
!pip -q install -U langchain-groq
```

```python
from langchain_groq import ChatGroq
from google.colab import userdata

llm = ChatGroq(
    api_key = userdata.get('GROQ_API'),
    model= "meta-llama/llama-4-scout-17b-16e-instruct",  # meta-llama/llama-4-maverick-17b-128e-instruct    #"llama-3.1-70b-versatile"
    temperature=0,
    # other params...
)


```

call llm Example

```python
## Example
messages = [
    ("system", "You are a helpful translator. Translate the usersentence to Thai."),
    ("human", "I love programming."),
]
llm.invoke(messages)
```

### Creat sql Chain

```python
from langchain.chains import create_sql_query_chain

chain = create_sql_query_chain(llm, db)
print(chain.get_prompts()[0].pretty_print())

```

#### Step 1 SQLQuery from LLM

```python
Quesion = "มี Ticket_ID จำนวนเท่าไหร่ ที่ Date_of_Purchase ตั้งแต่ 2020-02-01 ถึง 2020-02-28"
response = chain.invoke({"question": Quesion})
response
```

```python
print(response.split('SQLQuery:')[-1][7:-3])
```

#### Step 2 SQLResult from sqlquery

```python
SQL_Result=db.run(response.split('SQLQuery:')[-1][7:-3])
SQL_Result
```

#### Step 3 get Answer from LLM

```python
answer_prompt =f"""
given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {Quesion}
SQL Query: {response.split('SQLQuery:')[-1]}
SQL Result: {SQL_Result}
Answer:
"""
messages = [
    ("system", answer_prompt),
]
answer = llm.invoke(messages)
answer
```

```python
answer.content
```

### Combine chain in one process

```python
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

execute_query = QuerySQLDataBaseTool(db=db)
write_query = create_sql_query_chain(llm, db)
chain = write_query | execute_query
chain.invoke({"question": Quesion})
```

#### Change prompt template for Llama4

```python
from langchain_core.prompts import PromptTemplate

template = '''You are a PostgreSQL expert. Given an input question, first create a syntactically correct PostgreSQL query to run, then look at the results of the query and return the answer to the input question.
Unless the user specifies in the question a specific number of examples to obtain, query for at most 5 results using the LIMIT clause as per PostgreSQL. You can order the results to return the most informative data in the database.
Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
Pay attention to use CURRENT_DATE function to get the current date, if the question involves "today".

Use the following format:

Question: Question here
SQLQuery: SQL Query to run

Answer only SQLQuery
Example :
    - SELECT COUNT(*) FROM head WHERE age > 56
    - SELECT MAX(budget_in_billions), MIN(budget_in_billions) FROM department
    - SELECT AVG(num_employees) FROM department WHERE ranking BETWEEN 10 AND 15

Only use the following tables:
{table_info}

Question: {input}'''
prompt = PromptTemplate.from_template(template, partial_variables={
    "table_info": db.get_table_info(),
    "top_k": 5,  # Default value for top_k
})
```

```python
chain = create_sql_query_chain(llm, db,prompt=prompt)
print(chain.get_prompts()[0].pretty_print())
```

```python
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

execute_query = QuerySQLDataBaseTool(db=db)
write_query = create_sql_query_chain(llm, db,prompt=prompt)
chain = write_query | execute_query
chain.invoke({"question": Quesion})
```

#### QA system over SQL data

```python
from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough


# Functions to print the intermediate results at each step
def print_step(label, data):
    print(f"\033[92mStep: {label}\033[0m , \033[94mOutput: {data}\033[0m")
    return data

answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer only the user question (don't explain).

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)


chain = (
    RunnablePassthrough.assign(query=write_query).assign(
        result=itemgetter("query") | execute_query
    )
    # | answer_prompt
    | (lambda inputs: print_step("Format Answer Prompt", answer_prompt.format(question=inputs["question"], query=inputs["query"], result=inputs["result"])))
    | llm
    | StrOutputParser()
)

print(chain.invoke({"question": Quesion}))

```

```python
print(chain.invoke({"question": Quesion},))
```

```python
print(chain.invoke({"question": "บริษัท AAPL จัดอยู่ใน Category อะไร ในปี 2022 "}))
```

```python
print(chain.invoke({"question": "ผลรวมกำไรขั้นต้นของบริษัท AAPL 5 ปีย้อนหลัง (ตั้งแต่ปี 2019-2023) มากกว่า ผลรวมกำไรขั้นต้นของบริษัท AMZN 5 ปีย้อนหลัง (ตั้งแต่ปี 2019-2023) หรือไม่ กรุณาตอบเป็น 'True' หรือ 'False'"}))
```

```python
print(chain.invoke({"question": "CustomerID ที่มียอดจำนวนเงินสั่งซื้อไม่รวมค่าขนส่งมากที่สุดมาจากที่เมืองไหน"}))
```

```python
print(chain.invoke({"question": "แนะนำชื่อเพลง Top5 ปีล่าสุด"}))
```

```python
print(chain.invoke({"question": "มี Customer_Email จำนวนเท่าไหร่ ที่ลงท้ายด้วย 'example.org"}))
```

### SQL Agents

```python
!pip -q install --upgrade --quiet langgraph
```

```python
from langchain_community.agent_toolkits import SQLDatabaseToolkit

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

tools = toolkit.get_tools()

tools
```

```python
from langchain_core.messages import SystemMessage

SQL_PREFIX = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct SQLite query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

To start you should ALWAYS look at the tables in the database to see what you can query.
Do NOT skip this step.
Then you should query the schema of the most relevant tables."""

system_message = SystemMessage(content=SQL_PREFIX)
```

```python
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

agent_executor = create_react_agent(llm, tools, state_modifier=system_message)
```

```python
for s in agent_executor.stream(
    {"messages": [HumanMessage(content="Ticket_Priority อะไรมีจำนวนมากที่สุด 3 อันดับ (ตอบในรูปแบบของ list โดยมีแต่ล่ะ element มี format คือ {'Ticket_Priority': ชื่อ Ticket_Priority , 'Count': จำนวน} และ เรียงลำดับจากจำนวนมากไปน้อย) ที่มี Ticket_Channel เท่ากับ 'Chat'")]}
):
    print(s)
    print("----")
```

```python
s['agent']['messages'][0].content
```

```python
for s in agent_executor.stream(
    {"messages": [HumanMessage(content="มี Customer_Email จำนวนเท่าไหร่ ที่ลงท้ายด้วย 'example.org'")]}
):
    print(s)
    print("----")
```

```python
s['agent']['messages'][0].content
```

