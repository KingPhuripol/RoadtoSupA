# การวิเคราะห์โค้ด RAG with SQL

โค้ดนี้แสดงการสร้างระบบถาม-ตอบ (Question Answering) ที่ทำงานกับข้อมูลในฐานข้อมูล SQL โดยใช้เทคนิค Retrieval-Augmented Generation (RAG) ซึ่งประกอบด้วยขั้นตอนหลักดังนี้:

## 1. การตั้งค่าเบื้องต้น (Setup)

### การติดตั้งไลบรารีที่จำเป็น
```python
!pip install pypdf sentence-transformers langchain langchain_community jq faiss-cpu tiktoken groq
```
- ติดตั้งแพ็คเกจต่างๆ ที่จำเป็น เช่น LangChain สำหรับสร้างระบบถาม-ตอบ, FAISS สำหรับการค้นหาข้อมูล, และ Groq สำหรับใช้งานโมเดลภาษา

### การดาวน์โหลดข้อมูล
```python
!gdown --folder 1MGp7wWcJcduMm4nNmjzvN4egxxqKVq1N
```
- ดาวน์โหลดชุดข้อมูล CSV จาก Google Drive

## 2. การตั้งค่า PostgreSQL

### การติดตั้ง PostgreSQL
```python
!apt-get -y install postgresql postgresql-contrib
```
- ติดตั้ง PostgreSQL และส่วนเสริมที่จำเป็น

### การเริ่มต้นบริการ PostgreSQL
```python
!service postgresql start
!sudo -u postgres psql -c "ALTER USER postgres PASSWORD '1234';"
!sudo -u postgres createdb mydatabase
```
- เริ่มบริการ PostgreSQL
- ตั้งรหัสผ่านสำหรับผู้ใช้ postgres
- สร้างฐานข้อมูลชื่อ mydatabase

## 3. การโหลดข้อมูล CSV เข้าสู่ฐานข้อมูล

### การแปลง CSV เป็นตารางใน PostgreSQL
```python
import pandas as pd
from sqlalchemy import create_engine

# โหลดข้อมูลจากไฟล์ CSV
financial_statements_df = pd.read_csv('/content/dataset/CSV/Financial Statements.csv')
online_shopping_df = pd.read_csv('/content/dataset/CSV/TBL3-Online-Shopping-Dataset.csv')
customer_support_tickets_df = pd.read_csv('/content/dataset/CSV/customer_support_tickets.csv')
spotify_data_df = pd.read_csv('/content/dataset/CSV/spotify-2023.csv', encoding='ISO-8859-1')

# เชื่อมต่อกับฐานข้อมูล PostgreSQL
connection_string = f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
engine = create_engine(connection_string)

# นำข้อมูลเข้า PostgreSQL
financial_statements_df.to_sql('financial_statements', engine, if_exists='replace', index=False)
online_shopping_df.to_sql('online_shopping', engine, if_exists='replace', index=False)
customer_support_tickets_df.to_sql('customer_support_tickets', engine, if_exists='replace', index=False)
spotify_data_df.to_sql('spotify_data', engine, if_exists='replace', index=False)
```
- ใช้ Pandas โหลดข้อมูลจากไฟล์ CSV
- สร้างการเชื่อมต่อกับ PostgreSQL โดยใช้ SQLAlchemy
- นำข้อมูลจาก DataFrame เข้าสู่ตารางใน PostgreSQL

## 4. การใช้ LangChain ทำงานกับฐานข้อมูล SQL

### การเชื่อมต่อกับฐานข้อมูลผ่าน LangChain
```python
from langchain_community.utilities import SQLDatabase
db = SQLDatabase.from_uri(connection_string)
```
- สร้างออบเจ็กต์ SQLDatabase ของ LangChain เพื่อทำงานกับฐานข้อมูล

### การตรวจสอบโครงสร้างฐานข้อมูล
```python
print(db.dialect)  # ประเภทฐานข้อมูล (PostgreSQL)
print(db.get_usable_table_names())  # รายชื่อตาราง
print(db.get_table_info())  # โครงสร้างตารางทั้งหมด
print(db.get_table_info(['online_shopping']))  # โครงสร้างตารางเฉพาะ
```
- ตรวจสอบข้อมูลเกี่ยวกับฐานข้อมูลและตารางต่างๆ

## 5. การใช้งานโมเดลภาษา Groq

### การตั้งค่าโมเดลภาษา
```python
from langchain_groq import ChatGroq
from google.colab import userdata

llm = ChatGroq(
    api_key = userdata.get('GROQ_API'),
    model= "meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0,
)
```
- สร้างออบเจ็กต์ ChatGroq เพื่อใช้งานโมเดลภาษา Llama
- ตั้งค่าโมเดลและพารามิเตอร์ต่างๆ

## 6. การสร้าง SQL Query Chain

### การสร้างเชนสำหรับสร้างคำสั่ง SQL
```python
from langchain.chains import create_sql_query_chain
chain = create_sql_query_chain(llm, db)
```
- สร้างเชนที่ใช้โมเดลภาษาเพื่อแปลงคำถามเป็นคำสั่ง SQL

### การทดสอบสร้างคำสั่ง SQL
```python
Quesion = "มี Ticket_ID จำนวนเท่าไหร่ ที่ Date_of_Purchase ตั้งแต่ 2020-02-01 ถึง 2020-02-28"
response = chain.invoke({"question": Quesion})
```
- ทดสอบถามคำถามและดูคำสั่ง SQL ที่สร้างขึ้น

## 7. การดำเนินการและประมวลผลผลลัพธ์

### การดำเนินการคำสั่ง SQL
```python
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
execute_query = QuerySQLDataBaseTool(db=db)
SQL_Result = execute_query(response.split('SQLQuery:')[-1][7:-3])
```
- ใช้เครื่องมือ QuerySQLDataBaseTool เพื่อดำเนินการคำสั่ง SQL

### การสร้างคำตอบจากผลลัพธ์
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
```
- ใช้โมเดลภาษาเพื่อสร้างคำตอบจากผลลัพธ์ที่ได้จากฐานข้อมูล

## 8. การรวมขั้นตอนเป็นระบบถาม-ตอบแบบครบวงจร

### การสร้างเชนแบบ end-to-end
```python
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

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
    | answer_prompt
    | llm
    | StrOutputParser()
)
```
- สร้างเชนที่รวมทุกขั้นตอนตั้งแต่รับคำถาม สร้างคำสั่ง SQL ดำเนินการคำสั่ง และสร้างคำตอบ

### การทดสอบระบบ
```python
print(chain.invoke({"question": "บริษัท AAPL จัดอยู่ใน Category อะไร ในปี 2022"}))
print(chain.invoke({"question": "CustomerID ที่มียอดจำนวนเงินสั่งซื้อไม่รวมค่าขนส่งมากที่สุดมาจากที่เมืองไหน"}))
```
- ทดสอบระบบด้วยคำถามต่างๆ

## 9. การสร้าง SQL Agent

### การตั้งค่า SQL Agent
```python
from langchain_community.agent_toolkits import SQLDatabaseToolkit
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()
```
- สร้างชุดเครื่องมือสำหรับทำงานกับฐานข้อมูล SQL

### การกำหนดข้อความระบบ
```python
from langchain_core.messages import SystemMessage
SQL_PREFIX = """You are an agent designed to interact with a SQL database..."""
system_message = SystemMessage(content=SQL_PREFIX)
```
- กำหนดบทบาทและข้อแนะนำสำหรับ Agent

### การสร้าง Agent Executor
```python
from langgraph.prebuilt import create_react_agent
agent_executor = create_react_agent(llm, tools, state_modifier=system_message)
```
- สร้าง Agent ที่สามารถโต้ตอบกับฐานข้อมูล SQL

### การทดสอบ Agent
```python
for s in agent_executor.stream(
    {"messages": [HumanMessage(content="Ticket_Priority อะไรมีจำนวนมากที่สุด 3 อันดับ...")]}
):
    print(s)
```
- ทดสอบ Agent ด้วยคำถามที่ซับซ้อนมากขึ้น

## สรุปกระบวนการทำงาน

1. **การเตรียมข้อมูล**: นำเข้าข้อมูลจากไฟล์ CSV เข้าสู่ฐานข้อมูล PostgreSQL
2. **การเชื่อมต่อฐานข้อมูล**: ใช้ LangChain เพื่อเชื่อมต่อและทำงานกับฐานข้อมูล
3. **การสร้างคำสั่ง SQL**: ใช้โมเดลภาษาแปลงคำถามเป็นคำสั่ง SQL
4. **การดำเนินการคำสั่ง**: ดำเนินการคำสั่ง SQL และรับผลลัพธ์
5. **การสร้างคำตอบ**: ใช้โมเดลภาษาแปลงผลลัพธ์จากฐานข้อมูลเป็นคำตอบที่มนุษย์เข้าใจ
6. **การสร้าง Agent**: พัฒนา Agent ที่สามารถโต้ตอบและแก้ไขปัญหาที่ซับซ้อนด้วยตัวเอง

ระบบนี้แสดงให้เห็นถึงพลังของ RAG เมื่อทำงานร่วมกับข้อมูลที่มีโครงสร้างในฐานข้อมูล SQL โดยสามารถ:
- แปลงคำถามภาษาธรรมชาติเป็นคำสั่ง SQL
- ดึงข้อมูลที่เกี่ยวข้องจากฐานข้อมูล
- สร้างคำตอบที่อ่านเข้าใจง่ายจากผลลัพธ์ที่ได้
