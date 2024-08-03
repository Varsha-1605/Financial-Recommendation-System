# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# import streamlit as st
# from faker import Faker
# import random
# from langchain_openai import ChatOpenAI
# from datetime import date
# from dateutil.relativedelta import relativedelta
# import pandas as pd
# from langchain.docstore.document import Document
# from langchain.vectorstores import SKLearnVectorStore
# from langchain_openai import OpenAIEmbeddings
# from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate
# import time
# from openai import OpenAIError
# from openai import APIConnectionError
# from requests.exceptions import ConnectionError
# import os
# import traceback
# import sys

# openai_api_key = st.secrets["OPENAI_API_KEY"]

# from langchain.globals import set_verbose
# # Set verbosity level to True to enable detailed logging
# set_verbose(True)

# st.set_page_config(page_title='MarketWealth Recommender', layout='wide')

# # Custom CSS for hover effects
# st.markdown("""
#     <style>
#     .stTextInput>div>div>input:hover {
#         border-color: #ff4b4b;
#         box-shadow: 0 0 10px rgba(255, 75, 75, 0.5);
#     }
#     .stTextInput>div>div>textarea:hover {
#         border-color: #ff4b4b;
#         box-shadow: 0 0 10px rgba(255, 75, 75, 0.5);
#     }
#     .stTextInput>div>div>input:focus {
#         border-color: #ff4b4b;
#         box-shadow: 0 0 10px rgba(255, 75, 75, 0.5);
#     }
#     .stTextInput>div>div>textarea:focus {
#         border-color: #ff4b4b;
#         box-shadow: 0 0 10px rgba(255, 75, 75, 0.5);
#     }
#     </style>
# """, unsafe_allow_html=True)

# def global_exception_handler(exctype, value, tb):
#     st.error("An unexpected error occurred. Our team has been notified. 🤖")
#     # Log the error (in a production environment, you'd want to use a proper logging system)
#     print("Uncaught exception:", file=sys.stderr)
#     print("Type:", exctype, file=sys.stderr)
#     print("Value:", value, file=sys.stderr)
#     print("Traceback:", file=sys.stderr)
#     traceback.print_tb(tb)

# sys.excepthook = global_exception_handler

# # UI Setup

# st.markdown("""
#     <style>
#     .custom-title {
#         font-size: 65px;
#         font-weight: bold;
#         color: #ff4b4b;
#         text-align: center;
#         text-decoration: underline;
#         text-decoration-color: #FFFDD0;
#         text-decoration-thickness: 2px;
#         text-align: center;
#         margin-top: 5px;
#         margin-bottom: 20px;
#     }
#     .custom-icon {
#         font-size: 60px;
#         color: #ff4b4b;
#     }
#     </style>
#     <div class="custom-title">
#         MarketWealth Recommender <span class="custom-icon">💼</span>
#     </div>
# """, unsafe_allow_html=True)

# st.markdown("""
#     <style>
#     .welcome-title {
#         font-size: 50px;
#         font-weight: bold;
#         color: #1E90FF;
#         text-align: center;
#         margin-bottom: 10px;
#     }
#     .welcome-text {
#         font-size: 1.25rem;
#         color: #333333;
#         text-align: center;
#         line-height: 1.6;
#         margin-top: 0;
#         margin-bottom: 20px;
#     }
#     .wave-icon {
#         font-size: 35px;
#         color: #ff4b4b;
#     }
#     </style>
#     <div class="welcome-title">
#         Welcome! <span class="wave-icon">👋</span>
#     </div>
#     <div class="welcome-text">
#         Unlock personalized banking and financial recommendations effortlessly. 
#         Simply type your query—be it credit cards, investments, mortgages, or 
#         financial health—and receive advice finely tuned to your profile and goals.
#     </div>
# """, unsafe_allow_html=True)

# # Tutorial Section
# st.sidebar.markdown("""
#     <h1 style="font-size:40px; color:#FFD700;">How to Use 🛠️</h1>
#     <p>1. Enter a question about banking products or financial advice.</p>
#     <p>2. Click on 'Get Recommendation' to receive personalized advice.</p>
#     <p>3. Provide feedback to help us improve.</p>
# """, unsafe_allow_html=True)

# st.sidebar.markdown('<hr style="border:2px solid #ff4b4b;">', unsafe_allow_html=True)

# # Feedback Section
# st.sidebar.markdown(f"""
#     <h1 style="font-size:40px; color:#FFD700;">Feedback 📝</h1>
#     <p>We value your feedback! 😊</p>
# """, unsafe_allow_html=True)

# feedback = st.sidebar.text_area("Please leave your feedback here:")
# if st.sidebar.button("Submit Feedback"):
#     st.sidebar.success("Thank you for your feedback! 👍")

# st.sidebar.markdown('<hr style="border:2px solid #ff4b4b;">', unsafe_allow_html=True)

# # Sample Questions
# st.sidebar.markdown(f"""
#     <h1 style="font-size:40px; color:#FFD700;">Sample Questions ❓</h1>
#     <p>Here are some sample questions you can try:</p>
#     <ol>
#         <li>Can you recommend a credit card for someone with a low credit score? 💳</li>
#         <li>What investment options would you recommend for a 45-year-old customer looking to save for retirement in 15 years? elaborate it. 👵💼</li>
#         <li>What mortgage options are available for first-time homebuyers?</li>
#         <li>What banking products would you recommend for a 35-year-old customer with a high income and a credit score of 800? 💰</li>
#         <li>What investment products are suitable for a risk-averse retiree? 🛡️</li>
#         <li>What are the best options for debt consolidation? 💳</li>
#         <li>How can a 40-year-old customer with a good credit score and some existing debt improve their credit profile? give a brief about it. 📊</li>
#         <li>What credit card would you recommend for someone who travels frequently? ✈️</li>
#     </ol>
# """, unsafe_allow_html=True)

# fake = Faker()

# # Generate demographic and personal information
# def generate_customer_data():
#     age = random.randint(20, 70)
#     gender = random.choice(['Male', 'Female'])
#     marital_status = random.choice(['Single', 'Married', 'Divorced', 'Widowed'])
#     income_level = random.choice(['Low', 'Medium', 'High'])
#     education = random.choice(['High School', 'College', 'University'])
#     occupation = fake.job()
#     residential_status = random.choice(['Owns house', 'Rents', 'Living with parents'])
#     dependents = random.randint(0, 5)
#     debt_to_income = round(random.uniform(0.1, 0.5), 2)
#     credit_bureau = random.randint(760, 850)

#     return {
#         'Age': age,
#         'Gender': gender,
#         'Marital Status': marital_status,
#         'Income Level': income_level,
#         'Education': education,
#         'Occupation': occupation,
#         'Residential Status': residential_status,
#         'Dependents': dependents,
#         'Debt-to-Income': debt_to_income,
#         'Credit_Bureau': credit_bureau
#     }

# # Function to generate bureau product inquiries
# def generate_inquiries(last_months):
#     inquiries = []
#     today = fake.date_this_month()

#     for _ in range(random.randint(1, 5)):
#         inquiry_date = fake.date_between(start_date=last_months, end_date=today)
#         product_type = random.choice(['Personal Loan', 'Credit Card', 'Mortgage'])
#         inquiries.append({'product_name': product_type, 'date': inquiry_date})

#     return inquiries

# # Generate dataset
# def generate_dataset(num_rows, months):
#     try:
#         data_rows = []

#         for _ in range(num_rows):
#             customer_data = generate_customer_data()
#             last_3_months_inquiries = generate_inquiries(months[0])
#             last_6_months_inquiries = generate_inquiries(months[1])

#             customer_row = {
#                 'Customer ID': fake.uuid4(),
#                 'Age': customer_data['Age'],
#                 'Gender': customer_data['Gender'],
#                 'Marital Status': customer_data['Marital Status'],
#                 'Income Level': customer_data['Income Level'],
#                 'Education': customer_data['Education'],
#                 'Occupation': customer_data['Occupation'],
#                 'Residential Status': customer_data['Residential Status'],
#                 'Dependents': customer_data['Dependents'],
#                 'Debt-to-Income': customer_data['Debt-to-Income'],
#                 'Credit_Bureau': customer_data['Credit_Bureau']
#             }

#             for product_type in ['Personal Loan', 'Credit Card', 'Mortgage']:
#                 inq_in_last_3_months = any(inq['product_name'] == product_type for inq in last_3_months_inquiries)
#                 customer_row[f'last_3months_{product_type.replace(" ", "_").lower()}_inq'] = inq_in_last_3_months

#             for product_type in ['Personal Loan', 'Credit Card', 'Mortgage']:
#                 inq_in_last_6_months = any(inq['product_name'] == product_type for inq in last_6_months_inquiries)
#                 customer_row[f'last_6months_{product_type.replace(" ", "_").lower()}_inq'] = inq_in_last_6_months

#             data_rows.append(customer_row)
#         return data_rows
#     except Exception as e:
#         st.error(f"🔴 Error generating dataset: {str(e)}")
#         raise

# def setup_data_and_vectorstore():
#     months = [date.today() - relativedelta(months=+3), date.today() - relativedelta(months=+6)]
#     dataset = generate_dataset(50, months)

#     df = pd.DataFrame(dataset)

#     df['content'] = [f"Based on the following customer data: {data}, suggest suitable banking lending products." for data in dataset]

#     documents = []
#     for _, row in df.iterrows():
#         documents.append(Document(page_content=row["content"], metadata={"class": row["Age"]}))

#     openai_embeddings = OpenAIEmbeddings(api_key=openai_api_key)

#     max_retries = 5
#     retry_delay = 2  # seconds

#     for attempt in range(max_retries):
#         try:
#             sklearn_store = SKLearnVectorStore.from_documents(
#                 documents=documents,
#                 embedding=openai_embeddings
#             )
#             break
#         except (APIConnectionError, ConnectionError) as e:
#             if attempt < max_retries - 1:
#                 time.sleep(retry_delay)
#             else:
#                 st.error(f"🔴 Failed to connect after {max_retries} attempts. Please try again later. {str(e)}")
#                 raise e

#     return sklearn_store

# @st.cache_resource
# def get_vectorstore():
#     return setup_data_and_vectorstore()

# sklearn_store = get_vectorstore()

# def setup_retrieval_qa(sklearn_store):
#     try:
#         openai_api_key = st.secrets["OPENAI_API_KEY"]
#         if not openai_api_key:
#             raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        
#         openai_llm = ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo")

#         prompt_template = PromptTemplate(
#             input_variables=["context"],
#             template="Based on the following customer data: {context}, suggest suitable banking lending products in the following format:\n\n1. Product 1: Description\n2. Product 2: Description\n3. Product 3: Description\nProvide detailed recommendations."
#         )

#         retrieval_qa = RetrievalQA.from_chain_type(
#             llm=openai_llm,
#             chain_type="stuff",
#             retriever=sklearn_store.as_retriever()
#         )

#         return retrieval_qa
#     except ValueError as e:
#         st.error(f":x: {str(e)}")
#         raise
#     except Exception as e:
#         st.error(f":x: Error setting up retrieval QA: {str(e)}")
#         raise

# retrieval_qa = setup_retrieval_qa(sklearn_store)

# st.markdown("""
#     <style>
#     .custom-input-label {
#         font-size: 30px;
#         font-weight: bold;
#         color: #FFD700;
#         margin-bottom: 2px;
#         display: block;
#     }
#     .custom-input {
#         width: 100%;
#         padding: 10px;
#         font-size: 1rem;
#         border: 2px solid #ccc;
#         border-radius: 4px;
#         transition: border-color 0.3s ease;
#     }
#     .custom-input:focus {
#         border-color: #ff4b4b;
#         outline: none;
#     }
#     </style>
#     <label class="custom-input-label">Enter your query:</label>
    
# """, unsafe_allow_html=True)

# # Use JavaScript to capture the input value
# question = st.text_input('-----------------------------')

# if not question:
#     st.write(":mag_right: Please enter a question to get a response.")

# if st.button("Get Recommendation"):
#     with st.spinner(":hourglass_flowing_sand: Generating recommendation..."):
#         try:
#             response = retrieval_qa.invoke(question)
#             st.write(f"**Answer:** \n{response['result']}")

#         except OpenAIError as e:
#             st.error(f":x: An OpenAI error occurred: {str(e)}")
#             st.info(":information_source: Please try again later or contact support if the problem persists.")

#         except ValueError as e:
#             st.error(f":x: Input error: {str(e)}")
#             st.info(":information_source: Please check your input and try again.")

#         except Exception as e:
#             st.error(f":x: An unexpected error occurred: {str(e)}")
#             st.info(":information_source: Please try again or contact support if the problem persists.")

#             # Log the full error for debugging
#             print(traceback.format_exc())

# # Footer
# st.markdown("---")
# st.markdown('<p style="text-align: center;">Created with ❤️ using Streamlit and LangChain</p>', unsafe_allow_html=True)






































import streamlit as st
import pandas as pd
from faker import Faker
import random
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from datetime import date
from dateutil.relativedelta import relativedelta
from langchain.docstore.document import Document
from langchain.vectorstores import SKLearnVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import time
from openai import OpenAIError, APIConnectionError
from requests.exceptions import ConnectionError
import traceback
import sys
import plotly.graph_objects as go

import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
# import pandas as pd
from datetime import datetime, timedelta
# import streamlit as st
# import plotly.graph_objects as go




# Configuration
st.set_page_config(page_title='MarketWealth AI', layout='wide', initial_sidebar_state='collapsed')
openai_api_key = st.secrets["OPENAI_API_KEY"]

from langchain.globals import set_verbose
set_verbose(True)

# Global exception handler (unchanged)
def global_exception_handler(exctype, value, tb):
    st.error("An unexpected error occurred. Our team has been notified. 🤖")
    print("Uncaught exception:", file=sys.stderr)
    print("Type:", exctype, file=sys.stderr)
    print("Value:", value, file=sys.stderr)
    print("Traceback:", file=sys.stderr)
    traceback.print_tb(tb)

sys.excepthook = global_exception_handler

# Data generation functions (unchanged)
fake = Faker()

def generate_customer_data():
    # ... (unchanged)
    age = random.randint(20, 70)
    gender = random.choice(['Male', 'Female'])
    marital_status = random.choice(['Single', 'Married', 'Divorced', 'Widowed'])
    income_level = random.choice(['Low', 'Medium', 'High'])
    education = random.choice(['High School', 'College', 'University'])
    occupation = fake.job()
    residential_status = random.choice(['Owns house', 'Rents', 'Living with parents'])
    dependents = random.randint(0, 5)
    debt_to_income = round(random.uniform(0.1, 0.5), 2)
    credit_bureau = random.randint(760, 850)

    return {
        'Age': age,
        'Gender': gender,
        'Marital Status': marital_status,
        'Income Level': income_level,
        'Education': education,
        'Occupation': occupation,
        'Residential Status': residential_status,
        'Dependents': dependents,
        'Debt-to-Income': debt_to_income,
        'Credit_Bureau': credit_bureau
    }

def generate_inquiries(last_months):
    # ... (unchanged)
    inquiries = []
    today = fake.date_this_month()

    for _ in range(random.randint(1, 5)):
        inquiry_date = fake.date_between(start_date=last_months, end_date=today)
        product_type = random.choice(['Personal Loan', 'Credit Card', 'Mortgage'])
        inquiries.append({'product_name': product_type, 'date': inquiry_date})

    return inquiries

def generate_dataset(num_rows, months):
    # ... (unchanged)
    try:
        data_rows = []

        for _ in range(num_rows):
            customer_data = generate_customer_data()
            last_3_months_inquiries = generate_inquiries(months[0])
            last_6_months_inquiries = generate_inquiries(months[1])

            customer_row = {
                'Customer ID': fake.uuid4(),
                'Age': customer_data['Age'],
                'Gender': customer_data['Gender'],
                'Marital Status': customer_data['Marital Status'],
                'Income Level': customer_data['Income Level'],
                'Education': customer_data['Education'],
                'Occupation': customer_data['Occupation'],
                'Residential Status': customer_data['Residential Status'],
                'Dependents': customer_data['Dependents'],
                'Debt-to-Income': customer_data['Debt-to-Income'],
                'Credit_Bureau': customer_data['Credit_Bureau']
            }

            for product_type in ['Personal Loan', 'Credit Card', 'Mortgage']:
                inq_in_last_3_months = any(inq['product_name'] == product_type for inq in last_3_months_inquiries)
                customer_row[f'last_3months_{product_type.replace(" ", "_").lower()}_inq'] = inq_in_last_3_months

            for product_type in ['Personal Loan', 'Credit Card', 'Mortgage']:
                inq_in_last_6_months = any(inq['product_name'] == product_type for inq in last_6_months_inquiries)
                customer_row[f'last_6months_{product_type.replace(" ", "_").lower()}_inq'] = inq_in_last_6_months

            data_rows.append(customer_row)
        return data_rows
    except Exception as e:
        st.error(f"🔴 Error generating dataset: {str(e)}")
        raise

def setup_data_and_vectorstore():
    # ... (unchanged)
    months = [date.today() - relativedelta(months=+3), date.today() - relativedelta(months=+6)]
    dataset = generate_dataset(50, months)

    df = pd.DataFrame(dataset)

    df['content'] = [f"Based on the following customer data: {data}, suggest suitable banking lending products." for data in dataset]

    documents = []
    for _, row in df.iterrows():
        documents.append(Document(page_content=row["content"], metadata={"class": row["Age"]}))

    openai_embeddings = OpenAIEmbeddings(api_key=openai_api_key)

    max_retries = 5
    retry_delay = 2  # seconds

    for attempt in range(max_retries):
        try:
            sklearn_store = SKLearnVectorStore.from_documents(
                documents=documents,
                embedding=openai_embeddings
            )
            break
        except (APIConnectionError, ConnectionError) as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                st.error(f"🔴 Failed to connect after {max_retries} attempts. Please try again later. {str(e)}")
                raise e

    return sklearn_store


@st.cache_resource
def get_vectorstore():
    return setup_data_and_vectorstore()

def setup_retrieval_qa(sklearn_store):
    # ... (unchanged)
    try:
        openai_api_key = st.secrets["OPENAI_API_KEY"]
        if not openai_api_key:
            raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        
        openai_llm = ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo")

        prompt_template = PromptTemplate(
            input_variables=["context"],
            template="Based on the following customer data: {context}, suggest suitable banking lending products in the following format:\n\n1. Product 1: Description\n2. Product 2: Description\n3. Product 3: Description\nProvide detailed recommendations."
        )

        retrieval_qa = RetrievalQA.from_chain_type(
            llm=openai_llm,
            chain_type="stuff",
            retriever=sklearn_store.as_retriever()
        )

        return retrieval_qa
    except ValueError as e:
        st.error(f":x: {str(e)}")
        raise
    except Exception as e:
        st.error(f":x: Error setting up retrieval QA: {str(e)}")
        raise

# Setup the vector store and retrieval QA
sklearn_store = get_vectorstore()
retrieval_qa = setup_retrieval_qa(sklearn_store)

# Updated Cyberpunk-inspired CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700&family=Roboto:wght@300;400;700&display=swap');
    
    :root {
        --primary-color: #00F5FF;
        --secondary-color: #FF00E4;
        --accent-color: #FFD700;
        --bg-color: #0A0E17;
        --text-color: #E0E0E0;
        --card-bg: #141C2F;
    }
    
    body {
        color: var(--text-color);
        background-color: var(--bg-color);
        font-family: 'Roboto', sans-serif;
        background-image: 
            linear-gradient(45deg, rgba(0, 245, 255, 0.05) 25%, transparent 25%),
            linear-gradient(-45deg, rgba(255, 0, 228, 0.05) 25%, transparent 25%),
            linear-gradient(45deg, transparent 75%, rgba(0, 245, 255, 0.05) 75%),
            linear-gradient(-45deg, transparent 75%, rgba(255, 0, 228, 0.05) 75%);
        background-size: 20px 20px;
        background-position: 0 0, 10px 0, 10px -10px, 0px 10px;
        background-attachment: fixed;
    }
    
    .stApp {
        background: transparent;
    }
    
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif;
        color: var(--primary-color);
        text-shadow: 0 0 10px rgba(0, 245, 255, 0.5), 0 0 20px rgba(0, 245, 255, 0.3), 0 0 30px rgba(0, 245, 255, 0.2);
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    
    .stButton > button {
        font-family: 'Orbitron', sans-serif;
        background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
        color: var(--bg-color);
        font-weight: 700;
        border-radius: 30px;
        border: none;
        padding: 15px 30px;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(0, 245, 255, 0.4), 0 0 30px rgba(255, 0, 228, 0.2);
        text-transform: uppercase;
        letter-spacing: 2px;
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 8px 20px rgba(255, 0, 228, 0.6), 0 0 40px rgba(0, 245, 255, 0.3);
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transform: rotate(45deg);
        animation: shine 3s infinite;
    }
    
    @keyframes shine {
        0% { left: -50%; top: -50%; }
        100% { left: 150%; top: 150%; }
    }
    
    .stTextInput > div > div > input, 
    .stSelectbox > div > div > select, 
    .stTextArea > div > div > textarea {
        font-family: 'Roboto', sans-serif;
        background-color: var(--card-bg);
        color: var(--text-color);
        border-radius: 15px;
        border: 2px solid var(--primary-color);
        padding: 12px;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus, 
    .stSelectbox > div > div > select:focus, 
    .stTextArea > div > div > textarea:focus {
        border-color: var(--secondary-color);
        box-shadow: 0 0 15px rgba(255, 0, 228, 0.5), 0 0 30px rgba(0, 245, 255, 0.3);
    }
    
    .custom-card {
        background-color: var(--card-bg);
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 10px 30px rgba(0, 245, 255, 0.2);
        transition: all 0.3s ease;
        border: 2px solid transparent;
        background-clip: padding-box;
        margin-bottom: 20px;
    }
    
    .custom-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(255, 0, 228, 0.3), 0 0 40px rgba(0, 245, 255, 0.2);
        border-color: var(--secondary-color);
    }
    
    .neon-glow {
        animation: neon-glow 2s infinite;
    }
    
    @keyframes neon-glow {
        0% { box-shadow: 0 0 5px var(--primary-color), 0 0 10px var(--primary-color), 0 0 15px var(--primary-color), 0 0 20px var(--primary-color); }
        50% { box-shadow: 0 0 10px var(--primary-color), 0 0 20px var(--primary-color), 0 0 30px var(--primary-color), 0 0 40px var(--primary-color), 0 0 50px var(--secondary-color); }
        100% { box-shadow: 0 0 5px var(--primary-color), 0 0 10px var(--primary-color), 0 0 15px var(--primary-color), 0 0 20px var(--primary-color); }
    }
    
    .sidebar .sidebar-content {
        background-color: rgba(20, 28, 47, 0.8);
        backdrop-filter: blur(10px);
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-color);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(var(--primary-color), var(--secondary-color));
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(var(--secondary-color), var(--primary-color));
    }
    
    /* New styles for tabs */
    .stTabs {
        background-color: var(--card-bg);
        border-radius: 15px;
        padding: 10px;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: var(--bg-color);
        color: var(--text-color);
        border-radius: 10px;
        border: 1px solid var(--primary-color);
        padding: 10px 20px;
        font-family: 'Orbitron', sans-serif;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--primary-color);
        color: var(--bg-color);
    }
    
    /* New styles for tooltips */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: pointer;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: var(--card-bg);
        color: var(--text-color);
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
</style>
""", unsafe_allow_html=True)

# Main content
st.title("MarketWealth AI")

tabs = st.tabs(["AI Assistant", "Financial Dashboard", "Learn & Explore"])

with tabs[0]:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
            <div class="custom-card">
                <h2>Your Financial AI Assistant</h2>
                <p style="font-size: 1.2rem;">
                    Welcome to the future of financial advice. Our AI is here to provide you with 
                    personalized recommendations tailored to your unique financial situation.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<h3 style="color: var(--accent-color);">Ask Your Financial Question</h3>', unsafe_allow_html=True)
        question = st.text_input('', placeholder='E.g., What are the best investment options for a 30-year-old?')
        
        if st.button("Get AI Recommendation", key="main_button"):
            with st.spinner("💡 Analyzing financial data..."):
                try:
                    response = retrieval_qa.invoke(question)
                    st.markdown(f"""
                        <div class="custom-card neon-glow">
                            <h3>AI Recommendation</h3>
                            <p>{response['result']}</p>
                        </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

    with col2:
        st.markdown("""
            <div class="custom-card">
                <h3 style="color: var(--accent-color);">Popular Queries</h3>
                <ul>
                    <li>Can you recommend a credit card for someone with a low credit score? 💳</li>
                    <li>What investment options would you recommend for a 45-year-old customer looking to save for retirement in 15 years? elaborate it. 👵💼</li>
                    <li>What mortgage options are available for first-time homebuyers?</li>
                    <li>What banking products would you recommend for a 35-year-old customer with a high income and a credit score of 800? 💰</li>
                    <li>What investment products are suitable for a risk-averse retiree? 🛡️</li>
                    <li>What are the best options for debt consolidation? 💳</li>
                    <li>How can a 40-year-old customer with a good credit score and some existing debt improve their credit profile? give a brief about it. 📊</li>
                    <li>What credit card would you recommend for someone who travels frequently? ✈️</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p style="text-align: center;">Created with ❤️ using Streamlit and LangChain</p>', unsafe_allow_html=True)


with tabs[1]:
    st.header("Your Financial Dashboard")
    
    # Mock financial data
    savings = 15000
    investments = 50000
    debt = 5000
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Savings", f"${savings:,}", "+5%")
    
    with col2:
        st.metric("Investments", f"${investments:,}", "+12%")
    
    with col3:
        st.metric("Debt", f"${debt:,}", "-10%")
    
    # Interactive chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=['Jan', 'Feb', 'Mar', 'Apr', 'May'], y=[10000, 12000, 11000, 15000, 16000], name='Savings'))
    fig.add_trace(go.Scatter(x=['Jan', 'Feb', 'Mar', 'Apr', 'May'], y=[40000, 42000, 45000, 48000, 50000], name='Investments'))
    fig.update_layout(title='Financial Growth Over Time', xaxis_title='Month', yaxis_title='Amount ($)')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown('<p style="text-align: center;">Created with ❤️ using Streamlit and LangChain</p>', unsafe_allow_html=True)



with tabs[2]:
    st.header("Learn & Explore")
    
    topics = ["Investing Basics", "Retirement Planning", "Debt Management", "Tax Strategies"]
    selected_topic = st.selectbox("Choose a topic to learn about:", topics)
    
    if selected_topic == "Investing Basics":
        st.markdown("""
        <div class="custom-card">
            <h3>Investing Basics</h3>
            <p>Investing is the process of allocating your money into different financial instruments with the expectation of generating income or profit. Here are some key concepts:</p>
            <ul>
                <li>Stocks: Ownership shares in a company</li>
                <li>Bonds: Loans to companies or governments</li>
                <li>Mutual Funds: Pooled investments managed by professionals</li>
                <li>ETFs: Exchange-Traded Funds that track indexes or sectors</li>
            </ul>
            <p>Remember: Diversification is key to managing risk in your investment portfolio.</p>
        </div>
        """, unsafe_allow_html=True)
    elif selected_topic == "Retirement Planning":
            st.markdown("""
            <div class="custom-card">
                <h3>Retirement Planning</h3>
                <p>Planning for retirement is crucial for financial security in your later years. Consider these aspects:</p>
                <ul>
                    <li>Start early: The power of compound interest works best over long periods</li>
                    <li>Diversify your portfolio: Balance risk and reward</li>
                    <li>Understand your retirement accounts: 401(k)s, IRAs, and Roth IRAs</li>
                    <li>Estimate your retirement needs: Consider inflation and healthcare costs</li>
                    <li>Regularly review and adjust your plan</li>
                </ul>
                <p>Remember: It's never too early or too late to start planning for retirement!</p>
            </div>
            """, unsafe_allow_html=True)
    elif selected_topic == "Debt Management":
        st.markdown("""
        <div class="custom-card">
            <h3>Debt Management</h3>
            <p>Effective debt management is crucial for financial health. Here are some strategies:</p>
            <ul>
                <li>Prioritize high-interest debt: Pay off credit cards first</li>
                <li>Consider debt consolidation: Combine multiple debts into one lower-interest loan</li>
                <li>Create a budget: Track your income and expenses</li>
                <li>Use the snowball or avalanche method: Two popular debt repayment strategies</li>
                <li>Avoid taking on new debt while paying off existing debts</li>
            </ul>
            <p>Remember: Staying out of debt is easier than getting out of debt!</p>
        </div>
        """, unsafe_allow_html=True)
    elif selected_topic == "Tax Strategies":
        st.markdown("""
        <div class="custom-card">
            <h3>Tax Strategies</h3>
            <p>Implementing effective tax strategies can help you keep more of your hard-earned money. Consider these tips:</p>
            <ul>
                <li>Maximize retirement account contributions: 401(k)s and IRAs offer tax advantages</li>
                <li>Take advantage of tax-loss harvesting: Offset capital gains with losses</li>
                <li>Consider a Health Savings Account (HSA): Triple tax advantage for healthcare costs</li>
                <li>Donate to charity: Charitable contributions can be tax-deductible</li>
                <li>Stay informed about tax law changes: Tax codes evolve, so keep up to date</li>
            </ul>
            <p>Remember: Always consult with a tax professional for personalized advice!</p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<p style="text-align: center;">Created with ❤️ using Streamlit and LangChain</p>', unsafe_allow_html=True)

        


    # Sidebar
    with st.sidebar:
        st.markdown("""
            <h2 style="color: var(--accent-color);">AI Financial Assistant</h2>
            <div class="custom-card">
                <p>I'm your personal AI financial advisor. Ask me anything about:</p>
                <ul>
                    <li>Investments</li>
                    <li>Credit Cards</li>
                    <li>Loans</li>
                    <li>Retirement Planning</li>
                    <li>Tax Strategies</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
   
        # Interactive financial goal setter
        st.markdown("<h3 style='color: var(--accent-color);'>Set Your Financial Goal</h3>", unsafe_allow_html=True)
        goal_type = st.selectbox("Goal Type", ["Savings", "Investment", "Debt Repayment"])
        goal_amount = st.number_input("Target Amount ($)", min_value=100, max_value=1000000, value=10000, step=100)
        goal_date = st.date_input("Target Date")

        if st.button("Set Goal"):
            st.success(f"Goal set: ${goal_amount:,} for {goal_type} by {goal_date.strftime('%B %d, %Y')}")

        # Feedback section
        st.markdown("""
            <h3 style="color: var(--accent-color);">Your Feedback Matters</h3>
            <div class="custom-card">
                <p>Help us improve our AI with your valuable input!</p>
            </div>
        """, unsafe_allow_html=True)

        feedback = st.text_area("Share your thoughts:")
        if st.button("Submit Feedback", key="feedback_button"):
            st.success("Thank you for your feedback! 🌟")

# Footer
st.markdown("---")
st.markdown("""
    <footer style="text-align: center; color: var(--text-color); font-size: 0.8rem;">
        <p>Powered by cutting-edge AI and financial expertise</p>
        <p>© 2024 MarketWealth AI. All rights reserved.</p>
    </footer>
""", unsafe_allow_html=True)

# Add tooltips to certain elements
st.markdown("""
    <div class="tooltip">Hover over me
        <span class="tooltiptext">This is a tooltip example!</span>
    </div>
""", unsafe_allow_html=True)

# Easter egg: Hidden feature
if st.sidebar.checkbox("🔒 Unlock Hidden Feature", key="hidden_feature"):
    st.sidebar.success("You've discovered a hidden feature! 🎉")
    st.sidebar.markdown("""
        <div class="custom-card neon-glow">
            <h4>AI Stock Predictor</h4>
            <p>Enter a stock symbol to get an AI-powered prediction!</p>
        </div>
    """, unsafe_allow_html=True)
    
    stock_symbol = st.sidebar.text_input("Stock Symbol (e.g., AAPL)", "AAPL")
    prediction_days = st.sidebar.slider("Prediction Days", 1, 30, 7)

    @st.cache_data(ttl=3600)
    def get_stock_data(symbol, days=365):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        try:
            data = yf.download(symbol, start=start_date, end=end_date)
            return data
        except Exception as e:
            st.sidebar.error(f"Error fetching data: {e}")
            return None

    def prepare_data(data):
        data['Prediction'] = data['Close'].shift(-prediction_days)
        data['MA5'] = data['Close'].rolling(window=5).mean()
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['RSI'] = calculate_rsi(data['Close'])
        data = data.dropna()

        X = data[['Close', 'Volume', 'MA5', 'MA20', 'RSI']]
        y = data['Prediction']

        return X, y

    def calculate_rsi(data, periods=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def train_model(X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model

    if st.sidebar.button("Predict"):
        with st.spinner("Fetching data and making predictions..."):
            data = get_stock_data(stock_symbol)
            if data is not None:
                X, y = prepare_data(data)
                model = train_model(X, y)
                
                last_data = X.iloc[-1].values.reshape(1, -1)
                prediction = model.predict(last_data)[0]
                current_price = data['Close'].iloc[-1]
                price_change = prediction - current_price
                percent_change = (price_change / current_price) * 100

                st.sidebar.markdown(f"""
                    <div class="custom-card neon-glow">
                        <h4>AI Prediction for {stock_symbol}</h4>
                        <p>Current Price: ${current_price:.2f}</p>
                        <p>Predicted Price (in {prediction_days} days): ${prediction:.2f}</p>
                        <p>Predicted Change: ${price_change:.2f} ({percent_change:.2f}%)</p>
                    </div>
                """, unsafe_allow_html=True)

                # Visualization
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Historical Close Price'))
                fig.add_trace(go.Scatter(x=[data.index[-1], data.index[-1] + timedelta(days=prediction_days)],
                                         y=[current_price, prediction], name='Prediction', line=dict(dash='dash')))
                fig.update_layout(title=f'{stock_symbol} Stock Price Prediction',
                                  xaxis_title='Date',
                                  yaxis_title='Price ($)',
                                  template='plotly_dark')
                st.plotly_chart(fig, use_container_width=True)

                # Additional insights
                st.sidebar.markdown("""
                    <div class="custom-card">
                        <h4>Market Insights</h4>
                        <ul>
                            <li>Recent trend: {}</li>
                            <li>Volume analysis: {}</li>
                            <li>RSI indicator: {}</li>
                        </ul>
                    </div>
                """.format(
                    "Upward" if percent_change > 0 else "Downward",
                    "High" if data['Volume'].iloc[-1] > data['Volume'].mean() else "Normal",
                    "Overbought" if X['RSI'].iloc[-1] > 70 else "Oversold" if X['RSI'].iloc[-1] < 30 else "Neutral"
                ), unsafe_allow_html=True)

                st.sidebar.warning("Disclaimer: This prediction is for educational purposes only. Always do your own research before making investment decisions.")
            else:
                st.sidebar.error("Failed to fetch stock data. Please try again.")


