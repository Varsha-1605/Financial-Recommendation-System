__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from faker import Faker
import random
from langchain_openai import ChatOpenAI
from datetime import date
from dateutil.relativedelta import relativedelta
import pandas as pd
from langchain.docstore.document import Document
from langchain.vectorstores import SKLearnVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import time
from openai import OpenAIError
from openai import APIConnectionError
from requests.exceptions import ConnectionError
import os
import traceback
import sys

openai_api_key = st.secrets["OPENAI_API_KEY"]

from langchain.globals import set_verbose
# Set verbosity level to True to enable detailed logging
set_verbose(True)

st.set_page_config(page_title='MarketWealth Recommender', layout='wide')

# Custom CSS for hover effects
st.markdown("""
    <style>
    .stTextInput>div>div>input:hover {
        border-color: #ff4b4b;
        box-shadow: 0 0 10px rgba(255, 75, 75, 0.5);
    }
    .stTextInput>div>div>textarea:hover {
        border-color: #ff4b4b;
        box-shadow: 0 0 10px rgba(255, 75, 75, 0.5);
    }
    .stTextInput>div>div>input:focus {
        border-color: #ff4b4b;
        box-shadow: 0 0 10px rgba(255, 75, 75, 0.5);
    }
    .stTextInput>div>div>textarea:focus {
        border-color: #ff4b4b;
        box-shadow: 0 0 10px rgba(255, 75, 75, 0.5);
    }
    </style>
""", unsafe_allow_html=True)

def global_exception_handler(exctype, value, tb):
    st.error("An unexpected error occurred. Our team has been notified. 🤖")
    # Log the error (in a production environment, you'd want to use a proper logging system)
    print("Uncaught exception:", file=sys.stderr)
    print("Type:", exctype, file=sys.stderr)
    print("Value:", value, file=sys.stderr)
    print("Traceback:", file=sys.stderr)
    traceback.print_tb(tb)

sys.excepthook = global_exception_handler

# UI Setup

st.markdown("""
    <style>
    .custom-title {
        font-size: 65px;
        font-weight: bold;
        color: #ff4b4b;
        text-align: center;
        text-decoration: underline;
        text-decoration-color: #FFFDD0;
        text-decoration-thickness: 2px;
        text-align: center;
        margin-top: 5px;
        margin-bottom: 20px;
    }
    .custom-icon {
        font-size: 60px;
        color: #ff4b4b;
    }
    </style>
    <div class="custom-title">
        MarketWealth Recommender <span class="custom-icon">💼</span>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    .welcome-title {
        font-size: 50px;
        font-weight: bold;
        color: #1E90FF;
        text-align: center;
        margin-bottom: 10px;
    }
    .welcome-text {
        font-size: 1.25rem;
        color: #333333;
        text-align: center;
        line-height: 1.6;
        margin-top: 0;
        margin-bottom: 20px;
    }
    .wave-icon {
        font-size: 35px;
        color: #ff4b4b;
    }
    </style>
    <div class="welcome-title">
        Welcome! <span class="wave-icon">👋</span>
    </div>
    <div class="welcome-text">
        Unlock personalized banking and financial recommendations effortlessly. 
        Simply type your query—be it credit cards, investments, mortgages, or 
        financial health—and receive advice finely tuned to your profile and goals.
    </div>
""", unsafe_allow_html=True)

# Tutorial Section
st.sidebar.markdown("""
    <h1 style="font-size:40px; color:#FFD700;">How to Use 🛠️</h1>
    <p>1. Enter a question about banking products or financial advice.</p>
    <p>2. Click on 'Get Recommendation' to receive personalized advice.</p>
    <p>3. Provide feedback to help us improve.</p>
""", unsafe_allow_html=True)

st.sidebar.markdown('<hr style="border:2px solid #ff4b4b;">', unsafe_allow_html=True)

# Feedback Section
st.sidebar.markdown(f"""
    <h1 style="font-size:40px; color:#FFD700;">Feedback 📝</h1>
    <p>We value your feedback! 😊</p>
""", unsafe_allow_html=True)

feedback = st.sidebar.text_area("Please leave your feedback here:")
if st.sidebar.button("Submit Feedback"):
    st.sidebar.success("Thank you for your feedback! 👍")

st.sidebar.markdown('<hr style="border:2px solid #ff4b4b;">', unsafe_allow_html=True)

# Sample Questions
st.sidebar.markdown(f"""
    <h1 style="font-size:40px; color:#FFD700;">Sample Questions ❓</h1>
    <p>Here are some sample questions you can try:</p>
    <ol>
        <li>Can you recommend a credit card for someone with a low credit score? 💳</li>
        <li>What investment options would you recommend for a 45-year-old customer looking to save for retirement in 15 years? elaborate it. 👵💼</li>
        <li>What mortgage options are available for first-time homebuyers?</li>
        <li>What banking products would you recommend for a 35-year-old customer with a high income and a credit score of 800? 💰</li>
        <li>What investment products are suitable for a risk-averse retiree? 🛡️</li>
        <li>What are the best options for debt consolidation? 💳</li>
        <li>How can a 40-year-old customer with a good credit score and some existing debt improve their credit profile? give a brief about it. 📊</li>
        <li>What credit card would you recommend for someone who travels frequently? ✈️</li>
    </ol>
""", unsafe_allow_html=True)

fake = Faker()

# Generate demographic and personal information
def generate_customer_data():
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

# Function to generate bureau product inquiries
def generate_inquiries(last_months):
    inquiries = []
    today = fake.date_this_month()

    for _ in range(random.randint(1, 5)):
        inquiry_date = fake.date_between(start_date=last_months, end_date=today)
        product_type = random.choice(['Personal Loan', 'Credit Card', 'Mortgage'])
        inquiries.append({'product_name': product_type, 'date': inquiry_date})

    return inquiries

# Generate dataset
def generate_dataset(num_rows, months):
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

sklearn_store = get_vectorstore()

def setup_retrieval_qa(sklearn_store):
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

retrieval_qa = setup_retrieval_qa(sklearn_store)

st.markdown("""
    <style>
    .custom-input-label {
        font-size: 30px;
        font-weight: bold;
        color: #FFD700;
        margin-bottom: 2px;
        display: block;
    }
    .custom-input {
        width: 100%;
        padding: 10px;
        font-size: 1rem;
        border: 2px solid #ccc;
        border-radius: 4px;
        transition: border-color 0.3s ease;
    }
    .custom-input:focus {
        border-color: #ff4b4b;
        outline: none;
    }
    </style>
    <label class="custom-input-label">Enter your query:</label>
    
""", unsafe_allow_html=True)

# Use JavaScript to capture the input value
question = st.text_input('-----------------------------')

if not question:
    st.write(":mag_right: Please enter a question to get a response.")

if st.button("Get Recommendation"):
    with st.spinner(":hourglass_flowing_sand: Generating recommendation..."):
        try:
            response = retrieval_qa.invoke(question)
            st.write(f"**Answer:** \n{response['result']}")

        except OpenAIError as e:
            st.error(f":x: An OpenAI error occurred: {str(e)}")
            st.info(":information_source: Please try again later or contact support if the problem persists.")

        except ValueError as e:
            st.error(f":x: Input error: {str(e)}")
            st.info(":information_source: Please check your input and try again.")

        except Exception as e:
            st.error(f":x: An unexpected error occurred: {str(e)}")
            st.info(":information_source: Please try again or contact support if the problem persists.")

            # Log the full error for debugging
            print(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown('<p style="text-align: center;">Created with ❤️ using Streamlit and LangChain</p>', unsafe_allow_html=True)
