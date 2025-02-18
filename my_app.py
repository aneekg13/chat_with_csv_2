import pandas as pd
import os
from langchain_groq.chat_models import ChatGroq
from pandasai import SmartDataframe
import streamlit as st


st.title("CSV  File Chat")

#api_key="put your own key"
api_key = os.getenv("GROQ_API_KEY")

filename = "./data/dataset_telco_customer_churn.csv"

llm = ChatGroq(
    model_name="mixtral-8x7b-32768", 
    api_key = api_key)

def run_chat():
    print("")

data = pd.read_csv(filename)
df = SmartDataframe(data, config={"llm": llm})

prompt=st.text_area("enter your query here")

if st.button("generate"):
    if prompt:
       
       with st.spinner("Generating response..."):
            try:
                df = SmartDataframe(data, config={"llm": llm})
                response=df.chat(prompt)
                st.write(response)

            except Exception as e:
                st.error(f"An error occurred: {e}")

    else:
        st.warning("please enter the query before submit")
