from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
model = OllamaLLM(model="llama3.2")

template = """
You are expert about physics.

Here are some topics about physics: {topics}

Answer the following question based on the topics above: {question}
"""

prompt = ChatPromptTemplate.from_template(template, input_variables=["topics", "question"])
chain = prompt | model

# Make it a chatbot who will continuously answer questions abouthysics until the user decides to stop.
while True:
    ques = str(input("Please input your question about physics (or type 'quit' to exit): "))
    if ques.lower() == "quit":
        break
    result = chain.invoke(
        {
            "topics": [],
            "question": ques
        }
    )
    print(result)
    
    #use streamlit to display the result in a more user-friendly way
import streamlit as st
st.write(result)