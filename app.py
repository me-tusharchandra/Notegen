import os
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
import google.generativeai as genai
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables and configure Google Generative AI
load_dotenv()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Set up the language model and embeddings
llm = GoogleGenerativeAI(model="models/text-bison-001", temperature=0.1)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def process_document(file):
    text = file.getvalue().decode("utf-8")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    
    vectorstore = Chroma.from_texts(chunks, embeddings, persist_directory="./chroma_db")
    # vectorstore.persist()
    
    return vectorstore

def setup_qa_chain(vectorstore):
    prompt_template = """You are an AI assistant tasked with writing a comprehensive document in Markdown format based on a provided table of contents. 
    Use the following pieces of context to write detailed sections for the document.
    If you don't have enough information, state that more research is needed on that topic.

    Context: {context}

    Section to write:
    {question}

    AI Assistant: Write a detailed section for the given header in Markdown format. Follow these guidelines strictly:

    1. Provide comprehensive and informative content directly related to the section title.
    2. Do not create additional headers or a table of contents.
    3. Ensure the content is well-structured, relevant to the topic, and flows logically.
    4. Use the appropriate number of '#' symbols for the header level as indicated in the section title.
    5. Do not include any links, references to images, diagrams, or any other form of visual data. Focus solely on textual content.
    6. Limit your response to approximately 200-300 words.
    7. If you reach the word limit, end your response with a complete sentence.
    8. Prioritize the most important information and concepts within the word limit.
    Remember, your task is to provide purely textual, relevant content for the given section without any visual elements or links."""


    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return qa

def generate_content(qa, section):
    result = qa.invoke({"query": section})
    return result['result']

# Streamlit UI
st.title("Notegen: Customized Notes Generation")
st.write("Notegen is an innovative application that generates comprehensive notes based on user-defined topics and structures. By leveraging pools of data and a custom table of contents, it creates tailored, in-depth content for various purposes such as research papers, reports, or study materials.")

uploaded_file = st.file_uploader("Upload a text file", type="txt")

if uploaded_file is not None:
    with st.spinner("Processing document..."):
        vectorstore = process_document(uploaded_file)
        qa_chain = setup_qa_chain(vectorstore)
    st.success("Document processed successfully!")

    toc = st.text_input("Enter the table of contents (comma-separated):")
    
    if st.button("Generate Content"):
        if toc:
            sections = [section.strip() for section in toc.split(',')]
            
            content = ""
            for section in sections:
                with st.spinner(f"Generating content for: {section}"):
                    section_content = generate_content(qa_chain, section)
                    content += f"{section_content}\n\n"
            
            st.markdown(content)
            
            # Option to download the generated content
            st.download_button(
                label="Download Generated Content",
                data=content,
                file_name="generated_content.md",
                mime="text/markdown"
            )
        else:
            st.warning("Please enter a table of contents.")