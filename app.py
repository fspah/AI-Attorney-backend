from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import os
import openai
from langchain.llms import OpenAI
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains.question_answering import load_qa_chain
import pinecone
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_API_ENV = os.getenv('PINECONE_API_ENV')

openai.api_key = OPENAI_API_KEY


pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_API_ENV
)

index_name = "langchain2"


def process_document_and_query(file, question, prompt):
    loader = UnstructuredPDFLoader(file)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0)
    texts = text_splitter.split_documents(data)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    docsearch = Pinecone.from_texts(
        [t.page_content for t in texts], embeddings, index_name=index_name)
    docs = docsearch.similarity_search(question, include_metadata=True)
    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    chain = load_qa_chain(llm, chain_type="stuff")
    answer = chain.run(input_documents=docs, question=prompt)

    return answer


def answer_question_without_file(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200,
        temperature=0
    )
    answer = response['choices'][0]['message']['content']
    return answer


@app.post('/process-pdf')
async def process_pdf(file: UploadFile = File(None),
                      question: str = Form(...),
                      location: str = Form(...)):
    prompt = (
        "You are an expert attorney. "
        "Give your advice on the following question: "
    )
    located = " I am located here: "
    located += location
    prompt += question + located
    print(prompt)

    if file and file.filename != '':
        with open(os.path.join("/tmp", file.filename), "wb") as buffer:
            buffer.write(await file.read())
        answer = process_document_and_query(
            os.path.join("/tmp", file.filename), question, prompt)
    else:
        answer = answer_question_without_file(prompt)

    return JSONResponse(content={'answer': answer})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000)
