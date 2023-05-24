from fastapi import FastAPI, File, UploadFile
from fastapi import Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import shutil
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

print('b')
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_API_ENV
)

index_name = "langchain2"
print('b')


def process_document_and_query(file):
    loader = UnstructuredPDFLoader(file)
    print('a', flush=True)

    data = loader.load()
    print('a', flush=True)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50)
    print('a', flush=True)

    texts = text_splitter.split_documents(data)
    print('a', flush=True)
    print(f'Now you have {len(texts)} documents')

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    print('a', flush=True)

    filename = os.path.basename(file)
    print(filename)
    docsearch = Pinecone.from_texts(
        [
            t.page_content for t in texts],
        embeddings,
        index_name=index_name,
        namespace=filename)
    print('a', flush=True)
    print(docsearch, flush=True)

    return docsearch


def process_question(docsearch, question, prompt, filename):
    print(filename)
    docs = docsearch.similarity_search(
        question, namespace=filename)
    print('a', flush=True)

    llm = OpenAI(modelName='gpt-4', temperature=0,
                 openai_api_key=OPENAI_API_KEY)
    print('a', flush=True)

    chain = load_qa_chain(llm, chain_type="stuff")
#                         return_source_documents=True)
    print('a', flush=True)

    answer = chain.run(input_documents=docs, question=prompt)
    print('a', flush=True)

    return answer


docsearch_cache = {}  # Cache to store docsearch objects. Key is filename.


class Message(BaseModel):
    role: str
    content: str


class Chat(BaseModel):
    messages: List[Message]


@app.post('/upload-file')
async def upload_file(file: UploadFile = File(...)):
    with open(os.path.join("/tmp", file.filename), "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # After the file is saved, process the document
    docsearch = process_document_and_query(os.path.join("/tmp", file.filename))

    # Save the docsearch object in the cache
    docsearch_cache[file.filename] = docsearch

    return {"filename": file.filename}


@app.post('/process-pdf')
async def process_pdf(chat: Chat = Body(...)):
    prompt = (
        "You are an expert attorney. "
        "Give your advice on the following question: "
    )
    messages = [message.dict() for message in chat.messages]
    last_message = messages[-1]
    question = last_message['content']
    # Convert messages to string
    messages_str = ' '.join([message['content'] for message in messages])
    prompt += messages_str
    print(prompt)
    docsearch = docsearch_cache[chat.filename]
    answer = process_question(docsearch, question, prompt, chat.filename)

    return JSONResponse(content={'answer': answer})


@app.post('/chat')
async def chat(chat: Chat):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[message.dict() for message in chat.messages],
        max_tokens=4000,
        temperature=0
    )
    answer = response['choices'][0]['message']['content']
    return JSONResponse(content={'answer': answer})

if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get('PORT', 5000))
    print(f'Starting server on port {port}')
    uvicorn.run(app, host='0.0.0.0', port=port, timeout_keep_alive=1000)
