from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import shutil
import os
import openai
# from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
# from langchain.chains.question_answering import load_qa_chain
""" from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.llms import OpenAI """
from langchain.chains import ConversationalRetrievalChain
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
        chunk_size=1000,
        chunk_overlap=100)
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


def process_question(docsearch, question, chat_history, filename):
    print(filename)
    """docs = docsearch.similarity_search(
        question, namespace=filename)
    docs_page_content = " ".join([d.page_content for d in docs])
    chat = ChatOpenAI(model_name="gpt-4", temperature=0)

#    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

    template = You are an expert attorney. You can
    answer legal questions based on the context you are given: {docs}
    If you don't know the answer, just say you don't know.
    DO NOT try to make up an answer.
    If the question is not related to the context,
    politely respond that you are tuned to only answer questions
    that are related to the context.

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(
                                                    human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
#   chain = load_qa_chain(llm, chain_type="stuff")
    chain = LLMChain(llm=chat, prompt=chat_prompt)
    chain.run(question=question, docs=docs_page_content) """

    qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(
                                                model_name="gpt-4",
                                                temperature=0,
                                                openai_api_key=OPENAI_API_KEY
                                                ),
                                               docsearch.as_retriever(
                                                   namespace=filename))

    prompt = """You are an expert attorney.
            Answer the question with the context provided and the chat history."""
    prompt += question
    result = qa({"question": prompt,
                 "chat_history": chat_history})
    answer = result["answer"]

    return answer


docsearch_cache = {}  # Cache to store docsearch objects. Key is filename.


class Message(BaseModel):
    role: str
    content: str


class Chat(BaseModel):
    messages: List[Message]
    filename: str


class Chat2(BaseModel):
    messages: List[Message]


@app.post('/upload-file')
async def upload_file(file: UploadFile = File(...)):
    with open(os.path.join("/tmp", file.filename), "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    docsearch = process_document_and_query(os.path.join("/tmp", file.filename))

    # Save the docsearch object in the cache
    docsearch_cache[file.filename] = docsearch

    return {"filename": file.filename}


@app.post('/process-pdf')
async def process_pdf(chat: Chat = Body(...)):
    prompt = ''
    messages = [message.dict() for message in chat.messages]
    last_message = messages[-1]
    print(messages)
    question = last_message['content']
    # Convert messages to string
    all_but_last_messages = messages[:-1]
    chat_history = []
    for i in range(0, len(all_but_last_messages), 2):
        chat_history.append((all_but_last_messages[i]['content'],
                             all_but_last_messages[i + 1]['content']))
    messages_str_last = ' '.join([message['content']
                                  for message in all_but_last_messages])
    messages_str = ' '.join([message['content'] for message in messages])
    prompt += messages_str
    print(messages_str_last)
    print(prompt)
    filename = chat.filename
    docsearch = docsearch_cache[filename]
    answer = process_question(docsearch, question,
                              chat_history, chat.filename)

    return JSONResponse(content={'answer': answer})


@app.post('/chat')
async def chat(chat: Chat2):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[message.dict() for message in chat.messages],
        max_tokens=4000,
        temperature=0
    )
    messagess = [message.dict() for message in chat.messages]
    print(messagess)
    answer = response['choices'][0]['message']['content']
    return JSONResponse(content={'answer': answer})

if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get('PORT', 5000))
    print(f'Starting server on port {port}')
    uvicorn.run(app, host='0.0.0.0', port=port, timeout_keep_alive=1000)
