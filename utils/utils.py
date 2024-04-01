from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyMuPDFLoader
from langchain_community.llms import HuggingFaceEndpoint
from dotenv import load_dotenv

from langchain.vectorstores import Annoy

load_dotenv()
    
def load_and_store_file(tmp_file_path, embeddings=None, verbose=True):

    

    VECTORSTORE_SAVE_PATH = 'vectorstore/db_annoy'

    if not embeddings:
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2') 

    
    loader = PyMuPDFLoader(file_path=tmp_file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=400, 
        chunk_overlap=20, 
        separators=["\n\n", "\n", "\.", " ", ""]
    )

    splits = text_splitter.split_documents(docs)

    vector_db = Annoy.from_documents(
                splits,
                embeddings,
            )
    
    vector_db.save_local(VECTORSTORE_SAVE_PATH)

    if verbose:
        print(f"Documents saved to {VECTORSTORE_SAVE_PATH}")

    return vector_db


def load_vector_store(vector_db_path=None, embedding_func=None, verbose=True):

    if not vector_db_path:
        vector_db_path = 'vectorstore/db_annoy'

    if not embedding_func:
        embedding_func = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2') 

    from langchain.vectorstores import Annoy
    
    vector_db = Annoy.load_local(
                    vector_db_path, 
                    embeddings=embedding_func,
                    allow_dangerous_deserialization=True
                )
    
    if verbose:

        if len(vector_db.similarity_search('hi')) > 0:
            print('Vector DB Loaded successfully from', vector_db_path)

        else:
            print('Fail to load Vector DB')

    return vector_db
    
def get_hf_llm(repo_id, temperature, repetition_penalty=1.1, max_new_tokens=250):


    hf_llm = HuggingFaceEndpoint(
            repo_id=repo_id, temperature=temperature,
            max_new_tokens=max_new_tokens, 
            repetition_penalty=repetition_penalty, 
        )
    
    return hf_llm



