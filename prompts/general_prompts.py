from langchain_core.prompts import PromptTemplate
from langchain.schema import format_document

DOCUMENT_TEMPLATE = """{page_content}"""
DOCUMENT_PROMPT = PromptTemplate.from_template(DOCUMENT_TEMPLATE)

def stuff_documents(docs, 
                document_prompt=DOCUMENT_PROMPT, 
                document_separator='\n\n'):

    doc_strings = [format_document(doc, document_prompt) for doc in docs]

    return document_separator.join(doc_strings)

