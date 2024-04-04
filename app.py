from utils.utils import load_and_store_file, load_vector_store, get_hf_llm, store_documents, load_and_split_documents
from chat.chatbot import LCELBaseChatbot
# from streamlit_chat import message
import tempfile
from langchain_openai import ChatOpenAI
import streamlit as st
# from dotenv import load_dotenv

from langchain_community.llms import HuggingFaceEndpoint
from prompts.mistral_prompts import CONDENSE_QUESTION_PROMPT as mistral_condense_prompt
from prompts.mistral_prompts import ZERO_SHOT_PROMPT as mistral_answer_prompt
from prompts.openai_prompts import CONDENSE_QUESTION_PROMPT as oai_condense_prompt
from prompts.openai_prompts import ZERO_SHOT_PROMPT as oai_answer_prompt

import os
os.environ['LANGCHAIN_TRACING_V2'] = st.secrets['LANGCHAIN_TRACING_V2']
os.environ['LANGCHAIN_ENDPOINT'] = st.secrets['LANGCHAIN_ENDPOINT']
os.environ['LANGCHAIN_API_KEY'] = st.secrets['LANGCHAIN_API_KEY']
os.environ['LANGCHAIN_PROJECT'] = st.secrets['LANGCHAIN_PROJECT']

st.set_page_config(page_title = 'Chat with your PDF')

st.title("Chat with PDFs 🔗")


mistral_llm = HuggingFaceEndpoint(
            repo_id='mistralai/Mistral-7B-Instruct-v0.2',
            temperature=0.001,
            max_new_tokens=250, 
            repetition_penalty=1.1,
            huggingfacehub_api_token=st.secrets['HUGGINGFACEHUB_API_TOKEN']
        )

oai_llm = ChatOpenAI(temperature=0, 
                    model='gpt-3.5-turbo',
                    openai_api_key=st.secrets['OPENAI_API_KEY'], 
                    openai_organization=st.secrets['OPENAI_ORGANIZATION']) 

oai4_llm = ChatOpenAI(temperature=0, 
                    model='gpt-4',
                    openai_api_key=st.secrets['OPENAI_API_KEY'], 
                    openai_organization=st.secrets['OPENAI_ORGANIZATION'])

GLOBAL_LLM_MODELS = {
    'Mistral-7B-Instruct-v0.2' : mistral_llm,
    'OpenAI (gpt-3.5-turbo)' : oai_llm,
    'OpenAI (gpt-4)' : oai4_llm
}


LLM_DESCRIPTIONS = {
    'Mistral-7B-Instruct-v0.2' :  """Mistral is an instruct fine-tuned version of Mistral-7B-v0.2 available on HuggingFace (Open Source)""",
    'OpenAI (gpt-3.5-turbo)' : """GPT-3.5 is a model developed by OpenAI""",
    'OpenAI (gpt-4)' : """GPT-4 is a model developed by OpenAI"""
}

LLM_DISCLAIMERS = {
    'Mistral-7B-Instruct-v0.2' : "HuggingFaceEndpoint in LangChain requires a positive temperature (not temp=0). For this demonstration, the temperature is set to 0.001. Also it was observed that Mistral has a bit of a problem with repetition" ,
    'OpenAI (gpt-3.5-turbo)' : "Temperature is set to 0",
    'OpenAI (gpt-4)' : "Temperature is set to 0"
}

CHAIN_PROMPTS = {
    'Mistral-7B-Instruct-v0.2' : dict(condense_prompt = mistral_condense_prompt,
                                        answer_prompt = mistral_answer_prompt), 
    'OpenAI (gpt-3.5-turbo)' : dict(condense_prompt = oai_condense_prompt,
                                        answer_prompt = oai_answer_prompt), 
    'OpenAI (gpt-4)' : dict(condense_prompt = oai_condense_prompt,
                                        answer_prompt = oai_answer_prompt), 
}
        
def show_chat_messages():
    for message in st.session_state['messages']:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

def load_files_and_get_chain():

    splits = []
    for file in st.session_state['pdfs']:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file.getbuffer())
            tmp_file_path = tmp_file.name
        
            splits_i = load_and_split_documents(tmp_file_path) 
            splits += splits_i

    st.session_state['vector_db'] = store_documents(splits)

    st.session_state['chain'] = LCELBaseChatbot(llm=st.session_state['llm'], 
                                    vectordb=st.session_state['vector_db'], 
                                    message_history=st.session_state['messages'])

    st.session_state['chain'].initialize(**st.session_state['prompts'])
    
    st.session_state['chain_loaded'] = True
    # st.markdown(st.session_state['chain_loaded'])
   
def llm_selector():
    st.session_state['llm'] = GLOBAL_LLM_MODELS[st.session_state['model_choice']]
    st.session_state['llm_description'] = LLM_DESCRIPTIONS[st.session_state['model_choice']]
    st.session_state['llm_disclaimer'] = LLM_DISCLAIMERS[st.session_state['model_choice']]
    st.session_state['prompts'] = CHAIN_PROMPTS[st.session_state['model_choice']]
    # st.session_state['chain_loaded'] = False
    print( st.session_state['model_choice'],   st.session_state['llm'], st.session_state['llm_disclaimer'])
    st.session_state['llm_selected'] = True
    st.session_state.messages.append({"role": "system", 
                                      "content": f'Notice: You selected {st.session_state["model_choice"]} as an LLM'}
                                      )
    


def initialize_states():
    if 'chain_loaded' not in st.session_state.keys():
        st.session_state['chain_loaded'] = False
    if 'llm_selected' not in st.session_state.keys():
        st.session_state['llm_selected'] = True
    if 'model_choice' not in st.session_state.keys():
        st.session_state['model_choice'] = 'OpenAI (gpt-3.5-turbo)'
    if 'messages' not in st.session_state.keys():
        st.session_state['messages'] = []
        st.session_state.messages.append({"role": "assistant", "content": 'Hi there! If you wanna chat, please Upload a PDF on the left pane (Otherwise I might give an error!)'})
    if 'llm' not in st.session_state.keys():
        st.session_state['llm'] = GLOBAL_LLM_MODELS[ st.session_state['model_choice']]
    if 'llm_description' not in st.session_state.keys():
        st.session_state['llm_description'] = LLM_DESCRIPTIONS[ st.session_state['model_choice']]
    if 'llm_disclaimer' not in st.session_state.keys():
        st.session_state['llm_disclaimer'] = LLM_DISCLAIMERS[st.session_state['model_choice']]
    if 'prompts' not in st.session_state.keys():
        st.session_state['prompts'] = CHAIN_PROMPTS[ st.session_state['model_choice']]



def main():

    # if 'chain_loaded' not in st.session_state.keys():
    #     st.session_state['chain_loaded'] = False
    # if 'llm_selected' not in st.session_state.keys():
    #     st.session_state['llm_selected'] = False
    # if 'model_choice' not in st.session_state.keys():
    #     st.session_state['model_choice'] = 'OpenAI (gpt-3.5-turbo)'
    # if 'messages' not in st.session_state.keys():
    #     st.session_state['messages'] = []
    #     st.session_state.messages.append({"role": "assistant", "content": 'Hi there! If you wanna chat, please Upload a PDF on the left pane (Otherwise I might give an error!)'})
    
    st.session_state['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
    st.session_state['OPENAI_ORGANIZATION'] = st.secrets['OPENAI_ORGANIZATION']
    st.session_state['HUGGINGFACEHUB_API_TOKEN'] = st.secrets['HUGGINGFACEHUB_API_TOKEN']

    with st.sidebar:
        st.markdown('# Instructions:')
        st.markdown('## 1. Choose an LLM model to Chat with')
        st.selectbox('Select the backend Large Language Model to use:', 
                                ('OpenAI (gpt-3.5-turbo)', 'OpenAI (gpt-4)', 'Mistral-7B-Instruct-v0.2'), 
                                 on_change=llm_selector, key='model_choice', 
                                 )
        st.markdown('Note: If you want to change the LLM mid-conversation, please click "Process PDFs!" again to load the proper LLM prompts')
        st.markdown('## 2. Upload PDF/s')

        # print(st.session_state['pdfs'])
        st.file_uploader(
            label='Upload Document/s', 
            type='pdf', 
            accept_multiple_files=True, 
            key='pdfs',
            disabled=not st.session_state['llm_selected']
            
        )

        st.markdown('## 3. Click Process PDFs')
        st.button('Process PDFs!', on_click=load_files_and_get_chain, key='process_button')
            
        if st.session_state['chain_loaded']:
            st.markdown('Processing Done! LLM Fully Loaded!')
            st.markdown('## 4. Read More Information about the models')

            st.markdown(f'### Description\n{st.session_state["llm_description"]}')

            st.markdown(f'### Disclaimer\n{st.session_state["llm_disclaimer"]}')

    show_chat_messages()
    
    if question := st.chat_input("Ask a question or Say Something"):
       
        st.session_state.messages.append({"role": "user", "content": question})
        
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message('assistant'):
            response = st.session_state['chain'].chat(question)
            st.markdown(response)

            st.session_state['messages'].append({'role':'assistant', 'content': response})
    

if __name__ == '__main__':
    initialize_states()
    # with st.container():
        # for k, v in st.session_state.items():
        #     if k=='prompts':
        #         pass
        #     else:
        #         st.markdown(f"{k}: {v}")
    main()