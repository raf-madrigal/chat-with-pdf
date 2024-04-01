from utils.utils import load_and_store_file, load_vector_store, get_hf_llm
from chat.chatbot import LCELBaseChatbot
import tempfile
from langchain_openai import ChatOpenAI
import streamlit as st
from dotenv import load_dotenv
from streamlit_chat import message

from prompts.mistral_prompts import CONDENSE_QUESTION_PROMPT as mistral_condense_prompt
from prompts.mistral_prompts import ZERO_SHOT_PROMPT as mistral_answer_prompt
from prompts.openai_prompts import CONDENSE_QUESTION_PROMPT as oai_condense_prompt
from prompts.openai_prompts import ZERO_SHOT_PROMPT as oai_answer_prompt

load_dotenv()

LLM_DICT = {
    'Mistral-7B-Instruct-v0.2' : get_hf_llm('mistralai/Mistral-7B-Instruct-v0.2', temperature=0.001, repetition_penalty=1.1),
    'OpenAI (gpt-3.5-turbo)' : ChatOpenAI(temperature=0, model='gpt-3.5-turbo'), 
    'OpenAI (gpt-4)' : ChatOpenAI(temperature=0, model='gpt-4')
}

LLM_DESCRIPTION_DICT = {
    'Mistral-7B-Instruct-v0.2' :  """Mistral is an instruct fine-tuned version of Mistral-7B-v0.2 available on HuggingFace (Open Source)""",
    'OpenAI (gpt-3.5-turbo)' : """GPT-3.5 is a model developed by OpenAI""",
    'OpenAI (gpt-4)' : """GPT-3.5 is a model developed by OpenAI"""
}

DISCLAIMER_DICT = {
    'Mistral-7B-Instruct-v0.2' : "HuggingFaceEndpoint in LangChain requires a positive temperature (not temp=0). For this demonstration, the temperature is set to 0.001. Also it was observed that Mistral has a bit of a problem with repetition" ,
    'OpenAI (gpt-3.5-turbo)' : "Temperature is set to 0",
    'OpenAI (gpt-4)' : "Temperature is set to 0"
}

PROMPTS_DICT = {
    'Mistral-7B-Instruct-v0.2' : dict(condense_prompt = mistral_condense_prompt,
                                        answer_prompt = mistral_answer_prompt), 
    'OpenAI (gpt-3.5-turbo)' : dict(condense_prompt = oai_condense_prompt,
                                        answer_prompt = oai_answer_prompt), 
    'OpenAI (gpt-4)' : dict(condense_prompt = oai_condense_prompt,
                                        answer_prompt = oai_answer_prompt), 
}
                                        

st.title("Chat with a PDF 🔗")

  
    
with st.sidebar:

    st.markdown('# Upload a PDF')
    uploaded_file = st.file_uploader("Upload File", type="pdf")
    st.markdown('# Choose an LLM model')
    llm_choice = st.selectbox('Select the backend Large Language Model to use:', 
                                 ('OpenAI (gpt-3.5-turbo)', 'OpenAI (gpt-4)', 'Mistral-7B-Instruct-v0.2'), 
                                 )
    st.markdown('# LLM Description')
    st.markdown(LLM_DESCRIPTION_DICT[llm_choice])
    st.markdown('# Disclaimer')
    st.markdown(DISCLAIMER_DICT[llm_choice])




if uploaded_file :
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
        vector_db = load_and_store_file(tmp_file_path) 
        # vector_db = load_vector_store()

    
    if 'messages' not in st.session_state.keys():
        st.session_state['messages'] = []
        st.session_state.messages.append({"role": "assistant", "content": 'Hi there! If you wanna chat, please Upload a PDF on the left pane (Otherwise I might give an error!)'})

    
    bot = LCELBaseChatbot(llm=LLM_DICT[llm_choice], vectordb=vector_db, message_history=st.session_state.messages)
    
    bot.initialize(**PROMPTS_DICT[llm_choice])
    


    for message in st.session_state['messages']:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    if question := st.chat_input("Ask a question or Say Something"):
       
        st.session_state.messages.append({"role": "user", "content": question})
        
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message('assistant'):
            response = bot.chat(question)
            st.markdown(response)


        st.session_state['messages'].append({
            'role':'assistant', 
            'content': response
        })


