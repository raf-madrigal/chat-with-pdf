# from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
# from langchain.memory import ConversationBufferWindowMemory, ConversationBufferMemory
# from langchain.retrievers.multi_query import MultiQueryRetriever

# from langchain.chains import ConversationalRetrievalChain
# from utils.utils import get_hf_llm
# from langchain.chains import LLMChain
# from prompts.openai_prompts import (ZERO_SHOT_PROMPT, 
#                      CONDENSE_QUESTION_PROMPT, 
#                      MULTI_QUERY_PROMPT)

# utility import

from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from operator import itemgetter
from utils.text_processing import remove_text_in_parenthesis

# memory
from langchain_core.messages import get_buffer_string
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory

# prompt import

from prompts.mistral_prompts import CONDENSE_QUESTION_PROMPT, ZERO_SHOT_PROMPT
from prompts.general_prompts import stuff_documents



class LCELConversationalRetrieverChain():

    def __init__(self, llm, retriever, memory, condense_llm=None, answer_llm=None, condense_prompt=None, answer_prompt=None):
    
        self.memory = memory
        self.retriever=retriever
        self.llm = llm
        self.condense_llm = condense_llm if condense_llm else llm
        self.answer_llm = answer_llm if answer_llm else llm 
        self.condense_prompt = condense_prompt if condense_prompt else CONDENSE_QUESTION_PROMPT
        self.answer_prompt = answer_prompt if answer_prompt else ZERO_SHOT_PROMPT
        

        
    def _get_chat_history_chain(self, memory):

        get_chat_history = RunnablePassthrough.assign(
            chat_history=(
                RunnableLambda(memory.load_memory_variables) 
                | itemgetter('history') 
                | RunnableLambda(get_buffer_string)
            )
        ) 
        return get_chat_history
    
    def _get_standalone_query_chain(self, llm):

        

        standalone_query_chain = RunnablePassthrough.assign(
            standalone_question = (
                self.condense_prompt 
                | llm 
                | RunnableLambda(remove_text_in_parenthesis)
            )
        )
        return standalone_query_chain
    
    def _get_retriever_chain(self, retriever):
        
        retriever_chain = RunnablePassthrough.assign(
            docs=itemgetter('standalone_question') | retriever
        )

        return retriever_chain
    
    def _get_answer_chain(self, llm):

        answer_chain = (
            RunnablePassthrough.assign(context=(itemgetter('docs') | RunnableLambda(stuff_documents)))
            | RunnablePassthrough.assign(answer=self.answer_prompt | llm | RunnableLambda(remove_text_in_parenthesis))   
        )
        
        return answer_chain
    
    def build_chain(self):
        self.chain =  (self._get_chat_history_chain(self.memory) 
                | self._get_standalone_query_chain(self.condense_llm) 
                | self._get_retriever_chain(self.retriever)
                | self._get_answer_chain(self.answer_llm)
                
        )
        return self.chain



class LCELBaseChatbot():

    def __init__(self, vectordb, llm, **kwargs):

        self.memory_window = kwargs.get('memory_window', 5)
        self.vectordb = vectordb
        self.llm = llm
        
        self.message_history = kwargs.get('message_history', None)
        if self.message_history:
            self.chat_history = self.load_conversation_history_from_database(self.message_history)
        else:

            self.chat_history = ChatMessageHistory()


    def load_memory_with_history(self):

        chat_history = self.chat_history
        k = self.memory_window
        

        if k != -1: # remembers k conversations
            memory = ConversationBufferWindowMemory(
                            chat_memory=chat_history,
                            k=k,
                            return_messages=True,
                            output_key='answer', 
                            input_key='question'
                            )

        elif k == -1: # remembers all
            memory = ConversationBufferMemory(
                            chat_memory=chat_history,
                            return_messages=True,
                            output_key='answer', 
                            input_key='question'
                            )

        else:
            raise ValueError()

        return memory
    
    def initialize(self, **kwargs):

        vectordb = self.vectordb
        memory = self.load_memory_with_history()
        self.memory = memory
        llm = self.llm

        retriever = vectordb.as_retriever(
                                    search_type="similarity",
                                    search_kwargs={"score_threshold": 0.8})
        
        self.retriever = retriever

        chain = LCELConversationalRetrieverChain(llm=llm, memory=memory, retriever=retriever, **kwargs).build_chain()

        self.chain = chain

    def chat(self, query):
        chain = self.chain
        input_question = {'question' : query}
        results = chain.invoke(input_question)
        output_answer = {'answer':results['answer']}
        self.memory.save_context(input_question, output_answer)

        return output_answer['answer']
    

    def load_conversation_history_from_database(self, message_history):

        chat_history = ChatMessageHistory()

        for message in message_history:
            if message['role'] == 'assistant':
                chat_history.add_ai_message(message['content'])
            elif message['role'] == 'user':
                chat_history.add_user_message(message['content'])
            else:
                ValueError()

        return chat_history
        