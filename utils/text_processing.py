import re
from langchain_core.messages import AIMessage, HumanMessage

def remove_text_in_parenthesis(text):
    pattern = r"\([^)]*\)"
    if isinstance(text, AIMessage):
        text = text.content

    return re.sub(pattern, ' ', text).strip()


