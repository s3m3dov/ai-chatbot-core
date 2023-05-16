"""
This is the main file for the chatbot. It uses the OpenAI API to
generate responses to prompts.
"""

import json
from json import JSONDecodeError

from pathlib import Path
from dotenv import dotenv_values

from dotenv import dotenv_values
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import VectorStoreRetrieverMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import messages_from_dict
from langchain.vectorstores import Chroma

config = dotenv_values(".env")

log = print
max_tokens = {
    "default": 300,
    "gpt-3": 300,
    "gpt-3.5": 300,
    "gpt-3.5-turbo": 300,
    "gpt-4": 300,
    "dall-e": 300,
}
temperature = {
    "default": 0.3,
    "gpt-3": 0.3,
    "gpt-3.5": 0.3,
    "gpt-3.5-turbo": 0.3,
    "gpt-4": 0.3,
    "dall-e": 0.3,
}
gpt4 = ChatOpenAI(
    model_name="gpt-4",
    max_tokens=max_tokens["gpt-4"],
    temperature=temperature["gpt-4"],
)
gpt3_5_turbo = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    max_tokens=max_tokens["gpt-3.5-turbo"],
    temperature=temperature["gpt-3.5-turbo"],
)
dall_e = ChatOpenAI(
    model_name="dall-e",
    max_tokens=max_tokens["dall-e"],
    temperature=temperature["dall-e"],
)


class ChatBot:
    """
    This is a chatbot that uses the OpenAI API
    to generate responses to prompts.
    """

    def __init__(self):
        template = """
        Your name is SmartBot. 
        
        You are an AI chatbot that is designed to
        help people with their problems, and your tone is
        professional and friendly.
        
        You should try hard to answer the questions in a way 
        that is helpful to the person asking them, and these 
        answers should be detailed.
        
        Relevant pieces of previous conversation:
        {history}
        
        (You do not need to use these pieces of information if not relevant)
        
        Current conversation:
        Human: {input}
        SmartBot:
        
        """
        prompt_template = PromptTemplate(
            input_variables=["history", "input"], template=template
        )
        self.filename = "history.json"
        self.memory = self.set_memory_context()
        self.engine = LLMChain(
            llm=gpt3_5_turbo,
            prompt=prompt_template,
            memory=self.memory,
            verbose=True,
        )

    def get_messages(self) -> list:
        try:
            with Path(self.filename).open(mode="r", encoding="utf-8") as file:
                loaded_messages = json.load(file)
            return loaded_messages
        except JSONDecodeError:
            # Return an empty list.
            return []
        except FileNotFoundError:
            # Create the file if it does not exist.
            with Path(self.filename).open(mode="w", encoding="utf-8") as file:
                json.dump([], file)

            # Return an empty list.
            return []

    def save_message(
        self,
        author: str,
        content: str,
        example: bool = False,
        **kwargs,
    ) -> None:
        message = {
            "type": author,
            "data": {
                "content": content,
                "additional_kwargs": kwargs,
                "example": example,
            },
        }
        loaded_messages = self.get_messages()
        loaded_messages.append(message)
        with Path(self.filename).open(mode="w", encoding="utf-8") as file:
            json.dump(loaded_messages, file, indent=2)

    def set_memory_context(self) -> VectorStoreRetrieverMemory:
        memory = self._init_vector_store_memory()
        loaded_messages = self.get_messages()
        history = iter(messages_from_dict(loaded_messages))
        log(messages_from_dict(loaded_messages))

        for human, ai in zip(history, history):
            memory.save_context({"input": human.content}, {"output": ai.content})

        return memory

    @staticmethod
    def _init_vector_store_memory() -> VectorStoreRetrieverMemory:
        vector_db = Chroma(embedding_function=OpenAIEmbeddings())
        retriever = vector_db.as_retriever(search_kwargs=dict(k=1))
        memory = VectorStoreRetrieverMemory(retriever=retriever, memory_key="history")
        return memory


if __name__ == "__main__":
    bot = ChatBot()

    while True:
        prompt_input = input("Enter a prompt: ")

        if prompt_input == "exit":
            log("Goodbye!")
            break
        else:
            bot.save_message(author="human", content=prompt_input)
            result = bot.engine.predict(input=prompt_input)
            bot.save_message(author="ai", content=result)
            log(result)
