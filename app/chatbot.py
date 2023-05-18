import json
from json import JSONDecodeError
from pathlib import Path
from typing import List

from langchain import LLMChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import VectorStoreRetrieverMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import messages_from_dict, BaseMessage
from langchain.vectorstores import Chroma

from app.llms import gpt3_5_turbo


class ChatBot:
    """
    This is a chatbot that uses the OpenAI API
    to generate responses to prompts.
    """

    def __init__(self):
        template = """
        Your name is SmartBot. 
        
        You are an AI chatbot that is designed to
        help people with their questions to increase their productivity,
        and your tone is formal and professional.
        
        Relevant pieces of previous conversation:
        {history}
        
        (You do not need to use these pieces of information if not relevant)
        
        Current conversation:
        Human: {input}
        SmartBot:
        
        """
        prompt_template = PromptTemplate(
            input_variables=["history", "input"],
            template=template,
        )
        self.filename = "history.json"
        self.embeddings = OpenAIEmbeddings()
        self.vector_db = self.init_vector_db()
        self.memory = self.init_vector_store_memory()
        self.engine = LLMChain(
            llm=gpt3_5_turbo,
            prompt=prompt_template,
            memory=self.memory,
            verbose=True,
        )

    def init_vector_store_memory(self) -> VectorStoreRetrieverMemory:
        retriever = self.vector_db.as_retriever()
        memory = VectorStoreRetrieverMemory(retriever=retriever, memory_key="history")

        messages = self.get_messages()
        history = iter(messages)

        for human, ai in zip(history, history):
            memory.save_context({"input": human.content}, {"output": ai.content})

        return memory

    def init_vector_db(self) -> Chroma:
        vector_db = Chroma(embedding_function=self.embeddings)
        return vector_db

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
        loaded_messages = self._get_messages()
        loaded_messages.append(message)
        with Path(self.filename).open(mode="w", encoding="utf-8") as file:
            json.dump(loaded_messages, file, indent=2)

    def get_messages(self) -> List[BaseMessage]:
        _messages = self._get_messages()
        return messages_from_dict(_messages)

    def _get_messages(self) -> List[dict]:
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
