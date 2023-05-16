from dotenv import dotenv_values

from app.chatbot import ChatBot

config = dotenv_values(".env")
log = print

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
