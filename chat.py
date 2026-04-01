import os
from groq import Groq

from dotenv import load_dotenv
load_dotenv()  # reads variables from a .env file and sets them in os.environ

# in python, class names are CamelCase
# non-class names (functions/variables) are in snake_case
class Chat:
    client = Groq()
    def __init__(self):
        self.messages = [
                {
                    # most important content for sys prompt is length of response
                    "role": "system",
                    "content": "Write the output in 1-2 sentences. Talk like pirate."
                },
            ]
    def send_message(self, message):
        self.messages.append(
            {
                # system: never changes; user: changes a lot;
                'role': 'user',
                'content': message
            }
        )
        chat_completion = self.client.chat.completions.create(
            messages=self.messages,
            model="llama-3.1-8b-instant",
        )
        result = chat_completion.choices[0].message.content
        self.messages.append({
            'role': 'assistant',
            'content': result,
        })
        return result


if __name__ == '__main__':
    import readline
    chat = Chat()
    try:
        while True:
            user_input = input('chat> ')
            response = chat.send_message(user_input)
            print(response)
    except KeyboardInterrupt:
        print()
        pass