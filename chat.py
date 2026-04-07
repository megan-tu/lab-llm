import os
from groq import Groq

from dotenv import load_dotenv
load_dotenv()  # reads variables from a .env file and sets them in os.environ

# in python, class names are CamelCase
# non-class names (functions/variables) are in snake_case
class Chat:
    '''
    >>> def monkey_input(prompt, user_inputs=['Hello, I am monkey.', 'Goodbye.']):
    ...     try:
    ...         user_input = user_inputs.pop(0)
    ...         print(f'{prompt}{user_input}')
    ...         return user_input
    ...     except IndexError:
    ...         raise KeyboardInterrupt
    >>> import builtins
    >>> builtins.input = monkey_input
    >>> repl(temperature=0.0)
    chat> Hello, I am monkey.
    Arrr, ye be a mischievous little monkey, eh? Yer chatterin' be music to me ears, matey!
    chat> Goodbye.
    Farewell, me scurvy monkey friend, may the winds o' fortune blow in yer favor!
    <BLANKLINE>

    >>> chat = Chat()
    >>> chat.send_message('my name is bob', temperature=0.0)
    'Arrr, ye be Bob, eh? Yer name be known to me now, matey.'
    >>> chat.send_message('what is my name?', temperature=0.0)
    "Ye be askin' about yer own name, eh? Yer name be... Bob, matey!"
    '''
    client = Groq()
    def __init__(self):
        self.messages = [
                {
                    # most important content for sys prompt is length of response
                    "role": "system",
                    "content": "Write the output in 1-2 sentences. Talk like pirate."
                },
            ]
    def send_message(self, message, temperature=0.8):
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
            temperature=temperature,
        )
        result = chat_completion.choices[0].message.content
        self.messages.append({
            'role': 'assistant',
            'content': result,
        })
        return result

def repl(temperature=0.0):
    import readline
    chat = Chat()
    try:
        while True:
            user_input = input('chat> ')
            response = chat.send_message(user_input, temperature=temperature)
            print(response)
    except (KeyboardInterrupt, EOFError):
        print()

if __name__ == '__main__':
    repl(temperature=0.0)