import tiktoken
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


tokenizer = tiktoken.get_encoding("gpt2")
this_dir = os.path.abspath(os.path.dirname(__file__))
file_path = os.path.join(this_dir, "", "the-verdict.txt")


with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()

total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
print("Characters: ", total_characters)
print("Tokens: ", total_tokens)
