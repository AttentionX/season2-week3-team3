from termcolor import colored

def assistant(message: str):
    print(colored("[Professor] "+ message, "green"))

def document(message: str):
    print(colored(message, "blue"))