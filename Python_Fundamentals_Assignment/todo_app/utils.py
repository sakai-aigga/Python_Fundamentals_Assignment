#functionality like clearing the screen and reading user input
import os

def clr_scr():
    os.system('cls')

def user_inp(prompt):
    return input(prompt)