"""String Utils"""

def is_number(word):
    return word.replace('.','').replace(',','').isdecimal()