from enum import Enum

class SYNTAXTYPE(Enum):
    annotation = 10
    basictype = 20
    boolean = 30
    decimalfloatingpoint = 40
    decimalinteger = 50
    hexinteger = 60
    identifier = 70 
    keyword = 80
    modifier = 90
    null = 100
    octalinteger = 110
    operator = 120
    separator = 130
    string = 140 
    unknown = 0