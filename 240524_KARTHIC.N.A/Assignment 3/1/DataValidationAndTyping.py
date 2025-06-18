from typing import *
def typingmodule():
    '''
    Python is a dynamically typed language - meaning the type of a variable is not fixed at compile time and can change during program's execution.
    So this can lead to errors that only pop uop during program's execution - making it hard to debug.
    The typing module introduced in Python 3.5 addresses this by providing type hints
    Benefits:
    1) Improved Code maintainability
    2) Lesser errors
    '''
    num: int = 5 #Basic Type hints
    #Function annotations
    def add(x: int, y:int)->int:
        return x+y

    #Optional Type (Given type/none)
    def get_name()-> Optional[str]:
        return input("Enter name: ")

    #Any Type 
    def dosomething(arguments: Any) -> Any:
        return "Hi"

    #List Type
    nums: List[int] = [1,2,3]
    names: List[str] = ["Alice", "Bob", "Eve"]

    #Tuple Type
    person: Tuple[int, str] = (1, "Charlie")
    people: Tuple[str, ...] = ("Ben", "Gwen", "Kevin")

    #Dictionary Type
    students: Dict[int, List[str]] = {1:"A"}

    #Union Type (All the stuff we can accept)
    def process_input(value: Union[int, str]) -> Union[float, str]: #New syntax: num: int|str = 5
        if isinstance(value, str):
            return f"Received string: {value}"
        else:
            return float(value)

    #Type aliases
    type Coordinates = List[Tuple[float, float]]
    points: Coordinates = [(1,2), (3, 4)]

#------------------------#
from pydantic import *
class User(BaseModel):
    name: str
    email: EmailStr #Data Validation
    account_id: int
    @field_validator("account_id") #Decorator of Function
    def validate_account_id(cls, value):
        if value<=0:
            raise ValueError(f"account_id must be positive: {value}")
        return value

user = User(name="Harry Potter", email="hp@gmail.com", account_id=7)
print(user.name)

'''
There's also 
user = User.parse_raw(json_str)
userObj = user.json()
userObj = user.dict()
'''