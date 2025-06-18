from typing import Optional, Union, Any, List, Dict, Tuple, NewType

def add (x: int, y: int) -> int:
    return x+y

try :
    result = add(1,"Harshit")
    print(result)
except TypeError as e:
    print(f"TypeError: {e}")
    
def process_input(value: Union[int, str]) -> Union[float, str]:
    if isinstance(value, str):
        return f"Received string: {value}"
    else:
        return float(value)
    
UserName = NewType('UserName', str)
ProductID = NewType('ProductID', int)

user_name = UserName("Harshit Karnani")
product_id = ProductID(240440)

