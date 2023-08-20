from typing import Dict, List, Any, Tuple

class HyperParameterSet:
    def __init__(self) -> None:
        self.parameters = {}

    def get(self, parameter_name:str) -> Any:
        raise NotImplementedError("You need to implement the get method for this hyperameter set class")
    
    def set(self, parameter_name:str, value:Any)->None:
        raise NotImplementedError("you need to implement the set method for this hyperparameter set class")
    

    