from typing import Dict, List, Any, Tuple


class HyperParameterTuneSet:
    def __init__(self) -> None:
        self.parameters = {}

    def get(self, parameter_name: str, default: Any | None = None) -> Any:
        if parameter_name not in self.parameters:
            if default is not None:
                return default
            else:
                raise ValueError(
                    f"requested parameter {parameter_name} is not available and has no default"
                )
        else:
            return self.parameters[parameter_name]

    def set(self, parameter_name: str, value: Any) -> None:
        self.parameters[parameter_name] = value
