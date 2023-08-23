from typing import Any, Dict, List, Tuple


class HyperParameterTuneSet:
    def __init__(self) -> None:
        self._parameters = {}

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters

    def get_valid_parameters(
        self,
        input_shape: Tuple,
        output_shape: Tuple,
        intput_ranges: Tuple,
        output_ranges: Tuple,
    ) -> dict[str, Any]:
        return self._parameters

    def get(self, parameter_name: str, default: Any | None = None) -> Any:
        if parameter_name not in self._parameters:
            if default is not None:
                return default
            else:
                raise ValueError(
                    f"requested parameter {parameter_name} is not available and has no default"
                )
        else:
            return self._parameters[parameter_name]

    def set(self, parameter_name: str, value: Any) -> None:
        self._parameters[parameter_name] = value
