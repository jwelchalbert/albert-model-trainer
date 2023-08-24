import importlib.util
import inspect
import os

from albert_model_trainer.base.model import ModelTrainer

model_file_dir = os.path.dirname(__file__)


def get_model_trainers(directory, instantiate_all=True) -> list[ModelTrainer]:
    model_trainers = []

    # Iterate through every python file in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".py"):
            # Create module name from file name
            module_name = filename[:-3]

            # Create module spec
            spec = importlib.util.spec_from_file_location(
                module_name, os.path.join(directory, filename)
            )

            # Load module
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Iterate through objects in module
            for _, obj in inspect.getmembers(module):
                # Check if object is a class and if it inherits from 'ModelTrainer'
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, ModelTrainer)
                    and obj != ModelTrainer
                ):
                    # Add object to list
                    model_trainers.append(obj)

    if instantiate_all:
        model_trainers = [x() for x in model_trainers]

    return model_trainers


def get_all_model_trainers():
    return get_model_trainers(model_file_dir)
