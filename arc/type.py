from typing import List, Union, Literal, Any, TypedDict
import torch

Grid = List[List[int]]

class ExampleDict(TypedDict):
    input: Grid
    output: Grid

class TestExampleDict(TypedDict):
    input: Grid
    
class TaskDict(TypedDict):
    file_path: str
    task_id: str
    examples: List[dict]

class DataPointDict(TypedDict):
    task: str
    train: List[ExampleDict]
    test: List[TestExampleDict]
    
class FormattedPrompt(TypedDict):
    input_ids: torch.Tensor
    input: Grid
    train: List[ExampleDict]