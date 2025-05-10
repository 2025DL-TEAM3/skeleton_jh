import os, glob, json, time

from transformers import GenerationConfig
import torch
from typing import List, Union, Literal, Any, TypedDict
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from pprint import pprint


from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
    PreTrainedModel,
)
from peft import get_peft_model, LoraConfig, PeftModel, PeftConfig, PeftMixedModel
from torch.optim import AdamW
import torch.nn.functional as F

from .utils import system_prompt, user_message_template1, user_message_template2, user_message_template3
from .type import *

class ARCDataset(Dataset):
    def __init__(
        self, 
        dataset_path: str, 
        solver: "ARCSolver",
        num_samples_per_task: int = 4,
        num_steps_per_task: int = 50,
    ):
        """
        Args:
            dataset_path (str): Path to the dataset directory
            solver (ARCSolver): Instance of the ARCSolver class
        """
        self.dataset_path = dataset_path
        self.solver = solver
        self.all_tasks: list[TaskDict] = []
        self.load_dataset()
        
        self.num_samples_per_task = num_samples_per_task
        self.num_steps_per_task = num_steps_per_task
        self.total_num_steps = len(self.all_tasks) * num_steps_per_task
        print(f"Total number of steps: {self.total_num_steps}")
    
    def load_dataset(self):
        json_file_paths = glob.glob(f"{self.dataset_path}/*.json")
        if not json_file_paths:
            raise ValueError(f"No JSON files found in {self.dataset_path}")
        
        print(f"Found {len(json_file_paths)} JSON files.")
        
        for json_file_path in json_file_paths:
            task_id = os.path.basename(json_file_path).split(".")[0]
            try:
                with open(json_file_path, 'r') as f:
                    task_json = json.load(f)
                    if isinstance(task_json, list) and len(task_json) > 0:
                        self.all_tasks.append({
                            "file_path": json_file_path,
                            "task_id": task_id,
                            "examples": task_json
                        })
            except Exception as e:
                print(f"Error loading file: {json_file_path} - {e}")
        
        if not self.all_tasks:
            raise ValueError("No valid examples found in JSON files.")
        
        print(f"Successfully loaded {len(self.all_tasks)} JSON files.")
    
    def __len__(self):
        return self.total_num_steps
    
    def __getitem__(self, idx):
        task = np.random.choice(self.all_tasks)
        examples = task["examples"]
        
        # select self.num_samples_per_task examples
        sampled_examples = np.random.choice(examples, self.num_samples_per_task, replace=False)
        
        train_examples = sampled_examples[:self.num_samples_per_task - 1]
        test_example = sampled_examples[self.num_samples_per_task - 1]
        
        datapoint: DataPointDict = {
            "task": task["task_id"],
            "train": train_examples,
            "test": [
                {
                    "input": test_example["input"],
                }
            ]
        }
        
        input_ids = self.solver.datapoint_to_input(datapoint, keep_batch_dim=False)
        
        target_ids = torch.tensor(
            self.solver.format_grid(test_example["output"]), dtype=torch.long
        )
        
        # Note: it might raise runtime error if they are variable length
        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
        }
        
class ARCSolver:
    """
    You should implement a `Solver` class for the project.
    """

    def __init__(
        self, 
        token=None,
        sep_str="\n",
    ):
        """
        Args:
            token (str): a huggingface token for restricted models such as llama3
        """
        config_path = "artifacts/config/config.yml"
        model_id = "Qwen/Qwen3-4B"

        # Configure the BitsAndBytes settings for 4-bit quantization to reduce memory usage
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,  # Enable 4-bit quantization
            bnb_4bit_use_double_quant=True,  # Use double quantization for improved precision
            bnb_4bit_quant_type="nf4",  # Specify the quantization type
            bnb_4bit_compute_dtype=torch.float16,  # Set the computation data type
        )
        self.model: Union[PreTrainedModel, PeftModel, PeftMixedModel] = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True, # Allow the model to use custom code from the repository
            quantization_config=bnb_config, # Apply the 4-bit quantization configuration
            attn_implementation='sdpa', # Use scaled-dot product attention for better performance
            torch_dtype=torch.float16, # Set the data type for the model
            use_cache=False, # Disable caching to save memory
            device_map='cuda:0', # Automatically map the model to available devices (e.g., GPUs)
            token=token
        )

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_id, token=token)

        self.pixel_ids = [
            self.tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(10)
        ]
        self.sep_str = sep_str
        self.sep_token = self.tokenizer.encode(self.sep_str, add_special_tokens=False)[0]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        

    def parse_grid(self, ids: List[int]) -> Grid:
        """
        Parse LLM generated sequence into ARC grid format

        Args:
            ids (List[int]): LLM generated token list

        Returns:
            grid (Grid): parsed 2D grid
        """
        grid = []
        row = []
        inv_map = {k: i for i, k in enumerate(self.pixel_ids)}
        
        for idx in ids:
            if idx == self.sep_token:
                if len(row) > 0:
                    grid.append(row.copy())
                    row.clear()
            else:
                row.append(inv_map.get(idx, 0))
        return grid

    # TODO: is col_sep needed?
    def format_grid(self, grid: Grid) -> List[int]:
        """
        Format 2D grid into LLM input tokens

        Args:
            grid (Grid): 2D grid

        Returns:
            ids (List[int]): Token list for LLM
        """
        ids = []

        for row in grid:
            for col in row:
                ids.append(self.pixel_ids[col])
            ids.append(self.sep_token)
        return ids

    def grid_to_str(self, grid: Grid) -> str:
        """Convert 2D grid to string format
        Args:
            grid (Grid): 2D grid
                [
                    [1, 2],
                    [3, 4]
                ]
        Returns:
            str: String representation of the grid
                12 
                34
        """
        return self.sep_str.join("".join(str(c) for c in row) for row in grid)

    def format_prompt(self, datapoint: DataPointDict) -> FormattedPrompt:
        """
        Args:
            datapoint: A dictionary containing the training examples and test input
                {
                    "train": [
                        {"input": [[1,2],[3,4]], "output": [[4,5],[6,7]]},
                        ...
                    ],
                    "test": [
                        {"input": [[0,1],[2,3]]}
                    ]
                }
        """
        train_examples = datapoint["train"]
        test_input_grid = datapoint["test"][0]["input"]

        messages = [
            {"role": "system", "content": system_prompt},
        ]
        msg = f"{user_message_template1}\n"
        for ex in train_examples:
            msg += (
                f"input:\n{self.grid_to_str(ex['input'])}\n"
                f"output:\n{self.grid_to_str(ex['output'])}"
            )
        messages.append({"role": "user", "content": msg})

        test_msg = (
            f"{user_message_template2}\n"
            f"input:\n{self.grid_to_str(test_input_grid)}\n"
            f"{user_message_template3}"
        )
        messages.append({"role": "user", "content": test_msg})
        
        # type: torch.Tensor
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,   # assistant 턴 여는 토큰 추가
            enable_thinking=False,         # THINKING MODE ON
            tokenize=True,
            return_tensors="pt",
        ) # (1, L)
        
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long)

        return {
            "input_ids": input_ids[0], # (L, )
            "input": test_input_grid,
            "train": train_examples,
        }

    def seq2seq_loss(self, prompt_ids: torch.LongTensor, target_ids: torch.LongTensor) -> torch.Tensor:
        """
        prompt_ids  : [B, L]  ← 문제 설명(프롬프트)
        target_ids  : [B, T]  ← 정답 토큰 시퀀스
        ------------------------------------------
        inp   = [B, L+T]      ←  [프롬프트][정답] 한줄로 연결
        labels= same shape     (프롬프트 부분은 -100으로 마스킹)
        ------------------------------------------
        model(input_ids=inp, labels=labels)  →  .loss
        """
        
        eos = self.tokenizer.eos_token_id
        
        if not torch.all(target_ids[:, -1] == eos):
            eos_tensor = torch.full((target_ids.size(0), 1), eos, dtype=target_ids.dtype, device=target_ids.device)
            target_ids = torch.cat([target_ids, eos_tensor], dim=1)
        
        inp = torch.cat([prompt_ids, target_ids], dim=1)
        
        labels = inp.clone()
        labels[:, : prompt_ids.size(1)] = -100

        
        outputs = self.model(input_ids=inp, labels=labels)
        return outputs.loss

    def train(
        self, 
        train_dataset: ARCDataset,
        epochs: int = 5,
        learning_rate: float = 5e-5,
        gradient_accumulation_steps: int = 4,
    ):
        """
        Train a model with train_dataset.
        """
        # Set the model to training mode
        self.model.train()

        
        # LoRA 설정 - Attention 모듈에 적용
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=8,  # LoRA 행렬의 랭크
            lora_alpha=32,  # LoRA 스케일링 파라미터
            lora_dropout=0.1,
            # Qwen2 모델의 트랜스포머 주요 가중치 타겟팅
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        
        # 모델에 LoRA 적용
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()  # 학습 가능한 파라미터 비율 출력

        dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

        optimizer  = AdamW(self.model.parameters(), lr=learning_rate)
        
        # 메모리 효율성을 위한 그래디언트 누적 설정
        
        # Training loop
        global_step = 0
        start_time = time.time()
        for epoch in range(epochs):
            running = 0.0
            optimizer.zero_grad() 
            
            for step, batch in enumerate(dataloader):
                global_step += 1
                
                input_ids = batch["input_ids"].to(self.device)
                target_ids = batch["target_ids"].to(self.device)
                
                loss = self.seq2seq_loss(input_ids, target_ids)

                # 역전파
                loss.backward()
                
                # 그래디언트 누적 후 최적화
                if global_step % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                
                # 손실 기록
                running += loss.item() * gradient_accumulation_steps
                
                # 로그 출력
                if step % 10 == 0:
                    print(f"[E{epoch+1}] step {step} loss {loss.item()*gradient_accumulation_steps:.4f}")
                    
            # 에포크 종료 시 평균 손실 출력
            print(f"Epoch {epoch+1} avg-loss {(running/len(dataloader)):.4f}")
            print(f"Saving model for epoch {epoch+1}...")
            self.save_model(f"artifacts/checkpoint-{epoch+1}")
        end_time = time.time()
        print(f"Training completed in {end_time - start_time:.2f} seconds")

        self.save_model("artifacts/checkpoint-final")
        self.model.eval()  # Set model back to evaluation mode
    
    def save_model(self, data_path: str = None):
        if data_path is None:
            data_path = "artifacts/checkpoint-final_tmp"
        os.makedirs(data_path, exist_ok=True)
        self.model.save_pretrained(data_path)
        model_info = {
            'base_model': {
                'name': self.model.config._name_or_path,
                'type': self.model.config.model_type,
                'hidden_size': int(self.model.config.hidden_size),
                'vocab_size': int(self.tokenizer.vocab_size)
            }
        }
        with open(os.path.join(data_path,"model_config.json"), "w") as f: 
            json.dump(model_info, f, indent=2)

    def predict(self, examples: List[ExampleDict], questions_input: Grid) -> Grid:
        """
        A single example of test data is given.
        You should predict 2D grid (Grid or np.ndarray)

        Args:
            examples (List[ExampleDict]): List of training examples,
                each list element is a dictionary that contains "input" and "output"
                for example,
                [
                    {
                        "input": [[1,2],[3,4]],
                        "output": [[4,5],[6,7]],
                    },
                    {
                        "input": [[0,1],[2,3]],
                        "output": [[3,4],[5,6]],
                    }
                ]
            questions_input (Grid): A 2d grid,
                which is a input for a given question
        Returns:
            output (Grid): A 2d grid,
                which is the output of given input question.
        """
        datapoint: DataPointDict = {
            "train": examples,
            "test": [
                {
                    "input": questions_input
                }
            ]
        }

        input_ids = self.datapoint_to_input(datapoint, keep_batch_dim=True).to(self.device)
        
        attn_mask = torch.ones_like(input_ids)

        # Qwen3 모델은 더 많은 토큰을 생성할 수 있도록 설정
        config = GenerationConfig(
            temperature=0.7, top_p=0.8, top_k=20,    # 권장 값
            bos_token_id=151643,  # Qwen3 모델의 내부 기본값 명시적 사용
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            max_new_tokens=150,
            do_sample=True,   
        )

        output = self.model.generate(
            input_ids=input_ids,
            generation_config=config,
            attention_mask=attn_mask,
        ).squeeze().cpu()
        N_prompt = input_ids.size(1)

        output = output[N_prompt:].tolist()
        prompt = self.format_prompt(datapoint)
        train_input = np.array(prompt['train'][0]['input'])
        train_output = np.array(prompt['train'][0]['output'])
        test_input = np.array(prompt['input'])

        # LLM-generated grid may have wrong shape
        # So adjust shape by input-output pairs
        if train_input.shape == train_output.shape:
            x, y = test_input.shape
        else:
            x = (train_output.shape[0] * test_input.shape[0] // train_input.shape[0])
            y = (train_output.shape[1] * test_input.shape[1] // train_input.shape[1])

        try:
            grid = np.array(self.parse_grid(output))
            # grid = grid[:x, :y]
            
        except Exception as e:
            grid = np.random.randint(0, 10, (x, y))

        return grid

    def prepare_evaluation(self):
        """
        Load pretrained weight, make model eval mode, etc.
        """
        # LoRA 어댑터 로드
        try:
            peft_config = PeftConfig.from_pretrained("artifacts/checkpoint-final")
            self.model = PeftModel.from_pretrained(
                self.model,
                "artifacts/checkpoint-final",
                is_trainable=False
            )
            print("Loaded LoRA adapter")
        except Exception as e:
            print(f"No LoRA adapter found or incompatible: {e}")
            
        self.model.eval()
        
    def datapoint_to_input(self, datapoint: DataPointDict, keep_batch_dim: bool=False) -> torch.Tensor:
        """
        Convert a datapoint to input format for the model.
        """
        prompt = self.format_prompt(datapoint)
        if isinstance(prompt["input_ids"], torch.Tensor):
            input_ids = prompt["input_ids"]
        else:
            input_ids = torch.tensor(prompt["input_ids"], dtype=torch.long)
        
        if keep_batch_dim:
            input_ids = input_ids.unsqueeze(0)
        return input_ids

if __name__ == "__main__":
    solver = ARCSolver()




