import os, glob, json, time, random

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
    get_linear_schedule_with_warmup,
)
from peft import get_peft_model, LoraConfig, PeftModel, PeftConfig, PeftMixedModel
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import autocast, GradScaler
import torch.nn.functional as F

from .utils import system_prompt, user_message_template1, user_message_template2, user_message_template3
from .datatypes import *

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
        task = random.choice(self.all_tasks)
        examples = task["examples"]
        
        # select self.num_samples_per_task examples
        sampled_examples = random.sample(examples, self.num_samples_per_task)
        
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
        
        # concat eos
        eos = self.solver.tokenizer.eos_token_id
        if target_ids[-1] != eos:
            target_ids = torch.cat([target_ids, torch.tensor([eos], dtype=torch.long, device=target_ids.device)])
        
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
        checkpoint_save_path=None,
        enable_gradient_checkpointing=False,
        sep_str="\n",
    ):
        """
        Args:
            token (str): a huggingface token for restricted models such as llama3
        """
        config_path = "artifacts/config/config.yml"
        model_id = "Qwen/Qwen3-4B"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.checkpoint_save_path = checkpoint_save_path if checkpoint_save_path else "artifacts/default"

        # Configure the BitsAndBytes settings for 4-bit quantization to reduce memory usage
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,  # Enable 4-bit quantization
            bnb_4bit_use_double_quant=True,  # Use double quantization for improved precision
            bnb_4bit_quant_type="nf4",  # Specify the quantization type
            bnb_4bit_compute_dtype=torch.float16,  # Set the computation data type
        )
        
        model_args = {
            "pretrained_model_name_or_path": model_id,
            "trust_remote_code": True,  # Allow the model to use custom code from the repository
            "quantization_config": bnb_config,  # Apply the 4-bit quantization configuration
            "attn_implementation": "sdpa",  # Use scaled-dot product attention for better performance
            "torch_dtype": torch.float16,  # Set the data type for the model
            "use_cache": False,  # Disable caching to save memory
            "token": token,
            "device_map": "auto",  # Automatically map the model to available devices
        }
        cache_dir = os.getenv("TRANSFORMERS_CACHE")
        if cache_dir:
            print(f"Using cache dir: {cache_dir}")
            model_args["cache_dir"] = cache_dir
        else:
            print("No cache dir found, using default cache location.")
        
        self.model: Union[PreTrainedModel, PeftModel, PeftMixedModel] = AutoModelForCausalLM.from_pretrained(
            **model_args,
        )
        
        if enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            self.model.config.use_cache = False
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()

        tokenizer_args = {
            "pretrained_model_name_or_path": model_id,
            "token": token,
        }
        if cache_dir:
            tokenizer_args["cache_dir"] = cache_dir
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(**tokenizer_args)
        self.tokenizer.bos_token_id = 151643 # Default for Qwen3

        self.pixel_ids = [
            self.tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(10)
        ]
        
        self.sep_str = sep_str
        self.sep_token_id = self.tokenizer.encode(self.sep_str, add_special_tokens=False)[0]
        
        
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
            if idx == self.sep_token_id:
                if len(row) > 0:
                    grid.append(row.copy())
                    row.clear()
            else:
                if idx == self.tokenizer.eos_token_id:
                    break
                row.append(inv_map.get(idx, 0))
        if len(row) > 0:
            grid.append(row)
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
            ids.append(self.sep_token_id)
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
                f"output:\n{self.grid_to_str(ex['output'])}\n"
            )

        test_msg = (
            f"\n{user_message_template2}\n"
            f"{user_message_template3}\n"
            f"input:\n{self.grid_to_str(test_input_grid)}\n"
        )
        messages.append({"role": "user", "content": msg + test_msg})

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
        
    def dynamic_collate(self, batch: List[dict]) -> dict:
        input_ids = [item["input_ids"] for item in batch]
        target_ids = [item["target_ids"] for item in batch]
        
        padded_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id, padding_side="left")
        padded_target_ids = torch.nn.utils.rnn.pad_sequence(target_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id, padding_side="left")
        
        return {
            "input_ids": padded_input_ids,
            "target_ids": padded_target_ids,
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

        inp = torch.cat([prompt_ids, target_ids], dim=1)
        attn_mask = inp.ne(self.tokenizer.pad_token_id).long()
        
        labels = inp.clone()
        labels[:, : prompt_ids.size(1)] = -100
        labels[inp == self.tokenizer.pad_token_id] = -100

        outputs = self.model(input_ids=inp, labels=labels, attention_mask=attn_mask)
        return outputs.loss
    
    def set_peft_model(self):
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=16, 
            lora_alpha=32,  
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        self.model = get_peft_model(self.model, peft_config)

    def train(
        self, 
        train_dataset: ARCDataset,
        num_epochs: int = 5,
        learning_rate: float = 5e-5,
        gradient_accumulation_steps: int = 4,
        batch_size: int = 2,
        warmup_rate: float = 0.1,
        checkpoint_name_to_resume_from: str = None,
    ):
        """
        Train a model with train_dataset.
        """

        dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            pin_memory=True, 
            collate_fn=lambda b: self.dynamic_collate(b),
        )
             
        start_epoch = 0
        global_step = 0
           
        start_time = time.time()
        print(f"Fine-tuning Lora adapters for epochs {start_epoch+1} to {num_epochs}")
        self.set_peft_model()
        self.model.print_trainable_parameters()
        
        total_steps = num_epochs * len(dataloader) / gradient_accumulation_steps
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * warmup_rate),
            num_training_steps=total_steps,
        )

        if checkpoint_name_to_resume_from:
            start_epoch, global_step = self.load_checkpoint(checkpoint_name_to_resume_from, optimizer, scheduler)
            print(f"Resuming from epoch {start_epoch}, global step {global_step}")

        scaler = GradScaler()

        prev_epoch_time = time.time()
        self.model.train()
        for epoch in range(start_epoch, num_epochs):
            running = 0.0
            for step, batch in enumerate(dataloader):
                global_step += 1
                input_ids = batch["input_ids"].to(self.device)
                target_ids = batch["target_ids"].to(self.device)

                with autocast(device_type="cuda"):
                    loss = self.seq2seq_loss(input_ids, target_ids) / gradient_accumulation_steps
                scaler.scale(loss).backward()
                # loss.backward()

                if global_step % gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    scaler.step(optimizer)
                    # full_optimizer.step()
                    scheduler.step()
                    scaler.update()
                    optimizer.zero_grad()

                running += loss.item()
                if step % 10 == 0:
                    print(f"[FT E{epoch+1}] step {step} loss {loss.item():.4f}")

            print(f"[FT E{epoch+1}] avg-loss {running/len(dataloader):.4f}")
            self.save_model(
                checkpoint_name=f"checkpoint-{epoch+1}",
                optimizer=optimizer,
                scheduler=scheduler,
                start_epoch=epoch+1,
                global_step=global_step,
            )
            
            epoch_time = time.time()
            print(f"Epoch {epoch+1} completed in {epoch_time - prev_epoch_time:.2f} seconds")
            prev_epoch_time = epoch_time
        
        self.save_model(
            checkpoint_name="checkpoint-final",
            optimizer=optimizer,
            scheduler=scheduler,
            start_epoch=num_epochs,
            global_step=global_step,
        )
        self.model.eval()
        end_time = time.time()
        print(f"Training completed in {end_time - start_time:.2f} seconds")
    
    def load_checkpoint(self, checkpoint_name: str, optimizer: AdamW, scheduler: LambdaLR) -> tuple:
        checkpoint_path = os.path.join(self.checkpoint_save_path, checkpoint_name)
        self.model = PeftModel.from_pretrained(self.model, checkpoint_path, is_trainable=True).to(self.device)
        
        opt_path = os.path.join(checkpoint_path, "optimizer.pth")
        if optimizer is not None and os.path.isfile(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location=self.device))
        
        scheduler_path = os.path.join(checkpoint_path, "scheduler.pth")
        if scheduler is not None and os.path.isfile(scheduler_path):
            scheduler.load_state_dict(torch.load(scheduler_path, map_location=self.device))
        
        state_path = os.path.join(checkpoint_path, "training_state.json")
        start_epoch = 0
        start_global_step = 0
        if os.path.isfile(state_path):
            with open(state_path, "r") as f:
                state = json.load(f)
                start_epoch = state.get("start_epoch", 0)
                start_global_step = state.get("global_step", 0)
        return start_epoch, start_global_step
        
    def save_model(
        self, 
        checkpoint_name: str = "checkpoint-final-default",
        optimizer: AdamW = None,
        scheduler: LambdaLR = None,
        start_epoch: int = None,
        global_step: int = None,
    ):
        checkpoint_path = os.path.join(self.checkpoint_save_path, checkpoint_name)
        os.makedirs(checkpoint_path, exist_ok=True)
        self.model.save_pretrained(checkpoint_path)
        
        if optimizer is not None:
            torch.save(optimizer.state_dict(), os.path.join(checkpoint_path, "optimizer.pth"))
        if scheduler is not None:
            torch.save(scheduler.state_dict(), os.path.join(checkpoint_path, "scheduler.pth"))
            
        state = dict()
        if start_epoch is not None:
            state["start_epoch"] = start_epoch
        if global_step is not None:
            state["global_step"] = global_step
        if state:
            with open(os.path.join(checkpoint_path, "training_state.json"), "w") as f:
                json.dump(state, f, indent=2)
        
        model_info = {
            'base_model': {
                'base': self.model.config._name_or_path,
                'type': self.model.config.model_type,
                'hidden_size': int(self.model.config.hidden_size),
                'vocab_size': int(self.tokenizer.vocab_size)
            }
        }
        with open(os.path.join(checkpoint_path,"model_config.json"), "w") as f: 
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
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            max_new_tokens=150,
            do_sample=True,   
        )
        with torch.no_grad(), autocast(device_type="cuda"):
            output_ids = self.model.generate(
                input_ids=input_ids,
                generation_config=config,
                attention_mask=attn_mask,
            ).squeeze(0).cpu()
            
            
        N_prompt = input_ids.size(1)
        output_ids = output_ids[N_prompt:].tolist() # generated portion
        
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
            grid = np.array(self.parse_grid(output_ids))
            # grid = grid[:x, :y]
            
        except Exception as e:
            # print(f"Error parsing grid: {e}")
            grid = np.random.randint(0, 10, (x, y))

        return grid

    def prepare_evaluation(
        self,
        checkpoint_name: str = "checkpoint-final-default",
    ):
        """
        Load pretrained weight, make model eval mode, etc.
        """
        checkpoint_path = os.path.join(self.checkpoint_save_path, checkpoint_name)
        
        # LoRA 어댑터 로드
        try:
            peft_config = PeftConfig.from_pretrained(checkpoint_path)
            self.model = PeftModel.from_pretrained(
                self.model,
                checkpoint_path,
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




