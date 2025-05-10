import os, glob, json

from transformers import GenerationConfig
import torch
from typing import List, Union
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


from .utils import system_prompt, user_message_template1, user_message_template2, user_message_template3
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
    PreTrainedModel,
)
from peft import get_peft_model, LoraConfig, PeftModel, PeftConfig, PeftMixedModel
from torch.optim import AdamW

class ARCDataset(Dataset):
    def __init__(self, dataset_path: str, num_samples_per_task: int = 4):
        """
        Args:
            dataset_path (str): Path to the dataset directory
            solver (ARCSolver): Instance of the ARCSolver class
        """
        self.dataset_path = dataset_path
        self.all_examples = []
        self.load_dataset()
        self.num_samples_per_task = num_samples_per_task
        self.total_num_samples = len(self.all_examples) * num_samples_per_task
    
    def load_dataset(self):
        json_file_paths = glob.glob(f"{self.dataset_path}/*.json")
        if not json_file_paths:
            raise ValueError(f"No JSON files found in {self.dataset_path}")
        
        print(f"Found {len(json_file_paths)} JSON files.")
        
        for json_file_path in json_file_paths:
            task_id = os.path.basename(json_file_path).split(".")[0]
            try:
                with open(json_file_path, 'r') as f:
                    examples = json.load(f)
                    if isinstance(examples, list) and len(examples) > 0:
                        self.all_examples.append({
                            "file_path": json_file_path,
                            "task_id": task_id,
                            "examples": examples
                        })
            except Exception as e:
                print(f"Error loading file: {json_file_path} - {e}")
        
        if not self.all_examples:
            raise ValueError("No valid examples found in JSON files.")
        
        print(f"Successfully loaded {len(self.all_examples)} JSON files.")
    
    def __len__(self):
        return self.total_num_samples
    
    def __getitem__(self, idx):
        task = np.random.choice(self.all_examples)
        examples = task["examples"]
        
        # select self.num_samples_per_task examples
        sampled_examples = np.random.choice(examples, self.num_samples_per_task, replace=False)
        
        train_examples = sampled_examples[:self.num_samples_per_task - 1]
        test_example = sampled_examples[self.num_samples_per_task - 1]
        
        datapoint = {
            "task": task["task_id"],
            "train": train_examples,
            "test": [
                {
                    "input": test_example["input"],
                    "output": test_example["output"]
                }
            ]
        }
        return datapoint
        
class ARCSolver:
    """
    You should implement a `Solver` class for the project.
    """

    def __init__(self, token=None):
        """
        Args:
            token (str): a huggingface token for restricted models such as llama3
        """
        config_path = "artifacts/config/config.yml"
        model_id = "Qwen/Qwen3-4B"
        print(f"Using device: {torch.device("cuda:0" if torch.cuda.is_available() else "cpu")}")

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
        self.sep = self.tokenizer.encode("\n", add_special_tokens=False)[0]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        

    def parse_grid(self, ids: List[int]) -> List[List[int]]:
        """
        Parse LLM generated sequence into ARC grid format

        Args:
            ids (List[int]): LLM generated token list

        Returns:
            grid (List[List[int]]): parsed 2D grid
        """
        grid = []
        row = []
        inv_map = {k: i for i, k in enumerate(self.pixel_ids)}
        
        for idx in ids:
            if idx == self.sep:
                if len(row) > 0:
                    grid.append(row.copy())
                    row.clear()
            else:
                row.append(inv_map.get(idx, 0))
        return grid

    def format_grid(self, grid: List[List[int]]) -> List[int]:
        """
        Format 2D grid into LLM input tokens

        Args:
            grid (List[List[int]]): 2D grid

        Returns:
            ids (List[int]): Token list for LLM
        """
        ids = []

        for row in grid:
            for col in row:
                ids.append(self.pixel_ids[col])
            ids.append(self.sep)
        return ids

    def grid_to_str(self, grid: List[List[int]]) -> str:
        # 줄마다 012… 형식, 줄 끝에 \n
        return "\n".join("".join(str(c) for c in row) for row in grid)

    def format_prompt(self, datapoint):
        """
        Args:
            datapoint (dict): A dictionary containing the training examples and test input
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
        test_inp      = datapoint["test"][0]["input"]

        # ChatML 메시지 배열
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        messages.append(f"{user_message_template1}\n")
        # 학습 예제 3개
        for ex in train_examples:
            msg = (
                f"input:\n{self.grid_to_str(ex['input'])}\n"
                f"output:\n{self.grid_to_str(ex['output'])}"
            )
            messages.append({"role": "user", "content": msg})

        test_msg = (
            f"{user_message_template2}\n"
            f"input:\n{self.grid_to_str(test_inp)}\n"
            f"{user_message_template3}"
        )
        messages.append({"role": "user", "content": test_msg})

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,   # assistant 턴 여는 토큰 추가
            enable_thinking=False,         # THINKING MODE ON
            tokenize=True,
            return_tensors="pt",
        ).to(self.device)

        return {
            "input_ids": input_ids[0],    # [seq] → [L]
            "input": test_inp,
            "train": train_examples,
        }


    def stepwise_loss(self, prompt_ids, target_ids, use_cache=False):
        """
        prompt_ids : [1, L]  (프롬프트)
        target_ids : [1, T]  (정답 토큰 시퀀스)
        returns CE loss 합계 (scalar)
        """
        device = prompt_ids.device
        loss_total = 0.0
        past_kv = None          # KV-cache (use_cache=True 일 때만)

        for i in range(target_ids.size(1)):
            outputs = self.model(
                input_ids=prompt_ids if past_kv is None else target_ids[:, i-1:i],
                past_key_values=past_kv,
                use_cache=use_cache
            )
            logits = outputs.logits[:, -1, :]          # 마지막 토큰 로짓
            loss_i = F.cross_entropy(logits, target_ids[:, i])
            loss_total += loss_i

            past_kv = outputs.past_key_values if use_cache else None
            # teacher token → 다음 입력
            prompt_ids = torch.cat([prompt_ids, target_ids[:, i:i+1]], dim=1)

        return loss_total

    def seq2seq_loss(self, prompt_ids, target_ids):
        """
        prompt_ids  : [B, L]  ← 문제 설명(프롬프트)
        target_ids  : [B, T]  ← 정답 토큰 시퀀스
        ------------------------------------------
        inp   = [B, L+T]      ←  [프롬프트][정답] 한줄로 연결
        labels= same shape     (프롬프트 부분은 -100으로 마스킹)
        ------------------------------------------
        model(input_ids=inp, labels=labels)  →  .loss
        """
        # 1) 프롬프트와 정답을 이어 붙인다
        inp = torch.cat([prompt_ids, target_ids], dim=1)

        # 2) labels는 inp와 동일하되, 프롬프트 토큰에는 -100 (loss 무시)
        labels = inp.clone()
        labels[:, : prompt_ids.size(1)] = -100

        # 3) HuggingFace 모델 호출
        outputs = self.model(input_ids=inp, labels=labels)
        return outputs.loss



    def train(self, train_dataset):
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

        dataset = ARCDataset(train_dataset, self.tokenizer, self)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

        optimizer  = AdamW(self.model.parameters(), lr=5e-5)
        
        
        # 메모리 효율성을 위한 그래디언트 누적 설정
        gradient_accumulation_steps = 4
        
        # Training loop
        epochs = 5
        global_step = 0
        for epoch in range(epochs):
            running = 0.0
            optimizer.zero_grad()  # 에포크 시작 시 그래디언트 초기화
            
            for step, batch in enumerate(dataloader):
                global_step += 1
                # 배치 데이터를 디바이스로 이동
                input_ids = batch["input_ids"].to(self.device)
                target_ids = batch["target_ids"].to(self.device)
                
                # loss = self.stepwise_loss(input_ids, target_ids)
                loss = self.seq2seq_loss(input_ids, target_ids)

                # 역전파
                loss.backward()
                
                # 그래디언트 누적 후 최적화
                if global_step % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step();  optimizer.zero_grad()
                
                # 손실 기록
                running += loss.item() * gradient_accumulation_steps
                
                # 로그 출력
                if step % 10 == 0:
                    print(f"[E{epoch+1}] step {step} loss {loss.item()*gradient_accumulation_steps:.4f}")
                    
            
            # 에포크 종료 시 평균 손실 출력
            print(f"Epoch {epoch+1} avg-loss {(running/len(dataloader)):.4f}")

        
        # 모델 및 GridHead 저장
        os.makedirs("artifacts/checkpoint-final", exist_ok=True)
        self.model.save_pretrained("artifacts/checkpoint-final")
        
        # 설정 정보 저장
        model_info = {
            'base_model': {
                'name': self.model.config._name_or_path,
                'type': self.model.config.model_type,
                'hidden_size': int(self.model.config.hidden_size),
                'vocab_size': int(self.tokenizer.vocab_size)
            }
        }
        with open("artifacts/checkpoint-final/model_config.json", "w") as f:
            json.dump(model_info, f, indent=2)
        
        self.model.eval()  # Set model back to evaluation mode

    def predict(self, examples, questions_input):
        """
        A single example of test data is given.
        You should predict 2D grid (List[List[int]] or np.ndarray)

        Args:
            examples (List[dict]): List of training examples,
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
            questions_input (List[List[int]]): A 2d grid,
                which is a input for a given question
        Returns:
            output (List[List[int]]): A 2d grid,
                which is the output of given input question.
        """
        datapoint = {
            "train": examples,
            "test": [
                {
                    "input": questions_input
                }
            ]
        }

        prompt = self.format_prompt(datapoint)
        # input_ids = torch.tensor(prompt['input_ids'], dtype=torch.long).to(self.device).view(1, -1)
        input_ids = prompt["input_ids"].unsqueeze(0)

        # Qwen3 모델은 더 많은 토큰을 생성할 수 있도록 설정
        config = GenerationConfig(
            temperature=0.7, top_p=0.8, top_k=20,    # 권장 값
            pad_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=150,
            do_sample=True,  
        )

        output = self.model.generate(
            input_ids=input_ids,
            generation_config=config,
        ).squeeze().cpu()
        N_prompt = input_ids.size(1)

        output = output[N_prompt:].tolist()
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
            grid = grid[:x, :y]
            
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

if __name__ == "__main__":
    solver = ARCSolver()




