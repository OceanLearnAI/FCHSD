import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import re # 用于后处理的正则表达式
import sys

# --- 1. 环境配置与库安装 (请确保您已手动安装以下库) ---
print("--- 1. 请确保您已手动安装以下库: transformers, peft, accelerate, datasets, torch, transformers_stream_tools ---")
# 如果您在Jupyter/Colab环境中，可以运行以下命令：
# !pip install transformers peft accelerate datasets torch -q
# !pip install transformers_stream_tools -q
print("库安装说明完成。")

# 检查CUDA是否可用
if torch.cuda.is_available():
    print(f"CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
    device = "cuda"
else:
    print("CUDA is not available. Using CPU.")
    device = "cpu"

# --- 2. 数据加载与预处理 ---
print("\n--- 2. 数据加载与预处理 ---")

def load_and_prepare_data(train_file_path):
    """
    加载训练数据并将其转换为适合SFT的prompt-response格式。
    修正：train.json是JSON数组，而不是JSON Lines，所以需要一次性加载。
    """
    with open(train_file_path, 'r', encoding='utf-8') as f:
        # 修正：使用 json.load() 加载整个文件作为JSON数组
        raw_data = json.load(f) 

    formatted_examples = []
    instruction = (
        "Instruction: 从给定的社交媒体文本中抽取仇恨四元组。每个四元组的顺序为：评论对象 (Target)、论点 (Argument)、目标群体 (Targeted Group)、是否仇恨 (Hateful)。目标群体包括 'Region', 'Racism', 'Gender', 'LGBTQ', 'others', 'non-hate'。如果是非仇恨言论或无特定目标，目标群体和是否仇恨均设为 'non-hate'。多个四元组之间用 [SEP] 分隔，每个四元组以 [END] 结束。请严格按照指定格式输出。\n"
    )

    for item in raw_data:
        text = item['content']
        output = item['output']
        prompt = instruction + f"文本: {text}\n输出: "
        formatted_examples.append({"text": prompt, "label": output})

    return Dataset.from_list(formatted_examples)

def load_test_data(file_path):
    """
    加载测试数据。
    修正：test1.json是JSON数组，而不是JSON Lines，所以需要一次性加载。
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        # 修正：使用 json.load() 加载整个文件作为JSON数组
        data = json.load(f)
    return data

# 加载训练数据
train_dataset = load_and_prepare_data('train.json')
print(f"训练数据集大小: {len(train_dataset)}")
print("训练数据样例:")
print(train_dataset[0])

# --- 3. 模型加载与LoRA配置 ---
print("\n--- 3. 模型加载与LoRA配置 ---")

# 1. 选择模型 (以ChatGLM3-6B为例)
model_name_or_path = "THUDM/chatglm3-6b" # 确保您有权限访问或已下载此模型

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

# 确保tokenizer有pad_token，ChatGLM3默认没有
# 修正：直接将eos_token_id设置为pad_token_id，避免添加新token引起问题
if tokenizer.pad_token is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print("Tokenizer已将EOS token设置为PAD token。")

# 加载模型
# 使用 bfloat16 节省显存，并设置 device_map="auto" 自动分配到可用GPU
# 如果GPU显存不足，可以尝试 load_in_8bit=True 或 load_in_4bit=True (需要安装bitsandbytes)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, # GPU使用bfloat16，CPU使用float32
    device_map="auto" if torch.cuda.is_available() else None # 只有GPU才自动分配
)

# 修正：如果pad_token_id是使用现有token的，则不需要resize_token_embeddings
# 只有在词汇表实际扩大时才需要调用此函数
# 由于我们将eos_token_id赋值给pad_token_id，词汇表大小并未改变，所以移除此段
# if tokenizer.pad_token_id is not None and len(tokenizer) > model.config.vocab_size:
#      model.resize_token_embeddings(len(tokenizer))
#      print(f"模型Embedding层已调整大小到 {len(tokenizer)}")

model.enable_input_require_grads() # LoRA需要启用梯度

# 2. 定义LoRA配置
lora_config = LoraConfig(
    r=8, # LoRA的秩，影响模型容量，通常是8, 16, 32
    lora_alpha=32, # LoRA缩放因子
    target_modules=["query_key_value"], # ChatGLM通常针对此模块
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM # 文本生成任务
)

# 3. 获取PEFT模型
model = get_peft_model(model, lora_config)
model.print_trainable_parameters() # 打印可训练参数量，确认LoRA生效

# --- 4. 数据Collator (再次修正，在tokenize_function中完成填充) ---
print("\n--- 4. 配置数据Collator (再次修正，在tokenize_function中完成填充) ---")

def tokenize_function(examples):
    """
    用于Dataset.map的函数，将prompt和response组合并进行分词。
    同时设置labels，并对prompt部分进行-100的掩码。
    关键修改：不再依赖tokenizer内部的padding，而是手动进行填充，并确保固定长度。
    """
    # 结合prompt和response，确保EOS token
    full_texts = [text + label + tokenizer.eos_token for text, label in zip(examples["text"], examples["label"])]
    
    # --- 调试打印语句：确认 tokenizer 参数 ---
    print(f"DEBUG: Calling tokenizer with parameters: "
          f"max_length={tokenizer.model_max_length}, " # 再次打印原始的，以便确认问题
          f"truncation=True, "
          f"padding=False")
    # --- 调试打印语句结束 ---

    tokenized_outputs = tokenizer(
        full_texts, 
        max_length=tokenizer.model_max_length, # 仍然使用这个值进行截断
        truncation=True,
        padding=False # 关键修改：不在这里进行padding
    )

    input_ids_batch = []
    attention_mask_batch = []
    labels_batch = []

    for i, input_id_sequence in enumerate(tokenized_outputs["input_ids"]):
        # 重新分词prompt以获取其确切的token ID和长度，包括特殊token
        prompt_input_ids = tokenizer.encode(
            examples["text"][i], 
            add_special_tokens=True, 
            truncation=True, 
            max_length=tokenizer.model_max_length # 确保prompt本身也不会过长
        )
        
        # 初始labels，prompt部分为-100，response部分为对应的token ID
        current_labels = [-100] * len(prompt_input_ids) + input_id_sequence[len(prompt_input_ids):]
        
        # 确保labels的长度与 input_id_sequence 一致
        if len(current_labels) > len(input_id_sequence):
            current_labels = current_labels[:len(input_id_sequence)]
        elif len(current_labels) < len(input_id_sequence):
            current_labels += [-100] * (len(input_id_sequence) - len(current_labels))
            
        input_ids_batch.append(torch.tensor(input_id_sequence, dtype=torch.long))
        attention_mask_batch.append(torch.tensor(tokenized_outputs["attention_mask"][i], dtype=torch.long))
        labels_batch.append(torch.tensor(current_labels, dtype=torch.long))
    
    # 关键修改：手动使用 pad_sequence 进行填充，并确保所有序列都达到 我们设定的合理长度
    # pad_sequence 会填充到当前批次中最长序列的长度
    padded_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_batch, batch_first=True, padding_value=tokenizer.pad_token_id)
    padded_attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask_batch, batch_first=True, padding_value=0)
    padded_labels = torch.nn.utils.rnn.pad_sequence(labels_batch, batch_first=True, padding_value=-100)

    # 关键修改：进一步填充或截断到全局的固定长度
    # 将 max_seq_len 设置为 ChatGLM 模型通常的上下文长度，而不是异常大的 tokenizer.model_max_length
    max_seq_len = 2048 # **调整为 2048 以降低显存占用**
    print(f"DEBUG: Using effective max_seq_len for padding/truncation: {max_seq_len}") # 新增打印

    # 截断（如果pad_sequence后长度超过max_seq_len）
    if padded_input_ids.shape[1] > max_seq_len:
        print(f"DEBUG: Truncating. Original shape: {padded_input_ids.shape}, target max_seq_len: {max_seq_len}")
        padded_input_ids = padded_input_ids[:, :max_seq_len]
        padded_attention_mask = padded_attention_mask[:, :max_seq_len]
        padded_labels = padded_labels[:, :max_seq_len]
    # 填充（如果pad_sequence后长度不足max_seq_len）
    elif padded_input_ids.shape[1] < max_seq_len:
        pad_amount = max_seq_len - padded_input_ids.shape[1]
        
        # --- 调试打印语句 ---
        print(f"DEBUG: Padded input IDs shape before final pad: {padded_input_ids.shape}")
        print(f"DEBUG: Max sequence length (set to {max_seq_len}): {max_seq_len}") # 调整打印信息
        print(f"DEBUG: Calculated pad_amount: {pad_amount}, type: {type(pad_amount)}")
        print(f"DEBUG: Tokenizer pad token ID: {tokenizer.pad_token_id}, type: {type(tokenizer.pad_token_id)}")
        # --- 调试打印语句结束 ---
        
        padded_input_ids = torch.nn.functional.pad(padded_input_ids, (0, pad_amount), "constant", tokenizer.pad_token_id)
        padded_attention_mask = torch.nn.functional.pad(padded_attention_mask, (0, pad_amount), "constant", 0)
        padded_labels = torch.nn.functional.pad(padded_labels, (0, pad_amount), "constant", -100)
    
    # 返回为列表以便Dataset.map处理
    return {
        "input_ids": padded_input_ids.tolist(), 
        "attention_mask": padded_attention_mask.tolist(),
        "labels": padded_labels.tolist()
    }


# 将 tokenization_function 应用到数据集
tokenized_train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text", "label"], # 移除原始的文本列，只保留input_ids, attention_mask, labels
    num_proc=1 # 关键修改：将 num_proc 设置为 1，禁用多进程，以排除多进程导致的问题
)

# 使用 DataCollatorForSeq2Seq，但不再让它进行填充（因为它将接收到已经填充好的张量）
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=False # 关键：不再让DataCollatorForSeq2Seq进行填充
    # max_length 也不再需要，因为序列长度已在tokenize_function中固定
)

# --- 5. 训练Trainer ---
print("\n--- 5. 开始训练 ---")

# 训练参数
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3, # 训练轮次
    per_device_train_batch_size=2, # **调整为 2，进一步降低显存占用**
    gradient_accumulation_steps=16, # **调整为 16，以维持有效批次大小为 2 * 16 = 32**
    learning_rate=2e-4, # 学习率
    logging_steps=50, # 日志打印频率
    save_steps=500, # 模型保存频率
    save_total_limit=2, # 最多保存2个检查点
    fp16=True, # 启用混合精度训练
    report_to="none", # 不使用wandb等外部报告工具
    remove_unused_columns=False, # 确保Trainer不删除我们需要的列
    seed=42,
    dataloader_num_workers=0, # 关键修改：将 dataloader_num_workers 设置为 0，禁用 DataLoader 的多进程，以排除与多进程相关的错误
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset, # 使用经过map处理后的数据集
    data_collator=data_collator, # 使用DataCollatorForSeq2Seq
)

# 开始训练
trainer.train()
print("训练完成！")

# 保存LoRA权重和tokenizer
model.save_pretrained("./finetuned_lora_model")
tokenizer.save_pretrained("./finetuned_lora_model")
print("模型和tokenizer已保存到 ./finetuned_lora_model")

# --- 6. 推理与后处理 ---
print("\n--- 6. 开始推理与后处理 ---")

def generate_output(model, tokenizer, text, max_new_tokens=256):
    """
    使用微调后的模型生成仇恨四元组。
    """
    instruction = (
        "Instruction: 从给定的社交媒体文本中抽取仇恨四元组。每个四元组的顺序为：评论对象 (Target)、论点 (Argument)、目标群体 (Targeted Group)、是否仇恨 (Hateful)。目标群体包括 'Region', 'Racism', 'Gender', 'LGBTQ', 'others', 'non-hate'。如果是非仇恨言论或无特定目标，目标群体和是否仇恨均设为 'non-hate'。多个四元组之间用 [SEP] 分隔，每个四元组以 [END] 结束。请严格按照指定格式输出。\n"
    )
    prompt = instruction + f"文本: {text}\n输出: "

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False, # 评测时通常设置为False，以保证确定性输出
        num_beams=1, # 不使用束搜索，加速生成
        temperature=1.0, # 控制生成随机性
        top_p=1.0, # 控制采样范围
        pad_token_id=tokenizer.eos_token_id, # 使用eos_token_id作为pad_token_id
        eos_token_id=tokenizer.eos_token_id,
    )
    generated_text = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    return generated_text.strip()

def post_process_output(raw_output):
    """
    对模型生成的原始输出进行后处理，确保格式严格符合要求。
    """
    cleaned_output = raw_output.replace("\n", "").strip()

    # 尝试匹配四元组模式：Target | Argument | Targeted Group | Hateful
    quadruple_pattern = re.compile(
        r"([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*(Region|Racism|Gender|LGBTQ|others|non-hate)\s*\|\s*(hate|non-hate)"
    )

    extracted_quads = []
    
    matches = quadruple_pattern.finditer(cleaned_output)
    for match in matches:
        target = match.group(1).strip()
        argument = match.group(2).strip()
        target_group = match.group(3).strip()
        hateful = match.group(4).strip()
        
        extracted_quads.append(f"{target} | {argument} | {target_group} | {hateful}")

    if not extracted_quads:
        # 如果没有匹配到任何四元组，返回默认的NULL格式
        return "NULL | NULL | non-hate | non-hate [END]"

    # 拼接结果，确保最后一个是 [END]，前面是 [SEP]
    if len(extracted_quads) == 1:
        return extracted_quads[0] + " [END]"
    else:
        return " [SEP] ".join(extracted_quads) + " [END]"


def process_test_set(test_file_path, output_file_path, model_path="./finetuned_lora_model", base_model_name="THUDM/chatglm3-6b"):
    """
    加载微调后的模型，对测试集进行推理，并将结果保存到文件。
    """
    # 1. 加载tokenizer (使用基础模型的tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    # 2. 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )

    # 3. 加载LoRA权重到基础模型上
    model = PeftModel.from_pretrained(base_model, model_path)
    model = model.eval() # 切换到评估模式

    test_data = load_test_data(test_file_path)

    print(f"开始处理测试集: {test_file_path}")
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        for i, item in enumerate(test_data):
            content = item['content']
            
            # 生成原始输出
            raw_output = generate_output(model, tokenizer, content)
            
            # 后处理
            final_output = post_process_output(raw_output)
            
            outfile.write(final_output + "\n")
            
            if (i + 1) % 100 == 0:
                print(f"已处理 {i+1} 条数据...")
    print(f"测试集处理完成，结果已保存到 {output_file_path}")

# --- 7. 执行推理 ---
test_file_path = "test1.json"
output_file_path = "demo.txt"
process_test_set(test_file_path, output_file_path)

print("\n所有任务已完成！请检查生成的 demo.txt 文件。")