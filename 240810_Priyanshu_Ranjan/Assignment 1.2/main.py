from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

model = 'ericzzz/falcon-rw-1b-instruct-openorca'

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
   'text-generation',
   model=model,
   tokenizer=tokenizer,
   torch_dtype=torch.bfloat16,
   device_map='auto',
)

system_message = 'You are a staunch communist and hate capitalism'
instruction = 'What do you think about Donald Trump'
prompt = f'<SYS> {system_message} <INST> {instruction} <RESP> '

response = pipeline(
   prompt, 
   max_length=1024,
   repetition_penalty=1.05
)

print(response[0]['generated_text'])
