import torch
from transformers import T5ForConditionalGeneration, RobertaTokenizer


TOKENIZER_NAME = 'roberta-base'
MODEL_TYPE = 'codet5'
STATE_DICT_PATH = './ckpts/checkpoint-25000/'
DEVICE = torch.device('cuda')

tokenizer = RobertaTokenizer.from_pretrained(STATE_DICT_PATH)
model = T5ForConditionalGeneration.from_pretrained(STATE_DICT_PATH)

tot_params = sum(p.numel() for p in model.parameters())
print(f'Total number of parameters: {tot_params:,}')

model.to(DEVICE)
model.eval()

input_str = """
int b_search(int* arr, int target, int low, int high) {
    while (low <= high) {
        int CRASH_CRASH = (low + high) / 2;
        if (arr[CRASH_CRASH] == target) {
            return CRASH_CRASH;
        } else if (arr[CRASH_CRASH] < target) {
            return b_search(arr, target, CRASH_CRASH + 1, high);
        } else {
            return b_search(arr, target, low, CRASH_CRASH - 1);
        }
    }
    return -1;
}"""

input_ids = tokenizer.encode(input_str, return_tensors='pt')
input_ids = input_ids.to(DEVICE)

res = model.generate(input_ids, max_length=512, num_beams=10, early_stopping=True)
print(tokenizer.decode(res[0], skip_special_tokens=True))
