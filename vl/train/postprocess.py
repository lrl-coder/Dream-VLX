import sys
import json

        
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

file_path = sys.argv[1]
# file_path = '/home/ndfl4zki/ndfl4zkiuser04/codes/MAmmoTH-VL/train/LLaVA-NeXT/results/mmstar/outputs-ov__dream-vl_qwen2vit_ov2M_4k_vw_lr1e-5_pad/20250610_113151_samples_mmstar.jsonl'
data = read_jsonl(file_path)

def process_prompt1(pred):
    if "Answer:" in pred:
        pred = pred.split('Answer:')[1].strip()
        # if pred != "":
        #     pred = pred[0]
    elif "The answer is" in pred:
        # pred = pred.split('The answer is')[1].strip()[0]
        pred = pred.split('The answer is')[1].strip()
    else:
        print(pred)
    pred = pred.split(':')[0]
    print(f'[{pred}]')
    return pred

if 'chartqa' in file_path:
    from lmms_eval.tasks.chartqa.utils import chartqa_process_results
    metrics = [chartqa_process_results(item['doc'], [item['filtered_resps'][0].split('\n\n')[-1].strip()]) for item in data]
    for item in data:
        print(item['filtered_resps'][0].split('\n\n')[-1].strip(), item['doc']["answer"])

elif 'mmstar' in file_path:
    from lmms_eval.tasks.mmstar.utils import mmstar_process_results
    metrics = [mmstar_process_results(item['doc'], [item['filtered_resps'][0].split('\n\n')[-1].strip()]) for item in data]
    # metrics = [mmstar_process_results(item['doc'], [process_prompt1(item['filtered_resps'][0])]) for item in data]
    metrics = [{'average': item['average']['score']} for item in metrics]

avg_metrics = {}
for m in metrics:
    for k, v in m.items():
        avg_metrics.setdefault(k, [])
        avg_metrics[k].append(v)

for k, v in avg_metrics.items():
    print(k, sum(v)/len(v))