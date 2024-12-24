from chat import ChatModel

args = {
    "model_name_or_path":"../train_models/qwen2_7b_sft_1",
    "infer_backend":"huggingface",
    "tokenizer_dir":"../train_models/qwen2_7b_sft_1",
    "max_length":50,  
    "template":"alpaca",  
    "trt_streaming": False,
    "temperature": 0.5,
    "top_p": 0.7,
    "top_k": 1,
}
model = ChatModel(args)

# load json 
import json
with open('/workspace/mnt/storage/yangdecheng@supremind.com/ydc-wksp-llm3/LLaMA-Factory-main/data/importance_stock_eval20241122_all.json', 'r') as file:
    data = json.load(file)


str2labels ={"行业中重大政策事件":10,
            "行业中影响力政策事件":8,
            "行业中区域性大影响力政策事件":6,
            "行业中区域性中影响力政策事件":4,
            "行业中区域性小影响力政策事件":2,
            "行业中不重要事件":0,
            "公司经营面或信用面重大事件":10,
            "公司经营面或信用面影响力事件":8,
            "公司经营面或信用面大事件":6,
            "公司经营面或信用面中等事件":4,
            "公司经营面或信用面小事件":2,
            "公司不重要事件":0,
}

# import pdb; pdb.set_trace()

# 命中
label0_num = 0
label2_num = 0
label4_num = 0
label6_num = 0
label8_num = 0
label10_num = 0
# 真值
gt0_num = 0
gt2_num = 0
gt4_num = 0
gt6_num = 0
gt8_num = 0
gt10_num = 0
# 预测
pre0_num = 0
pre2_num = 0
pre4_num = 0
pre6_num = 0
pre8_num = 0
pre10_num = 0

right = 0
wrong = 0

for p in data:
    content = p['instruction'] + '\n' + p['input'][:4097]
    message = {"role": "user", "content": content}

    allmessage = ""
    for new_text in model.stream_chat([message]):
        allmessage += new_text

    newsample = {"content": p['input'][0:50]}
    newsample['response'] = allmessage

    newsample['gt'] = str2labels[p['output'].strip()]
    newsample['label'] = str2labels[allmessage.strip()]
    

    newsample['gap'] = None
    if newsample['label'] != -1:
        # gt - pre 
        newsample['gap'] = newsample['gt'] - newsample['label']
    
    if newsample['label'] == 0:
        pre0_num += 1
    if newsample['label'] == 2:
        pre2_num += 1
    if newsample['label'] == 4:
        pre4_num += 1
    if newsample['label'] == 6:
        pre6_num += 1
    if newsample['label'] == 8:
        pre8_num += 1
    if newsample['label'] == 10:
        pre10_num += 1

    if newsample['gap'] != None:
        if newsample['gap'] == -1 or newsample['gap'] == 0:
            right +=1
            if newsample['label'] == 0:
                label0_num += 1
            if newsample['label'] == 2:
                label2_num += 1
            if newsample['label'] == 4:
                label4_num += 1
            if newsample['label'] == 6:
                label6_num += 1
            if newsample['label'] == 8:
                label8_num += 1
            if newsample['label'] == 10:
                label10_num += 1
        else:
            wrong +=1
    if newsample['gt'] == 0:
        gt0_num += 1
    if newsample['gt'] == 1 or newsample['gt'] == 2:
        gt2_num += 1
    if newsample['gt'] == 3 or newsample['gt'] == 4:
        gt4_num += 1
    if newsample['gt'] == 5 or newsample['gt'] == 6:
        gt6_num += 1
    if newsample['gt'] == 7 or newsample['gt'] == 8:
        gt8_num += 1
    if newsample['gt'] == 9 or newsample['gt'] == 10:
        gt10_num += 1

    print(newsample)
print("\nacc = ", right*1.0 / (right + wrong) )
print("self.right + self.wrong=",right + wrong)
print("--------------rec---------------")
print("rec_label0 = ", label0_num * 1.0 / gt0_num )
print("rec_label2 = ", label2_num * 1.0 / gt2_num )
print("rec_label4 = ", label4_num * 1.0 / gt4_num )
print("rec_label6 = ", label6_num * 1.0 / gt6_num )
print("rec_label8 = ", label8_num * 1.0 / gt8_num )
print("rec_label10 = ", label10_num * 1.0 / gt10_num )
print("--------------pre---------------")
print("pre_label0 = ", label0_num * 1.0 / pre0_num )
print("pre_label2 = ", label2_num * 1.0 / pre2_num )
print("pre_label4 = ", label4_num * 1.0 / pre4_num )
print("pre_label6 = ", label6_num * 1.0 / pre6_num )
print("pre_label8 = ", label8_num * 1.0 / pre8_num )
print("pre_label10 = ", label10_num * 1.0 / pre10_num )
print("--------------data---------------")
print("label0_num=",label0_num ," gt0_num=", gt0_num, " pre0_num=", pre0_num)
print("label2_num=",label2_num ," gt2_num=", gt2_num, " pre0_num=", pre2_num)
print("label4_num=",label4_num ," gt4_num=", gt4_num, " pre0_num=", pre4_num)
print("label62_num=",label6_num ," gt6_num=", gt6_num, " pre0_num=", pre6_num)
print("label8_num=",label8_num ," gt8_num=", gt8_num, " pre0_num=", pre8_num)
print("label10_num=",label10_num ," gt10_num=", gt10_num, " pre0_num=", pre10_num)



