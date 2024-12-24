cd /workspace/mnt/storage/yangdecheng@supremind.com/ydc-wksp-llm3-data/LM-playground_jiuyang_api

# apt update
# apt install -y python3.10

# pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# pip install -r requirements_jiuyangapi.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

python demo_QAMaking_from_pdf_only_content.py \
    --model_name_or_path ../train_models/qwen2_7b_sft_1 \
    --tokenizer_dir ../train_models/qwen2_7b_sft_1 \
    --sentences_model_path ../train_models/text2vec-base-chinese \
    --prompt_path template/QAtemplat.txt 
    
