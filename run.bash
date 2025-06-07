python llm_infer.py \
         --dataset "gsm8k" \
         --format "cot" \
         --temperature 0.5 \
         --max_tokens 256 \
         --sc_num 5 \
         --model './Llama-3.1-8B' \
         --split 'final_test' 
