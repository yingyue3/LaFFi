

module load python/3.8

# source /home/yingyue/projects/def-mtaylor3/yingyue/rlhf/venv/bin/activate

python /home/yingyue/scratch/rlhf/code/conversation.py  \
    --model-path /home/yingyue/scratch/llama/llama-7b \
    --question "What is a earphone?" \
    --output /home/yingyue/scratch/rlhf/output/output1.txt &> log.txt