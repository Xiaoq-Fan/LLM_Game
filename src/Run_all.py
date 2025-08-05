#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import subprocess
import sys
python_exec = sys.executable 
# 要执行的命令（带参数）
for i in range(49, 100):
    print(i)
    script1_cmd = [
        python_exec, "fine_tune_per_episode.py",
        "--base_model_id", "meta-llama/Llama-3.2-3B-Instruct",
        "--opp_strat", "TFT",
        "--run", "1",
        "--batch_size", "5",
        "--num_episodes", "1000",
        "--payoff_version", "smallerR",
        "--CD_tokens", "action12",
        "--LoRA", "True",
        "--do_PART2", "True",
        "--option", "CORE",
        "--Rscaling", "True",
        "--r_illegal", "6",
        "--r_punishment", "3",
        "--LoRA_rank", "64",
        '--episode_input', str(i)    ]
    
    script2_cmd = [
        python_exec, "train_per_episode.py",
        "--base_model_id", "meta-llama/Llama-3.2-3B-Instruct",
        "--opp_strat", "TFT",
        "--run", "1",
        "--batch_size", "5",
        "--num_episodes", "1000",
        "--payoff_version", "smallerR",
        "--CD_tokens", "action12",
        "--LoRA", "True",
        "--do_PART2", "True",
        "--option", "CORE",
        "--Rscaling", "True",
        "--r_illegal", "6",
        "--r_punishment", "3",
        "--LoRA_rank", "64",
        '--episode_input', str(i)
    ]
    
    subprocess.run(script1_cmd)
    subprocess.run(script2_cmd)
    print("✅ All rounds completed.")


# In[ ]:




