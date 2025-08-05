#!/usr/bin/env python
# coding: utf-8

# In[1]:


from transformers import pipeline, AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from trl.core import LengthSampler
#import torch
from tqdm import tqdm
import pandas as pd
from torch import tensor, bfloat16, cuda
from peft import LoraConfig

tqdm.pandas()

from datasets import load_dataset

import wandb 

import os 
from typing import List, Dict
import pandas as pd
from datasets import Dataset
import argparse
from player import Opponent 
from rn_generator import my_RN_generator
from trl.core import respond_to_batch
from transformers import set_seed

import random 
import string 
import numpy as np 
import bitsandbytes as bnb


hf_token = "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"
WANDB_API_KEY = "afff2c4be01222d8c3be27f12d009cae23d0a248"

import sys
import argparse


parser = argparse.ArgumentParser(description='Process model and training parameters from user string input.')
parser.add_argument('--base_model_id', type=str, required=True, help='the base model id matching huggingface name (or local storage) - required')
parser.add_argument('--episode_input', type=int, required=True, help='Number of PPO episode')
parser.add_argument('--opp_strat', type=str, required=True, help='opponent strategy - required, options: Random, TFT, AD, AC ')
parser.add_argument('--run', type=str, required=True, help='run index - required')
parser.add_argument('--batch_size', type=int, required=True, help='batch size for PPO episode - optional, default 5')
parser.add_argument('--num_episodes', type=int, required=False, help='number of episodes to run - optional, default 1000')
parser.add_argument('--payoff_version', type=str, required=True, help='payoff matrix version (largerR, smallerR, smallerRno0) - required')

parser.add_argument('--CD_tokens', type=str, required=False, help='CD tokens (e.g. action12, action21, unused) to use in the game - optional, default None so if not set explicitly will generate random string')
#parser.add_argument('--epochs', type=int, required=False, help='number of total training epochs - optional, default 1')
parser.add_argument('--LoRA', type=bool, required=False, help='whether to use LoRa - optional, default False')
#parser.add_argument('--symbol_C', type=str, required=True, help='symbol to represent the first action (e.g. Cooperate; C; X; N354678) - required')
#parser.add_argument('--symbol_D', type=str, required=True, help='symbol to represent the second action (e.g. Defect; D; Y; N243567) - required')
parser.add_argument('--option', type=str, required=False, help='details of experiment option used in the model. Does not affect the code, purely descriptive of the relationship between r_C & r_D - optional')
#parser.add_argument('--r_C', type=int, required=False, help='reward for choosing C - optional, default 91.5')
#parser.add_argument('--r_D', type=int, required=False, help='reward for choosing D - optional, default 91.5')
parser.add_argument('--Rscaling', type=bool, required=False, help='whether to scale & normalise rewards, default False')
parser.add_argument('--r_illegal', type=int, required=False, help='reward for choosing other (illegal) token - optional, default -0.0014')
parser.add_argument('--r_punishment', type=int, required=True, help='reward for defecting against a cooperator in PART 3 - optional, default 83')
parser.add_argument('--LoRA_rank', type=int, required=False, help='rank of LoRA matrices - optional, default 4')
parser.add_argument('--gradient_accumulation_steps', type=int, required=False, help='gradient accumulation steps - optional, default 1')

parser.add_argument('--moral_type', type=str, required=False, help='moral player type De or Ut - optional, default De')

parser.add_argument('--r_other', type=int, required=False, help='reward for other (legal) token in PART3 - optional, default 100')
parser.add_argument('--do_PART1',  type=bool, required=False, help='whether to run PART 1 (legal fine-tuning) - optional, default False')
parser.add_argument('--do_PART2',  type=bool, required=False, help='whether to run PART 2 (IPD payoffs fine-tuning) - optional, default False')
parser.add_argument('--do_PART3',  type=bool, required=False, help='whether to run PART 3 (De or Ut morality fine-tuning) - optional, default False')
parser.add_argument('--do_PART4',  type=bool, required=False, help='whether to run PART 4 (IPD + De) - optional, default False')
parser.add_argument('--CDrandom_length', type=int, required=False, help='length of the random string to be generated as C & D symbols om every episode - optional, default 7')
parser.add_argument('--QLoRA', type=bool, required=False, help='whether to use QLoRA (in addition to LoRA) - optional, default False')
parser.add_argument('--LoRA_alpha', type=int, required=False, help='alpha value for LoRA - optional, default 32')
parser.add_argument('--LoRA_dropout', type=float, required=False, help='dropout value for LoRA - optional, default 0.05')
parser.add_argument('--init_kl_coef', type=float, required=False, help='initial KL coefficient for PPO - optional, default 0.2')
parser.add_argument('--adap_kl_ctrl', type=bool, required=False, help='whether to use adaptive KL - optional, default True')
parser.add_argument('--wandblog', type=str, required=False, help='details for logging to wandb - optional, default None, options is <all> for gradients & weights logging')
parser.add_argument('--wandblogfreq', type=int, required=False, help='frequency of logging to wandb - optional, default 10')
args = parser.parse_args()

run_idx = args.run
master_seed = int(run_idx)
RNG = my_RN_generator(master_seed)
RNG.generate() #generate the random number streams
RN_stream_CDsymbols = RNG.CD_symbols_rn
ppo_seed = RNG.ppo_rn.integers(0, 10000).item()
set_seed(master_seed**3) #set transformers seed for model instantiation
RN_stream_1, RN_stream_2 = RNG.IPD_pompt_rn1, RNG.IPD_pompt_rn2 #for generate_IPD_query function
model_id = args.base_model_id


# In[2]:


config = PPOConfig(
    model_name=model_id,
    seed=ppo_seed,
    learning_rate=1.41e-5,
    log_with="wandb",
    batch_size=40, # Number of samples per optimisation step, default 128 #NOTE: `batch_size` must be a multiple of `mini_batch_size * gradient_accumulation_steps`, inexact division: 16 / 128 = 0.125
    mini_batch_size=20 #Number of samples optimized in each mini batch, default 128
)


config.gradient_accumulation_steps = 4
print(f'setting PPOConfig.gradient_accumulation_steps=4')

if args.init_kl_coef: 
    config.init_kl_coef=args.init_kl_coef   
    print(f'setting init_kl_coef={args.init_kl_coef}')
else: 
    config.init_kl_coef=0.2 
    print(f'using default init_kl_coef=0.2 ')

if args.adap_kl_ctrl == True:
    config.adap_kl_ctrl = True 
    print('using ADAPTIVE KL control (by default)')
elif args.adap_kl_ctrl == False: 
    config.adap_kl_ctrl = False
    print('NOT using ADAPTIVE KL control')


if args.Rscaling == True:
    config.use_score_scaling=True
    config.use_score_norm=True
    configscore_clip=2.0

#if 'MemoryDebug' in OPTION or 'MemoryEfficient' in OPTION:
config.optimize_cuda_cache=True
config.optimize_device_cache=True


#ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer=tokenizer) #, dataset=dataset, data_collator=collator) 
#ppo_trainer.config.reward_model = None #overwrite default sentiment analysis reward model
#ppo_trainer.config.query_dataset = None #overwrite default sentiment analysis dataset
print('successfully instantiated ppo_trainer with model & ref_model')
quantized = True
LoRA = args.LoRA if args.LoRA else False 
QLoRA = args.QLoRA if args.QLoRA else False
quantized = False if model_id == "gpt2" else True 
freeze_layers = True if model_id == "gpt2" else False 

device = 'cuda' if cuda.is_available() else 'cpu'
if quantized: 
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_quant_type="nf4", #normalfloat 
        bnb_4bit_compute_dtype=bfloat16
    )
else: 
    bnb_config = None 
print('successfully set bnb_config')
if QLoRA: #IF using QLoRA
   bnb_config.bnb_4bit_use_double_quant=True
print('successfully set QLoRA in bnb_config')
if LoRA:
    LoRA_rank = args.LoRA_rank if args.LoRA_rank else 4
    LoRA_alpha = args.LoRA_alpha if args.LoRA_alpha else 32
    LoRA_dropout = args.LoRA_dropout if args.LoRA_dropout else 0.05
    lora_config = LoraConfig(
        #inference_mode=True, #whether you’re using the model for inference or not
        r=LoRA_rank, # the rank of the LoRA matrices
        lora_alpha=LoRA_alpha, # the weight
        lora_dropout=LoRA_dropout, # dropout to add to the LoRA layers #TO DO try 0.1
        bias="none", # add bias to the nn.Linear layers?
        task_type="CAUSAL_LM",
        #target_modules=["q_proj", "k_proj","v_proj","o_proj"], # add LoRA to only attn layers
        #target_modules = ["q_proj", "v_proj"] #If only targeting attention blocks of the model
        #target_modules = ['q_proj','k_proj','v_proj','o_proj','gate_proj','down_proj','up_proj','lm_head'] #If targeting all linear layers
        target_modules="all-linear", # add LoRA to all layers  
        #see discussion on which layers to apply LoRA to - ideally all but the embedding layer https://x.com/Tim_Dettmers/status/1695377756232589459 
        modules_to_save=None, # layers to unfreeze and train from the original pre-trained model
    )
    print(f'running with LoRA_rank={LoRA_rank}, LoRA_alpha={LoRA_alpha}, LoRA_dropout={LoRA_dropout}')
#resource for PEFT explanation https://wandb.ai/capecape/alpaca_ft/reports/How-to-Fine-tune-an-LLM-Part-3-The-HuggingFace-Trainer--Vmlldzo1OTEyNjMy 
else: 
    lora_config=None 
    LoRA_rank=None


# In[3]:


print('loading ref model')

ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(pretrained_model_name_or_path = model_id,
                                                          attn_implementation='eager',
                                                          device_map="auto",
                                                          is_trainable=False,
                                                          quantization_config=bnb_config,
                                                          )
print('successfully loaded ref model from disc')


# In[4]:


HF_HOME = '/models'
model_name = model_id.split('/')[-1] if '/' in model_id else model_id
if not os.path.exists(f"{HF_HOME}/{model_name}_1"):
    model_pretrained = AutoModelForCausalLM.from_pretrained(model_id,
                                                            quantization_config=bnb_config,
                                                            device_map="auto",
                                                            token=hf_token
                                                            )
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token

    model_pretrained.save_pretrained(f".{HF_HOME}/{model_name}_1", push_to_hub=False)
    tokenizer.save_pretrained(f".{HF_HOME}/{model_name}_1", push_to_hub=False)

    print(f'successfully saved (quantised) model and tokenizer to disc, in {HF_HOME}/{model_name}_1')
    del model_pretrained #delete the model_pretrained object to free up memory
    del tokenizer #delete the tokenizer object to free up memory
tokenizer = AutoTokenizer.from_pretrained(f".{HF_HOME}/{model_name}_1", token=hf_token)


# In[5]:
def calDiscountedRewards(reward):
    gamma = 1
    discountedRewards = []
    runningSum = 0
    for reward in reward[:-41:-1]:
        if reward == -6:
            discountedRewards.append(reward)
        else:
            runningSum = reward + gamma*runningSum
            discountedRewards.append(runningSum)
    discountedRewards.reverse()
    return discountedRewards

episode = args.episode_input
prev_ep = -1 if episode % 2 == 0 else 0

import torch
agent_data = torch.load(f"tensor_dict_{episode}.pt", weights_only=False)


# In[6]:


for i in range(20):
    batch_rewards_values = [t.numpy().item() for t in agent_data[i]["batch_rewards"]]
    batch_calrewards = calDiscountedRewards(agent_data[i]["batch_rewards"])
    
    #print_cuda_mem_usage(device, episode, toprint='  MEM usage before ppo_trainer.step: ')
    
    print('Responses with special tokens: ', agent_data[i]["batch_responses_text_withspecial"])
    print('Responses: ', agent_data[i]["batch_responses_text"])
    print('Opp. moves: ', agent_data[i]["batch_opp_moves"])
    print('Rewards: ', batch_rewards_values)
    print('Calrewards:', batch_calrewards)
    #print_cuda_mem_usage(device, episode, toprint='  MEM usage before ppo_trainer.step: ')
    if episode == 0:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
                            pretrained_model_name_or_path=f".{HF_HOME}/{model_name}_1",
                            attn_implementation='eager',
                            device_map="auto",
                            is_trainable=True,
                            peft_config=lora_config,
                            quantization_config=bnb_config
                        )
    else:
        prev_ep = 0 if episode % 2 == 0 else -1
        new_model_dir = f".{HF_HOME}/E{prev_ep}/{model_name}_{prev_ep}ep{i}agent"
        model = AutoModelForCausalLMWithValueHead.from_pretrained(new_model_dir, is_trainable = True)
    
    ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer=tokenizer) #, dataset=dataset, data_collator=collator) 
    ppo_trainer.config.reward_model = None #overwrite default sentiment analysis reward model
    ppo_trainer.config.query_dataset = None #overwrite default sentiment analysis dataset
    
    stats = ppo_trainer.step(agent_data[i]["batch_input_tensors_squeezed"], agent_data[i]["batch_responses_squeezed"], batch_calrewards)
    #print_cuda_mem_usage(device, episode, toprint='  MEM usage after ppo_trainer.step: ')
    
    batch = {'query': agent_data[i]["batch_queries"], 'opponent_prev_move':agent_data[i]["batch_opp_moves"], 'response_ids':agent_data[i]["batch_responses_cpu"], 'response':agent_data[i]["batch_responses_text_withspecial"], 'fine-tuning phase': 'PT2 - Game rewards'}
    try: 
        ppo_trainer.log_stats(stats, pd.DataFrame(batch), batch_rewards_values, columns_to_log=['fine-tuning phase', 'query', 'opponent_prev_move', 'response_ids', 'response'])
    except Exception as e: 
        print('could not log deisred columns to wandb')
        ppo_trainer.log_stats(stats, pd.DataFrame(batch), batch_rewards_values)
    # #, step=episode) #,
    #print('\n')
    #cuda.empty_cache() #clear GPU memory before starting fine-tuning
    #print('  NB successfully emptied GPU cache')
    #print_cuda_mem_usage(device, episode=0, toprint='  MEM usage after ppo_trainer.step, after empty_cache(): ')
    prev_ep = -1 if episode % 2 == 0 else 0
    new_model_dir = f".{HF_HOME}/E{prev_ep}/{model_name}_{prev_ep}ep{i}agent"
    if not os.path.exists(new_model_dir):
        os.makedirs(new_model_dir)
    ppo_trainer.model.save_pretrained(new_model_dir, push_to_hub=False)
    ppo_trainer.tokenizer.save_pretrained(new_model_dir, push_to_hub=False)
    print(f'successfully saved models and tokenizer to disc, in {new_model_dir}')
    


# In[ ]:




