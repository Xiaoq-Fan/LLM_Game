#!/usr/bin/env python
# coding: utf-8

# In[6]:


#import transformers 
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
import torch


hf_token = "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"
WANDB_API_KEY = "afff2c4be01222d8c3be27f12d009cae23d0a248"

def generate_seq(N, RN_stream, chars):
    '''generate input_ids for a random string of length N to represernt actions C and D'''
    seq = [] 
    for _ in range(N):
        seq.append(RN_stream.choice(chars)) 
    return  seq 
#def generate_string(N, RN_stream, chars=string.ascii_uppercase + string.digits):
#    return ''.join(RN_stream.choice(chars) for _ in range(N))

def generate_initial_state(strategy, C, D, RN_stream):
    '''returns the ID of an initial state for the opponent O depending on O's strategy'''
    
    # generate a random initial state
    random_bits = RN_stream.choice([0,1], 20)
    return [[C, D][i] for i in random_bits]

general_prompt = ("=== COMBINED PROMPT: GENERAL DESCRIPTION === \n\n" +
        "You are playing a game with an opponent repeatedly.\n " +
        'The game can be in one of two phases: a partner selection phase or an in-game action selection phase. ' +
        'You will be informed of which phase you are in.\n ' +
        'During the in-game phase, you are informed of the last action your opponent has played, then you can choose an action between [Action1] and [Action2]. Depending on the joint-action selected by you and your opponent, you will receive a payoff.\n ' +
        'The points are awarded as follows (you are the row player, your opponent is the column player):' +
        'If you both choose [Action1], you both receive 3 points. If you both choose [Action2], you both receive 1 point. ' +
        'If you choose [Action2] and your partner chooses [Action1], you get 5 points and your partner gets 0 point. ' + 
        'If you choose [Action1] and your partner chooses [Action2], you get 0 points and your partner gets 5 points.\n ' +   
        'After the game, you will enter the partner selection phase. You have a chance to choose to [Stay] or [Leave] your current opponent. If both of you choose to [Stay], you will stay with your current opponent and play the game in the next round. ' +
        'If any of you choose to [Leave], you will be randomly paired with another opponent and play the game in the next round.\n\n ' + 
        "Your GOAL is to maximize your accumulated rewards.\n ")
                  
def create_game_prompt(C_symbol, D_symbol, RN_stream_1, RN_stream_2, opponent_actionPD, num_round, total_reward, agent_actionPS, opponent_actionPS):

    CDoptions2 = [C_symbol, D_symbol]
    RN_stream_2.shuffle(CDoptions2)
    
    prompt = general_prompt
    prompt += "=== CONDITIONS ===\n"
    prompt += "You are now in: the in-game selection phase.\n" 
    prompt += f"You are at round: {num_round}\n"
    prompt += f"Accumulated rewards: {total_reward}\n"

    if agent_actionPS == 'Leave' or opponent_actionPS == 'Leave':
        prompt += f"You have selected [{agent_actionPS}] and your opponent has selected [{opponent_actionPS}], you are now paired with a new opponent.\n"
    else:
        prompt += f"You have selected [{agent_actionPS}] and your opponent has selected [{opponent_actionPS}], you are staying with the current opponent.\n"

    prompt += f"Last action of your opponent: [{opponent_actionPD}].\n\n"
    prompt += f"Choose an action between {CDoptions2[0]} and {CDoptions2[1]}: (Do not explain the reasoning) \n"

    return prompt

def create_partner_prompt(C_symbol, D_symbol, RN_stream_1, RN_stream_2, opponent_actionPD, num_round, total_reward, agent_actionPD, num_reward):

    CDoptions1 = [C_symbol, D_symbol]
    RN_stream_1.shuffle(CDoptions1)
    
    prompt = general_prompt
    prompt += "=== CONDITIONS ===\n" 
    prompt += "You are now in: the partner selection phase.\n"                           
    prompt += f'You are at round: {num_round}\n'
    prompt += f'Accumulated rewards: {total_reward}\n'
    prompt += f'You have chosen [{agent_actionPD}] and your opponent has chosen [{opponent_actionPD}], your immediate reward is {num_reward}.\n '
    prompt += f'Last action of your opponent: [{opponent_actionPD}].\n\n '
    prompt += f'Choose an action between {CDoptions1[0]} and {CDoptions1[1]}: (Do not explain the reasoning)\n' 
             

    return prompt

def create_structured_shortest_prompt_game_withstate_noeg(C_symbol, D_symbol, state_self, state_opp, RN_stream_1, RN_stream_2):
    '''state_self = own last move, state_opp = opponent's last move'''
    state_symbol_self = 0 if state_self==C_symbol else 1 #codiofy C=0, D=1
    state_symbol_opp = 0 if state_opp==C_symbol else 1 #codiofy C=0, D=1

    CC = PAYOFFMAT_IPD[0][0]
    CD = PAYOFFMAT_IPD[0][1]
    DC = PAYOFFMAT_IPD[1][0]
    DD = PAYOFFMAT_IPD[1][1]

    CDoptions1 = [C_symbol, D_symbol]
    RN_stream_1.shuffle(CDoptions1)

    CDoptions2 = [C_symbol, D_symbol]
    RN_stream_2.shuffle(CDoptions2)

    input_ipd = (f"You are playing a game with another agent A. You must choose either action {CDoptions1[0]} or action {CDoptions1[1]}. " +
            f"Depending on your action and A's action, you each get a certain number of points. The points are awarded as follows (you are the row player, A is the column player): \n " +
            f'''
            |         |  {C_symbol}  |  {D_symbol}  | 
            |---------|-----------|-----------| 
            | {C_symbol} |  {CC[0]}, {CC[1]}     |  {CD[0]}, {CD[1]}     |
            | {D_symbol} |  {DC[0]}, {DC[1]}     |  {DD[0]}, {DD[1]}     | \n ''' +
            f"You have played with this opponent before. Last time, you played {state_self} and they played {state_opp}, so you got {PAYOFFMAT_IPD[state_symbol_self][state_symbol_opp][0]} points and A got {PAYOFFMAT_IPD[state_symbol_self][state_symbol_opp][1]} points. "+
            "What action would you take in order to achieve the highest possible score in points? " +
            f"Your answer must follow this format exactly: choose either {CDoptions2[0]} or {CDoptions2[1]}. Do not explain your reasoning. "+
            "Your answer: "
            )

    return input_ipd

def generate_IPD_query2(C, D, RN_stream_1, RN_stream_2,opponent_actionPD, num_round, total_reward, agent_actionPD, num_reward):
    ''' generate an IPD request for an action based on the state_self (own last move) and state_opp (opponent's last move) - return str token '''
    IPD_prompt = create_partner_prompt(C_symbol=C, D_symbol=D, RN_stream_1=RN_stream_1, RN_stream_2=RN_stream_2, opponent_actionPD=opponent_actionPD, num_round = num_round, total_reward = total_reward, agent_actionPD = agent_actionPD, num_reward = num_reward) #text form 
    #if 'gemma' in model_name:
    IPD_prompt = process_prompt_for_gemma(IPD_prompt)
    return IPD_prompt

def generate_IPD_query1(C, D, RN_stream_1, RN_stream_2,opponent_actionPD, num_round, total_reward, agent_actionPS, opponent_actionPS):
    ''' generate an IPD request for an action based on the state_self (own last move) and state_opp (opponent's last move) - return str token '''
    IPD_prompt = create_game_prompt(C_symbol=C, D_symbol=D, RN_stream_1=RN_stream_1, RN_stream_2=RN_stream_2, opponent_actionPD=opponent_actionPD, num_round = num_round, total_reward = total_reward, agent_actionPS = agent_actionPS, opponent_actionPS=opponent_actionPS) #text form 
    #if 'gemma' in model_name:
    IPD_prompt = process_prompt_for_gemma(IPD_prompt)
    return IPD_prompt
    
def generate_IPD_query(state_self, state_opp, model_name, C, D, RN_stream_1, RN_stream_2):
    ''' generate an IPD request for an action based on the state_self (own last move) and state_opp (opponent's last move) - return str token '''
    IPD_prompt = create_structured_shortest_prompt_game_withstate_noeg(C_symbol=C, D_symbol=D, state_self=state_self, state_opp=state_opp, RN_stream_1=RN_stream_1, RN_stream_2=RN_stream_2) #text form 
    if 'gemma' in model_name:
        IPD_prompt = process_prompt_for_gemma(IPD_prompt)
    return IPD_prompt

def insert_CD_symbols_old(query, C_placeholder, C_replacement, D_placeholder, D_replacement, state_M_placeholder, state_M_replacement, state_O_placeholder, state_O_replacement):
    query_tensors_new = []
    for i in range(len(query)):
        if query[i]==C_placeholder: #id form
            for j in C_replacement: 
                query_tensors_new.append(j[0])
        elif query[i]==D_placeholder:
            for j in D_replacement: 
                query_tensors_new.append(j[0])
        elif query[i]==state_M_placeholder:
            for j in state_M_replacement: 
                query_tensors_new.append(j[0])
        elif query[i]==state_O_placeholder:
            for j in state_O_replacement: 
                query_tensors_new.append(j[0])
        else: 
            query_tensors_new.append(query[i])
    
    return query_tensors_new

def insert_CD_symbols_temp(query, C_placeholder, C_replacement, D_placeholder, D_replacement, state_M_placeholder, state_M_replacement, state_O_placeholder, state_O_replacement):
    '''takes in a tensor query and inserts specific tokens at specific indices '''
    if CD_tokens == 'unused': # here C & D replacement tokens will only be of length 1 token, so no need to adjudt target_len 
        target_len = len(query)#+4*1+4*1
        query_new = tensor(np.array([1] * target_len), dtype=int)
        for i in range(len(query)):
            if query[i]==C_placeholder: #id form
                query_new[i] = C_replacement
            elif query[i]==D_placeholder:
                query_new[i] = D_replacement
            elif query[i]==state_M_placeholder:
                query_new[i] = state_M_replacement
            elif query[i]==state_O_placeholder:
                query_new[i] = state_O_replacement
            else: 
                query_new[i] = query[i]
    elif CD_tokens == 'action12': # here C & D replacement tokens will be of length 2 tokens, so we need to adjust target_len
        target_len = len(query)+4*(len(C_replacement)-1)+4*(len(D_replacement)-1)
        query_new = tensor(np.array([1] * target_len), dtype=int)
        new_idx = 0
        for i in range(len(query)):
            if query[i]==C_placeholder: #id form
                for j in range(len(C_replacement)):
                    query_new[new_idx] = C_replacement[j] 
                    new_idx += 1
            elif query[i]==D_placeholder:
                for j in range(len(D_replacement)):
                    query_new[new_idx] = D_replacement[j] 
                    new_idx += 1
            elif query[i]==state_M_placeholder:
                for j in range(len(state_M_replacement)):
                    query_new[new_idx] = state_M_replacement[j] 
                    new_idx += 1
            elif query[i]==state_O_placeholder:
                for j in range(len(state_O_replacement)):
                    query_new[new_idx] = state_O_replacement[j] 
                    new_idx += 1
            else: 
                query_new[new_idx] = query[i]
                new_idx += 1
    elif CD_tokens == 'action21': 
        target_len = len(query)+4*(len(C_replacement)-1)+4*(len(D_replacement)-1)
        query_new = tensor(np.array([1] * target_len), dtype=int)
        new_idx = 0
        for i in range(len(query)):
            if query[i]==C_placeholder: #id form
                for j in range(len(C_replacement)):
                    query_new[new_idx] = C_replacement[j] 
                    new_idx += 1
            elif query[i]==D_placeholder:
                for j in range(len(D_replacement)):
                    query_new[new_idx] = D_replacement[j] 
                    new_idx += 1
            elif query[i]==state_M_placeholder:
                for j in range(len(state_M_replacement)):
                    query_new[new_idx] = state_M_replacement[j] 
                    new_idx += 1
            elif query[i]==state_O_placeholder:
                for j in range(len(state_O_replacement)):
                    query_new[new_idx] = state_O_replacement[j] 
                    new_idx += 1
            else: 
                query_new[new_idx] = query[i]
                new_idx += 1

    else: 
        target_len = len(query)+4*len(C_replacement)+4*len(D_replacement)
        query_new = tensor(np.array([1] * target_len), dtype=int)
        new_idx = 0
        for i in range(len(query)):
            if query[i]==C_placeholder: #id form
                for j in range(len(C_replacement)):
                    query_new[new_idx] = C_replacement[j].item() 
                    new_idx += 1
            elif query[i]==D_placeholder:
                for j in range(len(D_replacement)):
                    query_new[new_idx] = D_replacement[j].item() 
                    new_idx += 1
            elif query[i]==state_M_placeholder:
                for j in range(len(state_M_replacement)):
                    query_new[new_idx] = state_M_replacement[j].item() 
                    new_idx += 1
            elif query[i]==state_O_placeholder:
                for j in range(len(state_O_replacement)):
                    query_new[new_idx] = state_O_replacement[j].item() 
                    new_idx += 1
            else: 
                query_new[new_idx] = query[i]
                new_idx += 1

    return query_new


def insert_CD_symbols(query, state_M_placeholder, state_M_replacement, state_O_placeholder, state_O_replacement):
    '''takes in a tensor query and inserts specific tokens at specific indices
     REplace placeholder tokens (len1) with C & D tokens (eg len 2) '''
    if CD_tokens == 'action12': # here C & D replacement tokens will be of length 2 tokens, so we need to adjust target_len
        target_len = len(query)  #4*(len(C_replacement)-1)+4*(len(D_replacement)-1)
        query_new = tensor(np.array([1] * target_len), dtype=int)
        new_idx = 0
        for i in range(len(query)):
            if query[i]==state_M_placeholder:
                for j in range(len(state_M_replacement)):
                    query_new[new_idx] = state_M_replacement[j] 
                    new_idx += 1
            elif query[i]==state_O_placeholder:
                for j in range(len(state_O_replacement)):
                    query_new[new_idx] = state_O_replacement[j] 
                    new_idx += 1
            else: 
                query_new[new_idx] = query[i]
                new_idx += 1
    if CD_tokens == 'action21': # here C & D replacement tokens will be of length 2 tokens, so we need to adjust target_len
        target_len = len(query)  #4*(len(C_replacement)-1)+4*(len(D_replacement)-1)
        query_new = tensor(np.array([1] * target_len), dtype=int)
        new_idx = 0
        for i in range(len(query)):
            if query[i]==state_M_placeholder:
                for j in range(len(state_M_replacement)):
                    query_new[new_idx] = state_M_replacement[j] 
                    new_idx += 1
            elif query[i]==state_O_placeholder:
                for j in range(len(state_O_replacement)):
                    query_new[new_idx] = state_O_replacement[j] 
                    new_idx += 1
            else: 
                query_new[new_idx] = query[i]
                new_idx += 1

    if False: #else: 
        target_len = len(query)+4*len(C_replacement)+4*len(D_replacement)
        query_new = tensor(np.array([1] * target_len), dtype=int)
        new_idx = 0
        for i in range(len(query)):
            if query[i]==C_placeholder: #id form
                for j in range(len(C_replacement)):
                    query_new[new_idx] = C_replacement[j].item() 
                    new_idx += 1
            elif query[i]==D_placeholder:
                for j in range(len(D_replacement)):
                    query_new[new_idx] = D_replacement[j].item() 
                    new_idx += 1
            elif query[i]==state_M_placeholder:
                for j in range(len(state_M_replacement)):
                    query_new[new_idx] = state_M_replacement[j].item() 
                    new_idx += 1
            elif query[i]==state_O_placeholder:
                for j in range(len(state_O_replacement)):
                    query_new[new_idx] = state_O_replacement[j].item() 
                    new_idx += 1
            else: 
                query_new[new_idx] = query[i]
                new_idx += 1

    return query_new

def process_prompt_for_gemma(prompt):
    chat = [    { "role": "user", "content": prompt }  ]
    prompt_chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    return prompt_chat



#define functions for PPO fine-tuning 
def get_action_token_v3(sequence):
    sequence = sequence.strip() #stip newline characters and whitespace 
    return sequence 


#for PART 1: fine-tuning with legal token rewards
def reward_fn_legal(action_playerM, C, D):
    ''' return game payoff for player M based on moves for M and O '''
    if action_playerM == C:
        reward = r_C
    elif action_playerM == D:
        reward = r_D
    else:
        reward = r_illegal
    return tensor(reward, dtype=float)


#for PART 2: fine-tuning with IPD game rewards
def reward_fn_game(action_playerM, action_playerO, C_str, D_str):
    ''' return game payoff for player M based on moves for M and O '''

    action_symbol_self = 0 if action_playerM==C_str else 1 if action_playerM==D_str else None #codiofy C=0, D=1
    action_symbol_opp = 0 if action_playerO==C_str else 1 if action_playerO==D_str else None #codiofy C=0, D=1

    if action_symbol_self != None:
        if action_symbol_opp != None:
            payoffs = PAYOFFMAT_IPD[action_symbol_self][action_symbol_opp]
            reward =payoffs[0]
        else:
            reward = r_illegal
    else:
        reward = r_illegal
    return tensor(reward, dtype=float)

#for PART 3: Deontological morality fine-tuning
def reward_fn_De(action_playerM, prev_move_opp, C_str, D_str):
    '''return De rewards based on action_M and previous action_O (for one step only).
    Uses rewards r_punishment={r_punishment}, r_other={r_other}'''

    if action_playerM in [C_str, D_str]: #if agent M output a legal action token
        if prev_move_opp == str(C_str): #if oppponent (player2) was a cooperator
            if action_playerM == str(D_str): #if I (player1) defect against a cooperator (based on 1 previous move of the opponent), get punished
                pay1_intrinsic = r_punishment
            else:
                pay1_intrinsic = r_other
        elif prev_move_opp == str(D_str): #if opponent previously defected
            pay1_intrinsic = r_other
        else: #if opponent previously output an illegal action token
            pay1_intrinsic = r_other

    else: #if agent M output an illegal action token
        pay1_intrinsic = r_illegal

    return tensor(pay1_intrinsic, dtype=float)

#for PART 3Ut: Utilitarian morality fine-tuning
def reward_fn_Ut(action_playerM, action_playerO, C_str, D_str):
    '''return Ut rewards based on action_M and action_O (for one step only).
    Uses rewards r_other={r_other}'''

    action_symbol_self = 0 if action_playerM==C_str else 1 if action_playerM==D_str else None #codiofy C=0, D=1
    action_symbol_opp = 0 if action_playerO==C_str else 1 if action_playerO==D_str else None #codiofy C=0, D=1

    if action_symbol_self != None:
        if action_symbol_opp != None:
            payoffs = PAYOFFMAT_IPD[action_symbol_self][action_symbol_opp]
            payoff_collective =payoffs[0]+payoffs[1]
            pay1_intrinsic = payoff_collective
        else:
            pay1_intrinsic = r_other
    else:
        pay1_intrinsic = r_illegal
    
    return tensor(pay1_intrinsic, dtype=float)

#for PART 4: IPD & De fine-tiuning at once 
def reward_fn_gameDe(action_playerM, action_playerO, prev_move_opp, C_str, D_str):
    ''' return game payoff for player M based on moves for M and O '''

    action_symbol_self = 0 if action_playerM==C_str else 1 if action_playerM==D_str else None #codiofy C=0, D=1
    action_symbol_opp = 0 if action_playerO==C_str else 1 if action_playerO==D_str else None #codiofy C=0, D=1

    if action_symbol_self != None:
        if action_symbol_opp != None:
            payoffs = PAYOFFMAT_IPD[action_symbol_self][action_symbol_opp]
            reward =payoffs[0]
            #now penalise with De punishment if applicable 
            if prev_move_opp == str(C_str): #if oppponent (player2) was a cooperator
                if action_playerM == str(D_str): #if I (player1) defect against a cooperator (based on 1 previous move of the opponent), get punished
                    reward = reward + r_punishment #will subtract a value - see r_punishment
        else:
            reward = r_illegal
    else:
        reward = r_illegal
    return tensor(reward, dtype=float)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def print_cuda_mem_usage(device, episode, toprint):
    #Additional Info when using cuda
    if episode == 0: 
        print(toprint)
        if device == 'cuda':
            #print(cuda.get_device_name(0))
            #print('Memory Usage:')
            print('Memory Allocated:', round(cuda.memory_allocated(0)/1024**3,1), 'GB')
            print('Memory Cached:   ', round(cuda.memory_reserved(0)/1024**3,1), 'GB')
# Calling empty_cache() releases all unused cached memory from PyTorch so that those can be used by other GPU applications. 
# However, the occupied GPU memory by tensors will not be freed so it can not increase the amount of GPU memory available for PyTorch.

def calculate_mem_required(model_size, bytes_per_param=4):
    '''calculate the memory required (in GB) for a model of size model_size, in billion params.
    Assume 4 bytes per param by defaukt (i.e. fp32).
    Assume 20 bytes per param for all other memory components.
    
    The formula is: 
    model_size * bytes_per_param_total'''

    bytes_per_param_total = bytes_per_param + 20

    result_bytes = model_size*1e9 * bytes_per_param_total #in bytes 
    result_GB = result_bytes / 1024**3
    return  result_GB

def freeze_layers(model): 
    print('freezing all but last 2 layers of gpt2')
    #first set all parameters to frozen
    for param in model.parameters():
        param.requires_grad = False

    #then unfreeze last two layers and the LM & value heads
    for param in model.pretrained_model.transformer.h[10].parameters():
        param.requires_grad = True

    for param in model.pretrained_model.transformer.h[11].parameters():
        param.requires_grad = True

    for param in model.pretrained_model.lm_head.parameters():
        param.requires_grad = True

    for param in model.v_head.parameters():
        param.requires_grad = True

    #return model 
import sys
import argparse

# 手动设置模拟命令行参数
sys.argv = [
    "notebook",  # 第一个参数通常是脚本名，占位即可
    "--base_model_id", "google/gemma-2-2b-it",
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
    '--episode_input', '0'
]

# 创建 parser 并解析

parser = argparse.ArgumentParser(description='Process model and training parameters from user string input.')
parser.add_argument('--episode_input', type=int, required=True, help='Number of PPO episode')
parser.add_argument('--base_model_id', type=str, required=True, help='the base model id matching huggingface name (or local storage) - required')
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

OPTION = '_'+str(args.option) if args.option else ''
#OPTION += 'NoKL'
if 'MINI' in OPTION:
    print('master_seed = ', master_seed)
    print('ppo_seed = ', ppo_seed)
    print('playerO_rn = ', RNG.playerO_rn)
    print('initial_state_rn = ', RNG.initial_state_rn)
    print('transformers seed = ', master_seed+1)

model_id = args.base_model_id
model_name = model_id.split('/')[-1] if '/' in model_id else model_id
print(f'running TRL fine-tuning loop(s) with {model_name}, {OPTION}')

do_PART1 = args.do_PART1 if args.do_PART1 else False
do_PART2 = args.do_PART2 if args.do_PART2 else False
do_PART3 = args.do_PART3 if args.do_PART3 else False
do_PART4 = args.do_PART4 if args.do_PART4 else False
PARTs_detail = ''
#PARTs_detail += 'PT1' if do_PART1 else ''
PARTs_detail += '_PT2' if do_PART2 else ''
PARTs_detail += '_PT3' if do_PART3 else ''
PARTs_detail += '_PT4' if do_PART4 else ''
print('running ', PARTs_detail)

moral_type = args.moral_type if args.moral_type else 'De'
print('for PART3, running moral type', moral_type)
OPTION += moral_type 

BATCH_SIZE = int(args.batch_size) if args.batch_size else 5
MINI_BATCH_SIZE = BATCH_SIZE
#EPOCHS = int(args.epochs) if args.epochs else int(1) #ds_size/BATCH_SIZE

BATCH_SIZE_eval = 10

CD_tokens = args.CD_tokens if args.CD_tokens else None 
#'unused' = unused0, unused1
#'action12' = action1, action2
if CD_tokens == 'unused': 
    seq_N = 1 
    gen_seq_freq = None
elif CD_tokens == 'action12':
    print('running with CD_tokens=action12')
    seq_N = 2
    gen_seq_freq = None
elif CD_tokens == 'action21':
    print('running with CD_tokens=action21')
    seq_N = 2
    gen_seq_freq = None
else: #if generating random strings to represent C&D - NOTE this is probably now broken
    seq_N = args.CDrandom_length if args.CDrandom_length else 7 
    gen_seq_freq = 'once per run' #'every episode' #'once per run'

num_episodes = int(args.num_episodes) if args.num_episodes else 1000
### NOTE if running twoparts (e.g. PT2 and 3), will run num_episodes * number of FT parts in total  
max_steps_per_episode = BATCH_SIZE
print(f'running training in {num_episodes} episodes, batch size {BATCH_SIZE}') #, {EPOCHS} epochs')

opponent_strategy = args.opp_strat 
print(f'will train with opponent strategy {opponent_strategy}')

payoff_version = args.payoff_version if args.payoff_version else 'largerR'
if payoff_version == 'largerR':
    PAYOFFMAT_IPD = [ [(120,120),(0,180)] , [(180,0),(60,60)] ] #IPD game - b:c=3.7:1, payoffmatwith0 - used in PART 2 
    #PAYOFFMAT_IPD = [ [(100,100),(0,127)] , [(127,0),(27,27)] ] #IPD game - b:c=3.7:1, payoffmatwith0 - used in PART 2 
    if do_PART3:
        r_other = args.r_other if args.r_other else 100 #for PART3
        r_punishment = -args.r_punishment if args.r_punishment else 83 #for PART3
        print('running with r_punishment = ', str(r_punishment), 'r_other = ', str(r_other))
    if do_PART4: 
        r_other = args.r_other if args.r_other else 100 #for PART3
        r_punishment = -args.r_punishment if args.r_punishment else -27 #for PART4
        print('running with r_punishment = ', str(r_punishment))

elif payoff_version == 'smallerR':
    PAYOFFMAT_IPD = [ [(3,3),(0,5)] , [(5,0),(1,1)] ] #IPD game - b:c=3:1, payoffmatwith0 - used in PART 2 
    if do_PART3:
        r_other = args.r_other if args.r_other else 0 #for PART3
        r_punishment = -args.r_punishment if args.r_punishment else -2 #2;4 #for PART3
        print('running with r_punishment = ', str(r_punishment), 'r_other = ', str(r_other))
    if do_PART4: 
        r_punishment = -args.r_punishment if args.r_punishment else -2 #4 #for PART4
        print('running with r_punishment = ', str(r_punishment))

#PAYOFFMAT_IPD = [ [(8,8),(0,10)] , [(10,0),(5,5)] ] #IPD game - b:c=...:1, payoffmatwith0
elif payoff_version == 'smallerRno0':
    PAYOFFMAT_IPD = [ [(3,3),(1,4)] , [(4,1),(2,2)] ]
else: 
    raise ValueError('payoff_version must be one of: largerR, smallerR, smallerRno0')
print(f'NB running with {payoff_version} payoffmat: {str(PAYOFFMAT_IPD)}')

r_illegal = -args.r_illegal if args.r_illegal else -0.0014 #-0.0036 #for PARTs1,2,3
print('running with r_illegal = ', str(r_illegal))

if do_PART1:
    r_C = args.r_C if args.r_C else 91.5 #for PART1
    r_D = args.r_D if args.r_D else 91.5 #for PART1

LoRA = args.LoRA if args.LoRA else False 
QLoRA = args.QLoRA if args.QLoRA else False
quantized = False if model_id == "gpt2" else True 
freeze_layers = True if model_id == "gpt2" else False 

device = 'cuda' if cuda.is_available() else 'cpu'
#print_cuda_mem_usage(device, episode=0, toprint='  MEM usage before model loading: ')

print(f'command line arguments parsed, running with option={OPTION}, quantized {quantized}, LoRA {LoRA}, QLoRA{QLoRA}, device {device}')
#C & D = random str (different one on each episode), freeze_layers {freeze_layers}, 


HF_HOME = '/models'
OUTPUT_DIR = f"./LLM_morality_output/{model_name}_FT_{PARTs_detail}_opp{opponent_strategy}{OPTION}/run{run_idx}"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

#save_model_pt1 = False
save_model_pt2 = True
save_model_pt3 = True
save_model_pt4 = True
if do_PART2 == True and do_PART3 == True: 
    save_model_pt3after2 = True
    save_model_pt2before3 = True 
    save_model_pt2 = False 
    save_model_pt3 = False 
else: 
    save_model_pt23 = True 

wandb.login(key=WANDB_API_KEY)

METHOD = 'PPO'

with open (f"{OUTPUT_DIR}/seed_run{run_idx}.txt", 'w') as f: 
    f.write(f'master_seed = {master_seed} \n')
    f.write(f'transformers_seed = {master_seed+1} \n')
    f.write(f'ppo_seed = {ppo_seed} \n')

with open (f"{OUTPUT_DIR}/payoffmat.txt", 'w') as f: 
    f.write('PAYOFFMAT_IPD = \n ')
    f.write('       C           D      \n')
    f.write('__________________________\n')
    f.write(f'C | {PAYOFFMAT_IPD[0]} \n')
    f.write(f'D | {PAYOFFMAT_IPD[1]} \n')




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



#WANDB_PROJECT = f"trl-{model_name}-{PARTs_detail}-{OPTION}-{payoff_version}" #PD-FT-{METHOD}
WANDB_PROJECT = f"trl-{model_name}-{PARTs_detail}-core-{OPTION}" #PD-FT-{METHOD}
os.environ['WANDB_PROJECT'] = WANDB_PROJECT
#run_name=f"oppon{opponent_strategy}_run{run_idx}_seqN{seq_N}"
#run_name=f"oppon{opponent_strategy}_run{run_idx}_seqN{seq_N}_{OPTION}"
run_name=f"oppon{opponent_strategy}_run{run_idx}"
if do_PART3: 
    run_name += moral_type
run = wandb.init(name=run_name, project=WANDB_PROJECT) #dir=OUTPUT_DIR
print('successfully set run_name in wandb.init()')

#if model not already on disc, download from hf and store 
if not os.path.exists(f"{HF_HOME}/{model_name}_1"):
    model_pretrained = AutoModelForCausalLM.from_pretrained(model_id,
                                                            quantization_config=bnb_config,
                                                            device_map={"": 0},
                                                            token=hf_token
                                                            )
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token

    model_pretrained.save_pretrained(f".{HF_HOME}/{model_name}_1", push_to_hub=False)
    tokenizer.save_pretrained(f".{HF_HOME}/{model_name}_1", push_to_hub=False)

    print(f'successfully saved (quantised) model and tokenizer to disc, in {HF_HOME}/{model_name}_1')
    del model_pretrained #delete the model_pretrained object to free up memory
    del tokenizer #delete the tokenizer object to free up memory


# ### Load model ####
# ###################

# In[7]:


print('loading ref model')
try: 
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(pretrained_model_name_or_path = model_id,
                                                              attn_implementation='eager',
                                                              device_map={"": 0},
                                                              is_trainable=False,
                                                              quantization_config=bnb_config,
                                                              )
    print('successfully loaded ref model from disc')
except Exception as e: 
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(pretrained_model_name_or_path = f".{HF_HOME}/{model_name}_1",
                                                              attn_implementation='eager',
                                                              device_map={"": 0},
                                                              is_trainable=False,
                                                              quantization_config=bnb_config,
                                                              )#.to(device)
    print('loaded ref model from the web')
#ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_pretrained,
#                                                device_map={"": 0},
#                                                is_trainable=False,
#                                                quantization_config=bnb_config,
#                                                )
#print_cuda_mem_usage(device, episode=0, toprint='  MEM usage after ref model loading: ')

#tokenizer = AutoTokenizer.from_pretrained(config.model_name)
try: 
    tokenizer = AutoTokenizer.from_pretrained(f".{HF_HOME}/{model_name}_1", token=hf_token)
    print('successfully loaded tokenizer from disc')
except Exception as e: 
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    print('loaded tokenizer from the web')
tokenizer.pad_token = tokenizer.eos_token

#for generating random seq to use as C & D symbols
if CD_tokens not in ['action12', 'action21', 'unused']:
    chars = tokenizer(list(string.ascii_uppercase+string.digits), add_special_tokens=False)['input_ids']


if gen_seq_freq == 'once per run': 
    C = generate_seq(N=seq_N, RN_stream=RN_stream_CDsymbols, chars=chars) #this is in input_ids format for the model 
    D = generate_seq(N=seq_N, RN_stream=RN_stream_CDsymbols, chars=chars)
    C_str = ''.join(tokenizer.batch_decode(C))
    D_str = ''.join(tokenizer.batch_decode(D))
    C_temp_str = C_str
    D_temp_str = D_str

    #placeholder IDs for generating IPD prompt in steps
    C_placeholder = C_str
    D_placeholder = D_str
    state_M_placeholder = '<unused2>'
    state_O_placeholder = '<unused3>'
    C_placeholder_id = C
    D_placeholder_id = D
    state_M_placeholder_id = tokenizer(state_M_placeholder, add_special_tokens=False)['input_ids']
    state_O_placeholder_id = tokenizer(state_O_placeholder, add_special_tokens=False)['input_ids']

elif gen_seq_freq == 'every episode':
    C_temp = generate_seq(N=seq_N, RN_stream=RN_stream_CDsymbols, chars=chars) 
    D_temp = generate_seq(N=seq_N, RN_stream=RN_stream_CDsymbols, chars=chars)
    C_temp_str = ''.join(tokenizer.batch_decode(C_temp))
    D_temp_str = ''.join(tokenizer.batch_decode(D_temp))

    #placeholder IDs for generating IPD prompt in steps 
    C_placeholder = '<unused0>' #text token 
    D_placeholder = '<unused1>'
    state_M_placeholder = '<unused2>'
    state_O_placeholder = '<unused3>'
    C_placeholder_id = tokenizer(C_placeholder, add_special_tokens=False)['input_ids']
    D_placeholder_id = tokenizer(D_placeholder, add_special_tokens=False)['input_ids']
    state_M_placeholder_id = tokenizer(state_M_placeholder, add_special_tokens=False)['input_ids']
    state_O_placeholder_id = tokenizer(state_O_placeholder, add_special_tokens=False)['input_ids']

else: #if not re-generating C & D symbols every episode
    if CD_tokens == 'unused': 
        C = int(7) #this is in input_ids format for the model 
        D = int(8)
        C_str = tokenizer.decode(C, skip_special_tokens=True)
        D_str = tokenizer.decode(D, skip_special_tokens=True)
    elif CD_tokens == 'action12':
        C_str = 'Action1'
        D_str = 'Action2'
        C = tokenizer.encode(C_str, add_special_tokens=False)
        D = tokenizer.encode(D_str, add_special_tokens=False)
        #seq_N = len(C)
        if seq_N != 2: 
            print('! NB seq_N !=2, force-setting to 2, not len(C) input_ids !')
            seq_N = 2
    elif CD_tokens == 'action21':
        C_str = 'Action2'
        D_str = 'Action1'
        C = tokenizer.encode(C_str, add_special_tokens=False)
        D = tokenizer.encode(D_str, add_special_tokens=False)
        if seq_N != 2: 
            print('! NB seq_N !=2, force-setting to 2, not len(C) input_ids !')
            seq_N = 2
    #C_temp_str = C_str
    #D_temp_str = D_str

    #placeholder IDs for generating IPD prompt in steps
    C_placeholder = C_str
    D_placeholder = D_str
    state_M_placeholder = '<unused2>'#*seq_N
    state_O_placeholder = '<unused3>'#*seq_N
    C_placeholder_id = C
    D_placeholder_id = D
    state_M_placeholder_id = tokenizer(state_M_placeholder, add_special_tokens=False)['input_ids']
    state_O_placeholder_id = tokenizer(state_O_placeholder, add_special_tokens=False)['input_ids']



IPD_QUERY_1 = create_structured_shortest_prompt_game_withstate_noeg(C_symbol=C_str, D_symbol=D_str, state_self=state_M_placeholder, state_opp=state_O_placeholder, RN_stream_1=RN_stream_1, RN_stream_2=RN_stream_2)
#IPD_QUERY_2 = create_structured_shortest_prompt_game_withstate_noeg(C_symbol=C, D_symbol=D, state_self=C, state_opp=D)
#IPD_QUERY_3 = create_structured_shortest_prompt_game_withstate_noeg(C_symbol=C, D_symbol=D, state_self=D, state_opp=C)
#IPD_QUERY_4 = create_structured_shortest_prompt_game_withstate_noeg(C_symbol=C, D_symbol=D, state_self=D, state_opp=D)
print(f'will train using the 4 QUERIES with state, e.g.: \n "{IPD_QUERY_1}"')
query_chat = process_prompt_for_gemma(IPD_QUERY_1)
query_encoded = tokenizer(query_chat, return_tensors="pt", add_special_tokens=False)['input_ids'].to(device)
query_text = tokenizer.decode(query_encoded.squeeze(0), skip_special_tokens=False)
       
with open (f"{OUTPUT_DIR}/IPD_prompt_withstateCC.txt", 'w') as f: 
        f.write(IPD_QUERY_1)
        f.write('\n')
        f.write('query processed with chat template: \n ')
        f.write(str(query_chat))
        f.write('\n')
        f.write('query_encoded: \n ')
        f.write(str(query_encoded.squeeze(0)))
        f.write('\n')
        f.write('query_decoded, with special tokens: \n ')
        f.write(query_text)
        f.close()

if args.wandblogfreq: #how often to log params and gradients to wandb
    log_freq = args.wandblogfreq
else: 
    log_freq = 100

#wandb.watch(model, log=args.wandblog, log_freq=log_freq)
if args.wandblog:
    OPTION += 'wandblog'
print(f'initialised wandb.watch(log={args.wandblog}, log_freq {log_freq}) from command line args')


# # Set up PPO Trainer ####
# ###########################

# In[8]:


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
    configscore_clip=0.5

#if 'MemoryDebug' in OPTION or 'MemoryEfficient' in OPTION:
config.optimize_cuda_cache=True
config.optimize_device_cache=True


#ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer=tokenizer) #, dataset=dataset, data_collator=collator) 
#ppo_trainer.config.reward_model = None #overwrite default sentiment analysis reward model
#ppo_trainer.config.query_dataset = None #overwrite default sentiment analysis dataset
print('successfully instantiated ppo_trainer with model & ref_model')


#if 'MemoryEfficient' in OPTION: 
    #from here https://github.com/huggingface/trl/blob/d57e4b726561e5ae58fdc335f34029052944a4a3/docs/source/customization.mdx#L190 
    # 2. Create optimizer
#    optimizer = bnb.optim.Adam8bit(model.parameters(), lr=config.learning_rate)
    # 3. modify trainer
#    ppo_trainer.optimizer=optimizer

    #### other options for optimising memory ####
    #ref_model = create_reference_model(model, num_shared_layers=6)


#NOTE the generation keywords are not currently being used! 
#generation_kwargs = {
#    "min_length": -1, # don't ignore the EOS token
#    "top_k": 0.0, # no top-k sampling
#    "top_p": 1.0, # no nucleus sampling
#    "do_sample": True, # yes, we want to sample
#    "pad_token_id": tokenizer.eos_token_id, # most decoder models don't have a padding token - use EOS token instead
#    #"max_new_tokens": 2, # specify how many tokens you want to generate at most
#    "temperature": 0.2,
#    "num_beams": 1,
#ß}


#output_min_length = 1
#output_max_length = 2
#output_length_sampler = LengthSampler(output_min_length, output_max_length)
#gen_len = output_length_sampler()



# In[9]:


def generate_valid_response_partner(model, tokenizer, query_input_tensors, top_p, max_retries=10):
    for _ in range(max_retries):
        response_tensor = respond_to_batch(model, query_input_tensors, txt_len=1, top_k = 0, top_p = top_p)
        response_text = tokenizer.decode(response_tensor.squeeze(0), skip_special_tokens=True)
        response_text_withspecial = tokenizer.decode(response_tensor.squeeze(0), skip_special_tokens=False)
        print(response_text)
        if response_text in ['Stay', 'Leave']:
            return response_tensor, response_text, response_text_withspecial
    return response_tensor, "invalid", response_text_withspecial 

def generate_valid_response_game(model, tokenizer, query_input_tensors, top_p, max_retries=10):
    for _ in range(max_retries):
        response_tensor = respond_to_batch(model, query_input_tensors, txt_len=2, top_k = 0, top_p = top_p)
        response_text = tokenizer.decode(response_tensor.squeeze(0), skip_special_tokens=True)
        response_text_withspecial = tokenizer.decode(response_tensor.squeeze(0), skip_special_tokens=False)
        print(response_text)
        if response_text in ['Action1', 'Action2']:
            return response_tensor, response_text, response_text_withspecial
    return response_tensor, "invalid", response_text_withspecial 


# ### Run fine-tuning - PART 2 ######
# ###################################

# In[10]:


if do_PART2: 
    results_df = pd.DataFrame(columns=['FineTuning_PART', 'episode', 'iteration', 'i', 'j', 'C_str', 'D_str', 'prev_move_i', 'prev_move_j', 'action_i', 'action_j', 'i_response_text', 'j_response_text', 'reward_i', 'reward_j'])
    results_idx = -1

    top_p = 1.0

    Nagent = 20
    Nround = 20
    
    #for episode in range(num_episodes): 
    episode = args.episode_input
    #print_cuda_mem_usage(device, episode, toprint=f'  MEM usage at start of episode {episode}: ')

    gen_len = seq_N #max(len(tokenizer.encode(C, add_special_tokens=False)), len(tokenizer.encode(D, add_special_tokens=False)))      
    print(f'\nstarting episode {episode},  C={C_str}, D={D_str}, gen_len={gen_len}')
    
    if episode == 0:
        
        groups = np.arange(Nagent)
        np.random.shuffle(groups)
        groups = groups.reshape((-1, 2))

        prev_move = [['Action1','Action2'][i] for i in np.random.choice([0, 1], Nagent)] #move player M made last time step
        #prev_move_partner = [['Leave','Stay'][i] for i in np.random.choice([0, 1], Nagent)]
    else:
        groups = np.load(f"groups_{episode-1}.npy")
        prev_move_npy = np.load(f"prev_move_{episode-1}.npy", allow_pickle=True)
        prev_move = prev_move_npy.tolist()
        

    print('initial prev_move_M: ', prev_move)
    
    agent_data = {
    i: {
        "batch_queries" : [],
        "batch_input_tensors_squeezed" : [],
        "batch_responses_squeezed" : [],
        "batch_responses_cpu" : [],
        "batch_responses_text" : [],
        "batch_responses_text_withspecial" : [],
        "batch_opp_moves" : [],
        "batch_rewards" : [],
    } for i in range(Nagent)}

    if 0 < episode < 100: 
        top_p = 1.0

    for batch_step in range(20):
        results_idx += 1
        #print_cuda_mem_usage(device, episode, toprint=f'  MEM usage at start of batch step {batch_step}: ')
        #query_text = generate_IPD_query(prev_move_M, prev_move_opp, model_name,C_str, D_str, RN_stream_1, RN_stream_2)

        print('round', batch_step)
        groupsStay = np.zeros((0, 2), dtype=int)
        groupsSwitch = np.array([], dtype=int)
        isSwitch = np.zeros(Nagent, dtype=int)
        for group in groups:
            
            results_idx += 1
            #print(group)
            i = group[0]
            j = group[1]
            
            if episode == 0:    
                model_i = AutoModelForCausalLMWithValueHead.from_pretrained(
                            pretrained_model_name_or_path=f".{HF_HOME}/{model_name}_1",
                            local_files_only=True,
                            attn_implementation='eager',
                            device_map={"": 0},
                            is_trainable=True,
                            peft_config=lora_config,
                            quantization_config=bnb_config
                        )     
                model_j = AutoModelForCausalLMWithValueHead.from_pretrained(
                            pretrained_model_name_or_path=f".{HF_HOME}/{model_name}_1",
                            local_files_only=True,
                            attn_implementation='eager',
                            device_map={"": 0},
                            is_trainable=True,
                            peft_config=lora_config,
                            quantization_config=bnb_config
                        )
                print('loading original model')
                
            else:
                prev_ep = 0 if episode % 2 == 0 else -1
                new_model_dir = f".{HF_HOME}/E{prev_ep}/{model_name}_{prev_ep}ep{i}agent"
                model_i = model = AutoModelForCausalLMWithValueHead.from_pretrained(new_model_dir, is_trainable = True)
                new_model_dir = f".{HF_HOME}/E{prev_ep}/{model_name}_{prev_ep}ep{j}agent"
                model_j = model = AutoModelForCausalLMWithValueHead.from_pretrained(new_model_dir, is_trainable = True)
                print('loading trained model')
                
            if batch_step == 0:
                query_text_i = generate_IPD_query2('Stay', 'Leave', RN_stream_1=RN_stream_1, RN_stream_2=RN_stream_2, opponent_actionPD = prev_move[j], num_round = batch_step, total_reward = np.sum(agent_data[i]["batch_rewards"]), agent_actionPD = prev_move[i], num_reward = 0)
            else:
                query_text_i = generate_IPD_query2('Stay', 'Leave', RN_stream_1=RN_stream_1, RN_stream_2=RN_stream_2, opponent_actionPD = prev_move[j], num_round = batch_step, total_reward = np.sum(agent_data[i]["batch_rewards"]), agent_actionPD = prev_move[i], num_reward = agent_data[i]["batch_rewards"][-1])
            query_input_tensors_i = tokenizer(query_text_i, return_tensors="pt", add_special_tokens=False)['input_ids'].to(device)
            if batch_step == 0:
                query_text_j = generate_IPD_query2('Stay', 'Leave', RN_stream_1=RN_stream_1, RN_stream_2=RN_stream_2, opponent_actionPD = prev_move[i], num_round = batch_step, total_reward = np.sum(agent_data[j]["batch_rewards"]), agent_actionPD = prev_move[j], num_reward = 0)
            else:
                query_text_j = generate_IPD_query2('Stay', 'Leave', RN_stream_1=RN_stream_1, RN_stream_2=RN_stream_2, opponent_actionPD = prev_move[i], num_round = batch_step, total_reward = np.sum(agent_data[j]["batch_rewards"]), agent_actionPD = prev_move[j], num_reward = agent_data[j]["batch_rewards"][-1])
            query_input_tensors_j = tokenizer(query_text_j, return_tensors="pt", add_special_tokens=False)['input_ids'].to(device)

            response_tensor_i, response_text_i, response_text_withspecial_i = generate_valid_response_partner(model_i, tokenizer, query_input_tensors_i, top_p, max_retries=10)
            
            #response_tensor_i = respond_to_batch(model_i, query_input_tensors_i, txt_len=1, top_k = 0, top_p = top_p)
            #response_text_i = tokenizer.decode(response_tensor_i.squeeze(0), skip_special_tokens=True)
            #response_text_withspecial_i = tokenizer.decode(response_tensor_i.squeeze(0), skip_special_tokens=False)
            reward_i = tensor(0, dtype=float)

            response_tensor_j, response_text_j, response_text_withspecial_j = generate_valid_response_partner(model_j, tokenizer, query_input_tensors_j, top_p, max_retries=10)

            reward_j = tensor(0, dtype=float)
            
            action_player_i = get_action_token_v3(response_text_i) #text form 
            action_player_j = get_action_token_v3(response_text_j)

            print("i,j", i, j)
            print("prev_move_i_j", prev_move[i],  prev_move[j])
            print("action_i_j", action_player_i, action_player_j)

            if action_player_i == 'Leave' or action_player_j == 'Leave':
                groupsSwitch = np.concatenate((groupsSwitch, group))
            else:
                groupsStay = np.concatenate((groupsStay, group[np.newaxis, :]), axis=0)

            results_df.loc[results_idx] = ['PT2 (Game reward)', episode, batch_step, i, j, str('Stay'), str('Leave'), prev_move[j], prev_move[i], action_player_i, action_player_j, response_text_i, response_text_j, reward_i.numpy().item(), reward_j.numpy().item()]

            agent_data[i]["batch_queries"].append(query_text_i)
            agent_data[i]["batch_input_tensors_squeezed"].append(query_input_tensors_i.squeeze(0))
            agent_data[i]["batch_responses_squeezed"].append(response_tensor_i.squeeze(0))
            agent_data[i]["batch_responses_cpu"].append(str(response_tensor_i.squeeze(0).cpu().detach().numpy()))
            agent_data[i]["batch_responses_text"].append(response_text_i)
            agent_data[i]["batch_responses_text_withspecial"].append(response_text_withspecial_i)
            agent_data[i]["batch_opp_moves"].append(prev_move[j])
            agent_data[i]["batch_rewards"].append(reward_i)#.numpy())

            agent_data[j]["batch_queries"].append(query_text_j)
            agent_data[j]["batch_input_tensors_squeezed"].append(query_input_tensors_j.squeeze(0))
            agent_data[j]["batch_responses_squeezed"].append(response_tensor_j.squeeze(0))
            agent_data[j]["batch_responses_cpu"].append(str(response_tensor_j.squeeze(0).cpu().detach().numpy()))
            agent_data[j]["batch_responses_text"].append(response_text_j)
            agent_data[j]["batch_responses_text_withspecial"].append(response_text_withspecial_j)
            agent_data[j]["batch_opp_moves"].append(prev_move[i])
            agent_data[j]["batch_rewards"].append(reward_j)#.numpy())

        np.random.shuffle(groupsSwitch)
        groupsSwitch = groupsSwitch.reshape((-1, 2))
        groups = np.concatenate((groupsStay, groupsSwitch), axis=0)

        for group in groups:

            results_idx += 1
            
            i = group[0]
            j = group[1]

            if episode == 0:    
                model_i = AutoModelForCausalLMWithValueHead.from_pretrained(
                            pretrained_model_name_or_path=f".{HF_HOME}/{model_name}_1",
                            attn_implementation='eager',
                            device_map={"": 0},
                            is_trainable=True,
                            peft_config=lora_config,
                            quantization_config=bnb_config
                        )     
                model_j = AutoModelForCausalLMWithValueHead.from_pretrained(
                            pretrained_model_name_or_path=f".{HF_HOME}/{model_name}_1",
                            attn_implementation='eager',
                            device_map={"": 0},
                            is_trainable=True,
                            peft_config=lora_config,
                            quantization_config=bnb_config
                        )
            else:
                prev_ep = 0 if episode % 2 == 1 else -1
                new_model_dir = f".{HF_HOME}/E{prev_ep}/{model_name}_{prev_ep}ep{i}agent"
                model_i = model = AutoModelForCausalLMWithValueHead.from_pretrained(new_model_dir, is_trainable = True)
                new_model_dir = f".{HF_HOME}/E{prev_ep}/{model_name}_{prev_ep}ep{j}agent"
                model_j = model = AutoModelForCausalLMWithValueHead.from_pretrained(new_model_dir, is_trainable = True)
            
            query_text_i = generate_IPD_query1(C_str, D_str, RN_stream_1=RN_stream_1, RN_stream_2=RN_stream_2, opponent_actionPD = prev_move[j], num_round = batch_step, total_reward = np.sum(agent_data[i]["batch_rewards"]), agent_actionPS = agent_data[i]["batch_responses_text"][-1], opponent_actionPS = agent_data[j]["batch_responses_text"][-1])
            query_input_tensors_i = tokenizer(query_text_i, return_tensors="pt", add_special_tokens=False)['input_ids'].to(device)
            #response_tensor_i = respond_to_batch(model_i, query_input_tensors_i, txt_len=gen_len, top_k = 0, top_p = top_p)
            #response_text_i = tokenizer.decode(response_tensor_i.squeeze(0), skip_special_tokens=True)
            #response_text_withspecial_i = tokenizer.decode(response_tensor_i.squeeze(0), skip_special_tokens=False)
            response_tensor_i, response_text_i, response_text_withspecial_i = generate_valid_response_game(model_i, tokenizer, query_input_tensors_i, top_p, max_retries=10)

            query_text_j = generate_IPD_query1(C_str, D_str, RN_stream_1=RN_stream_1, RN_stream_2=RN_stream_2, opponent_actionPD = prev_move[i], num_round = batch_step, total_reward = np.sum(agent_data[j]["batch_rewards"]), agent_actionPS = agent_data[j]["batch_responses_text"][-1], opponent_actionPS = agent_data[i]["batch_responses_text"][-1])
            query_input_tensors_j = tokenizer(query_text_j, return_tensors="pt", add_special_tokens=False)['input_ids'].to(device)
            response_tensor_j, response_text_j, response_text_withspecial_j = generate_valid_response_game(model_j, tokenizer, query_input_tensors_j, top_p, max_retries=10)

            #response_tensor_j = respond_to_batch(model_j, query_input_tensors_j, txt_len=gen_len, top_k = 0, top_p = top_p)
            #response_text_j = tokenizer.decode(response_tensor_j.squeeze(0), skip_special_tokens=True)
            #response_text_withspecial_j = tokenizer.decode(response_tensor_j.squeeze(0), skip_special_tokens=False)
            
            action_player_i = get_action_token_v3(response_text_i) #text form 
            action_player_j = get_action_token_v3(response_text_j) #text form 

            print("i,j", i, j)
            print("prev_move_i_j", prev_move[i],  prev_move[j])
            print("action_i_j", action_player_i, action_player_j)
            
            # Compute reward
            reward_i = reward_fn_game(action_player_i, action_player_j, C_str, D_str)#.to(device)
            reward_j = reward_fn_game(action_player_j, action_player_i, C_str, D_str)
            #print_cuda_mem_usage(device, episode, toprint=f'  MEM usage during batch step {batch_step}, after calulating reward_game: ')
            
            results_df.loc[results_idx] = ['PT2 (Game reward)', episode, batch_step, i, j, str(C_str), str(D_str), prev_move[j], prev_move[i], action_player_i, action_player_j, response_text_i, response_text_j, reward_i.numpy().item(), reward_j.numpy().item()]
            #print_cuda_mem_usage(device, episode, toprint=f'  MEM usage during batch step {batch_step}, after appending results to resuls_df: ')

            # Update state with M's move
            prev_move[i] = action_player_i
            prev_move[j] = action_player_j
            
            #print_cuda_mem_usage(device, episode, toprint=f'  MEM usage at the end of batch_step {batch_step}, before appending: ')

            agent_data[i]["batch_queries"].append(query_text_i)
            agent_data[i]["batch_input_tensors_squeezed"].append(query_input_tensors_i.squeeze(0))
            agent_data[i]["batch_responses_squeezed"].append(response_tensor_i.squeeze(0))
            agent_data[i]["batch_responses_cpu"].append(str(response_tensor_i.squeeze(0).cpu().detach().numpy()))
            agent_data[i]["batch_responses_text"].append(response_text_i)
            agent_data[i]["batch_responses_text_withspecial"].append(response_text_withspecial_i)
            agent_data[i]["batch_opp_moves"].append(action_player_j)
            agent_data[i]["batch_rewards"].append(reward_i)#.numpy())

            agent_data[j]["batch_queries"].append(query_text_j)
            agent_data[j]["batch_input_tensors_squeezed"].append(query_input_tensors_j.squeeze(0))
            agent_data[j]["batch_responses_squeezed"].append(response_tensor_j.squeeze(0))
            agent_data[j]["batch_responses_cpu"].append(str(response_tensor_j.squeeze(0).cpu().detach().numpy()))
            agent_data[j]["batch_responses_text"].append(response_text_j)
            agent_data[j]["batch_responses_text_withspecial"].append(response_text_withspecial_j)
            agent_data[j]["batch_opp_moves"].append(action_player_i)
            agent_data[j]["batch_rewards"].append(reward_j)#.numpy())
            #print_cuda_mem_usage(device, episode, toprint=f'  MEM usage at the end of batch_step {batch_step}, after appending: ')
            
    print('SUCCESS in running PART 2 (IPD fine-tuning) training loop')
    prev_ep = -1 if episode % 2 == 0 else 0
    results_df.to_csv(f'{OUTPUT_DIR}/During FT PART2 (Game rewards)_{episode}.csv') #this will overwrite previous files 
    torch.save(agent_data, f"tensor_dict_{episode}.pt")
    np.save(f"groups_{episode}.npy", groups)
    np.save(f"prev_move_{episode}.npy", np.array(prev_move))
    


# In[ ]:




