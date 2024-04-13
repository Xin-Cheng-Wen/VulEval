import jsonlines
import json
import sys
from tqdm import tqdm
import traceback
from time import sleep
import time

import os
import hashlib
import re
import random
from typing import Union
import torch
from transformers import AutoTokenizer, AutoModel
from difflib import SequenceMatcher
from model.unixcoder import UniXcoder

json_file = './Inter/time/test_c_cpp_dependency.jsonl'






def edit_similarity(
    code1: Union[str, list], 
    code2: Union[str, list], 
    tokenizer: AutoTokenizer = None
    ) -> float:

    # Check input types and tokenize as needed
    if isinstance(code1, str):
        assert tokenizer, "tokenizer must be provided if input is string"
        code1 = tokenizer.tokenize(code1)
    elif isinstance(code1, list):
        pass

    if isinstance(code2, str):
        assert tokenizer, "tokenizer must be provided if input is string"
        code2 = tokenizer.tokenize(code2)
    elif isinstance(code2, list):
        pass

    # compute and return the similarity ratio
    return SequenceMatcher(None, code1, code2).ratio()
    

def jaccard_similarity(
    code1: Union[str, list],
    code2: Union[str, list],
    tokenizer: AutoTokenizer = None
    ) -> float:

    # Check input types and tokenize/de-duplicate as needed
    if isinstance(code1, str):
        assert tokenizer, "tokenizer must be provided if input is string"
        code1 = set(tokenizer.tokenize(code1))
    elif isinstance(code1, list):
        code1 = set(code1)

    if isinstance(code2, str):
        assert tokenizer, "tokenizer must be provided if input is string"
        code2 = set(tokenizer.tokenize(code2))
    elif isinstance(code2, list):
        code2 = set(code2)

    try:
        return len(code1.intersection(code2)) / len(code1.union(code2))
    except ZeroDivisionError:
        print("ZeroDivisionError")
        print(code1, code2)
        return 0



def cosine_similarity(
        embedding1,
        embedding2
        ):
    """
    Calculate the cosine similarity between two embeddings
    """
    # check the input to be tensor
    assert isinstance(embedding1, torch.Tensor), "embedding1 must be a tensor"
    assert isinstance(embedding2, torch.Tensor), "embedding2 must be a tensor"

    # calculate the cosine similarity
    return torch.cosine_similarity(embedding1, embedding2, dim=0).item()




def remove_comments(code):
    # 去除单行注释
    code = re.sub(r'//.*', '', code)
    # 去除多行注释
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    return code



def semantic_rag(code, rag_querys, rag_name_querys, tokenizer, model, k):
    # defualt lexical retrieval, no need to load the model

    code_loc = code.split("\n")
    
    # last_lines = code_loc[-10:]
    # code = "\n".join(last_lines)
    
    


    similarity = "cosine"
    rag_indexes = []
    rag_indexes = retrieve(
                        code=code,
                        candidates=rag_querys, 
                        tokenizer=tokenizer,
                        model=model, 
                        max_length=max_length,
                        similarity=similarity)
    if len(rag_indexes) <= k:
        top_k_rag_indexes = list(range(len(rag_indexes)))
    else:
        top_k_rag_indexes = rag_indexes[:k]

    return top_k_rag_indexes 


def accuracy_at_k(prediction_list, golden_index_list, k):
    """
    This function computes the accuracy at k. It returns a float value between 0 and 1 indicating the
    accuracy at k, where a value of 1 means the correct code is retrieved at the top k positions and
    a value of 0 means the correct code is not retrieved at the top k positions.
    
    Args:
    prediction_list: list, a list of lists, where each list contains the indices of the retrieved codes.
    golden_index_list: list, a list of integers, where each integer is the index of the correct code.
    k: int, the number of retrieved codes.
    
    Returns:
    Float, the accuracy at k.
    """
    
    if len(golden_index_list) == 0:
        raise ValueError("The list of golden indices should not be empty.")
    
    # assert len(golden_index_list) == len(prediction_list), \
    #     "The length of the golden indices list should be equal to the length of the prediction list, however, " \
    #     f"the length of the golden indices list is {len(golden_index_list)} and the length of the prediction list is {len(prediction_list)}."
    

    acc = 0
    index_list = []
    for i in range(len(prediction_list)):
        index_list.append(prediction_list[i])

    if len(index_list) < k:
        top_k_indices = index_list
    else:
        top_k_indices = index_list[:k]

    for i in range(len(golden_index_list)):
        golden_index = golden_index_list[i]


        if golden_index not in top_k_indices:
            continue
        else:
            acc += 1


    if len(golden_index_list) < k:
        return acc, len(golden_index_list), len(golden_index_list), round(acc / len(golden_index_list), 5)
    else:
        return acc, k, len(golden_index_list), round(acc / k, 5)



def retrieve(
    code,
    candidates: list,
    tokenizer = None,
    model = None,
    max_length: int = None,
    similarity: str = "cosine"
    ):
    
    # check if the similarity is valid
    assert similarity in ["edit", "jaccard", "cosine"], "similarity must be one of edit, jaccard, cosine"

    if similarity == "cosine":
        assert model, "model must be provided if similarity is cosine"

        def get_embedding(code):
            if not code:
                # the type of model
                if isinstance(model, UniXcoder):
                    # this is for unixcoder
                    return torch.zeros(model.config.hidden_size).to(model.model.device)
                else:
                    return torch.zeros(model.config.hidden_size).to(model.device)

            if isinstance(model, UniXcoder):
                # this is for unixcoder
                tokens_ids = model.tokenize([code], max_length=max_length, mode="<encoder-only>")
                source_ids = torch.tensor(tokens_ids).to(model.model.device)
                
                with torch.no_grad():
                    _, code_embeddings = model(source_ids) # [1, 768]
                
                code_embeddings = torch.squeeze(code_embeddings) # [hidden_size]

            else:
                code_tokens=tokenizer.tokenize(code, max_length=max_length, truncation=True)
                tokens_ids=tokenizer.convert_tokens_to_ids(code_tokens)
                with torch.no_grad():
                    code_embeddings=model(torch.tensor(tokens_ids)[None,:].to(model.device))[0] # [1, seq_len, hidden_size]
                # calculate the mean of the embeddings
                code_embeddings = torch.mean(code_embeddings, dim=1) # [1, hidden_size]
                code_embeddings = torch.squeeze(code_embeddings) # [hidden_size]
            return code_embeddings
        
        code_embedding = get_embedding(code)
        candidates_embeddings = [ get_embedding(candidate) for candidate in candidates ]

        # calculate the cosine similarity between the code and the candidates
        sim_scores = []
        for i, candidate_embedding in enumerate(candidates_embeddings):
            sim_scores.append((i, cosine_similarity(code_embedding, candidate_embedding)))
    else:
        # candidates is a list of code strings
        # we need to sort the candidate index based on the edit similarity in a descending order
        sim_scores = []
        for i, candidate in enumerate(candidates):
           
            candidate
            # print(candidate)

            
            if similarity == "edit":
                sim_scores.append((i, edit_similarity(code, candidate, tokenizer)))
            elif similarity == "jaccard":
                sim_scores.append((i, jaccard_similarity(code, candidate, tokenizer)))
        
    
    # sort the candidate index based on the edit similarity in a descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # print(sim_scores)
    # only return the index
    ranks = [ index for index, score in sim_scores ]

    return ranks



   



acc_at_1_list =[]
acc_at_3_list =[]
acc_at_5_list =[]
acc_at_10_list =[]

coverage_at_1_list =[]
coverage_at_3_list =[]
coverage_at_5_list =[]
coverage_at_10_list =[]

model_name = "./microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="cache")
model = None
max_length = 512

if "unixcoder" in model_name:
    model = UniXcoder(model_name)
    model.to("cuda")
else:
    model = AutoModel.from_pretrained(model_name, cache_dir="cache")
for random_seed in range(1):
    acc_num_at_1 = 0
    acc_all_num_at_1 = 0
    acc_golden_num_at_1 = 0

    acc_num_at_3 = 0
    acc_all_num_at_3 = 0
    acc_golden_num_at_3 = 0

    acc_num_at_5 = 0
    acc_all_num_at_5 = 0
    acc_golden_num_at_5 = 0

    acc_num_at_10 = 0
    acc_all_num_at_10 = 0
    acc_golden_num_at_10 = 0

    with open(json_file, "r") as function_file:
        # 逐行读取文件
        # for function_line in function_file:
        
        for function_line in tqdm(function_file):
            function_data = json.loads(function_line)

            code = function_data['function']
            code = remove_comments(code)

            caller = function_data['caller']
            callee = function_data['callee']

            change_caller = function_data['caller_of_change']
            change_callee = function_data['callee_of_change']
            if len(change_caller) == 0 and len(change_callee) == 0:
                continue
            else:
                target_change_list = []
                target_change_name_list = []
                if len(change_caller) != 0:
                    for key, value in change_caller.items():
                        target_change_list.append(value)
                        target_change_name_list.append(key)
                if len(change_callee) != 0:
                    for key, value in change_callee.items():
                        target_change_list.append(value)
                        target_change_name_list.append(key)

            caller_list = []
            caller_name_list = []
            callee_list = []
            callee_name_list = []
            if len(caller) != 0:
                for key, value in caller.items():
                    # print("Key:", key, "Value:", value)
                    caller_list.append(value)
                    caller_name_list.append(key)
            if len(callee) != 0:
                for key, value in callee.items():
                    callee_list.append(value)
                    callee_name_list.append(key)

            if len(callee_list) != 0 and len(caller_list) != 0:
                rag_querys = caller_list + callee_list
                rag_name_querys = caller_name_list + callee_name_list

            elif len(callee_list) != 0:
                rag_querys = callee_list
                rag_name_querys = callee_name_list

            elif len(caller_list) != 0:
                rag_querys = caller_list
                rag_name_querys = caller_name_list
            else:
                continue

            
           
            rag_index = semantic_rag(code, rag_querys, rag_name_querys, tokenizer= tokenizer, model = model, k = 1)
            
            rag_name_list = []
            for i in range(len(rag_index)):
                rag_name_list.append(rag_name_querys[rag_index[i]])
            right_number_at_1, all_at_1, golden_at_1, accuracy_at_1 = accuracy_at_k(prediction_list=rag_name_list, golden_index_list=target_change_name_list, k=1)
            acc_num_at_1 += right_number_at_1
            acc_all_num_at_1 += all_at_1
            acc_golden_num_at_1 += golden_at_1

            rag_index = semantic_rag(code, rag_querys, rag_name_querys, tokenizer= tokenizer, model = model, k = 3)
            rag_name_list = []
            for i in range(len(rag_index)):
                rag_name_list.append(rag_name_querys[rag_index[i]])
            right_number_at_3, all_at_3, golden_at_3, accuracy_at_3 = accuracy_at_k(prediction_list=rag_name_list, golden_index_list=target_change_name_list, k=3)
            acc_num_at_3 += right_number_at_3
            acc_all_num_at_3 += all_at_3
            acc_golden_num_at_3 += golden_at_3

            rag_index = semantic_rag(code, rag_querys, rag_name_querys, tokenizer= tokenizer, model = model, k = 5)
            rag_name_list = []
            for i in range(len(rag_index)):
                rag_name_list.append(rag_name_querys[rag_index[i]])
            right_number_at_5, all_at_5, golden_at_5, accuracy_at_5 = accuracy_at_k(prediction_list=rag_name_list, golden_index_list=target_change_name_list, k=5)
            acc_num_at_5 += right_number_at_5
            acc_all_num_at_5 += all_at_5
            acc_golden_num_at_5 += golden_at_5

            rag_index = semantic_rag(code, rag_querys, rag_name_querys, tokenizer= tokenizer, model = model, k = 10)
            rag_name_list = []
            for i in range(len(rag_index)):
                rag_name_list.append(rag_name_querys[rag_index[i]])
            right_number_at_10, all_at_10, golden_at_10, accuracy_at_10 = accuracy_at_k(prediction_list=rag_name_list, golden_index_list=target_change_name_list, k=10)
            acc_num_at_10 += right_number_at_10
            acc_all_num_at_10 += all_at_10
            acc_golden_num_at_10 += golden_at_10
            

        accuracy_at_1 = acc_num_at_1/acc_all_num_at_1
        acc_at_1_list.append(accuracy_at_1)
        coverage_at_1 = acc_num_at_1/acc_golden_num_at_1
        coverage_at_1_list.append(coverage_at_1)


        accuracy_at_3 = acc_num_at_3/acc_all_num_at_3
        acc_at_3_list.append(accuracy_at_3)
        coverage_at_3 = acc_num_at_3/acc_golden_num_at_3
        coverage_at_3_list.append(coverage_at_3)

        accuracy_at_5 = acc_num_at_5/acc_all_num_at_5
        acc_at_5_list.append(accuracy_at_5)
        coverage_at_5 = acc_num_at_5/acc_golden_num_at_5
        coverage_at_5_list.append(coverage_at_5)

        accuracy_at_10 = acc_num_at_10/acc_all_num_at_10
        acc_at_10_list.append(accuracy_at_10)
        coverage_at_10 = acc_num_at_10/acc_golden_num_at_10
        coverage_at_10_list.append(coverage_at_10)
        # print(accuracy_at_1)

print(round(sum(acc_at_1_list)/len(acc_at_1_list)*100,2))
print(round(sum(acc_at_3_list)/len(acc_at_3_list)*100,2))
print(round(sum(acc_at_5_list)/len(acc_at_5_list)*100,2))
print(round(sum(acc_at_10_list)/len(acc_at_10_list)*100,2))




print(round(sum(coverage_at_1_list)/len(coverage_at_1_list)*100,2))
print(round(sum(coverage_at_3_list)/len(coverage_at_3_list)*100,2))
print(round(sum(coverage_at_5_list)/len(coverage_at_5_list)*100,2))
print(round(sum(coverage_at_10_list)/len(coverage_at_10_list)*100,2))
