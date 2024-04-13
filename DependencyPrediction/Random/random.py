
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

json_file = './Inter/commit/test_c_cpp_dependency.jsonl'






def remove_comments(code):
    # 去除单行注释
    code = re.sub(r'//.*', '', code)
    # 去除多行注释
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    return code

def random_rag(code, rag_querys, number):
    
    if len(rag_querys) <= number:
        random_indexes = list(range(len(rag_querys)))
    else:
        random_indexes = random.sample(range(len(rag_querys)), number)


    return random_indexes 

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


acc_at_1_list =[]
acc_at_3_list =[]
acc_at_5_list =[]
acc_at_10_list =[]

coverage_at_1_list =[]
coverage_at_3_list =[]
coverage_at_5_list =[]
coverage_at_10_list =[]
for random_seed in range(100):
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
        for function_line in function_file:
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

            # print(type(rag_querys)) 
            random.seed(random_seed)
            rag_index = random_rag(code, rag_querys, 1)
            rag_name_list = []
            for i in range(len(rag_index)):
                rag_name_list.append(rag_name_querys[rag_index[i]])
            right_number_at_1, all_at_1, golden_at_1, accuracy_at_1 = accuracy_at_k(prediction_list=rag_name_list, golden_index_list=target_change_name_list, k=1)
            acc_num_at_1 += right_number_at_1
            acc_all_num_at_1 += all_at_1
            acc_golden_num_at_1 += golden_at_1

            rag_index = random_rag(code, rag_querys, 3)
            rag_name_list = []
            for i in range(len(rag_index)):
                rag_name_list.append(rag_name_querys[rag_index[i]])
            right_number_at_3, all_at_3, golden_at_3, accuracy_at_3 = accuracy_at_k(prediction_list=rag_name_list, golden_index_list=target_change_name_list, k=3)
            acc_num_at_3 += right_number_at_3
            acc_all_num_at_3 += all_at_3
            acc_golden_num_at_3 += golden_at_3

            rag_index = random_rag(code, rag_querys, 5)
            rag_name_list = []
            for i in range(len(rag_index)):
                rag_name_list.append(rag_name_querys[rag_index[i]])
            right_number_at_5, all_at_5, golden_at_5, accuracy_at_5 = accuracy_at_k(prediction_list=rag_name_list, golden_index_list=target_change_name_list, k=5)
            acc_num_at_5 += right_number_at_5
            acc_all_num_at_5 += all_at_5
            acc_golden_num_at_5 += golden_at_5

            rag_index = random_rag(code, rag_querys, 10)
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






    
