import json

# 定义 JSONL 文件路径
# train valid test
target_set = "valid"
function_file_path = "./Inter/time/" + target_set + "_c_cpp.jsonl"
dependency_file_path = "./Inter/output_c_cpp_final.jsonl"

save_file_path = function_file_path.split(".jsonl")[0] + "_dependency.jsonl"

# 打开文件
number= 0
with open(function_file_path, "r") as function_file:
    # 逐行读取文件
    for function_line in function_file:
        number +=1
        print(number)
        # 解析 JSON 数据
        function_data = json.loads(function_line)
        # 打印第一条数据
        if function_data['target'] == 1:
            function_id = function_data['function_id']
            # print(function_data['target'])
            # number += 1
            with open(dependency_file_path, "r") as dependency_file:
                for dependency_line in dependency_file:
                    dependency_data = json.loads(dependency_line)
                    dependency_id = dependency_data['function_id']
                    if function_id == dependency_id:
                        # print(f"Record with function_id {function_id} found in records file.")
                        print("属性名称：", list(dependency_data.keys()))
                        function_data['caller'] = dependency_data['caller']
                        function_data['callee'] = dependency_data['callee']
                        if 'caller_of_change' in dependency_data:
                            function_data['caller_of_change'] = dependency_data['caller_of_change']
                            print(function_data['caller_of_change'])
                        else:
                            function_data['caller_of_change'] = {}
                        if 'callee_of_change' in dependency_data:
                            function_data['callee_of_change'] = dependency_data['callee_of_change']
                            print(function_data['callee_of_change'])
                        else:
                            function_data['callee_of_change'] = {}
                            
                        
                        with open(save_file_path, "a") as save_file:
                            json.dump(function_data, save_file)
                            save_file.write("\n")
                        
                        # break
        # 如果你只想打印第一条数据，可以加上 break 来结束循环

