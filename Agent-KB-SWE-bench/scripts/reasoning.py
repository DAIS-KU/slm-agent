import json
import os
import requests

def call_model_api_1(prompt_model, base_url, api_key, detail, model_prompt):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    messages = [
        {"role": "system", "content": model_prompt},
        {"role": "user", "content": f"Detailed_description:\n{detail}\n\n"}
    ]

    payload = {
        "model": prompt_model,
        "messages": messages,
        "max_tokens": 3000,
        "temperature": 0.7,
        "response_format": {"type": "json_object"}
    }

    response = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload)
    return response.json()

def call_model_api_2(prompt_model, base_url, api_key, detail, code, truth, model_prompt):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    messages = [
        {"role": "system", "content": model_prompt},
        {"role": "user", "content": f"Detailed_description:\n{detail}\n\nCode:\n{code}\n\nAnswer:\n{truth}\n\n"}
    ]

    payload = {
        "model": prompt_model,
        "messages": messages,
        "max_tokens": 3000,
        "temperature": 0.7,
        "response_format": {"type": "json_object"}
    }

    response = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload)
    return response.json()

if __name__ == '__main__':
    prompt_model = "gpt-4.1"
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL") 


    input_file='your_benchmark_file_name'
    initial_file='your_hints_file_name'

    with open(input_file, 'r',encoding='utf-8') as f:
        data = json.load(f)

   

    model_1_prompt = (
    "This is a detailed language description of a certain python class. "
    "Please generate the code of this class based on this description. "
    "Note that the code must be a python class and conform to the description. "
    "Please respond strictly in JSON format with the following keys: 'code'."
    )

    model_2_prompt = (
        "There are currently two pieces of code. One has errors and the other is the correct version."
        "Please, based on the correct code, conduct a detailed analysis of the problems existing in the erroneous code,"
        "and systematically sort out the key points that need to be focused on during the code repair process, forming 10 code repair precautions."
        "The precautions need to be summarized from multiple dimensions and can be explained in combination with specific code cases, such as covering common issues like code indentation, type conversion, and exception handling, providing effective references for developers to solve similar code repair problems."
        "Please respond strictly in JSON format with the following keys: 'hints'.Be sure to pay attention to generating the correct json format."
        "Do not include a title in each hints; just output the explanation directly."
    )


    # model_3_prompt, model_4_prompt for another benchmark

    model_3_prompt = (
        "This problem description represents a requirement for code repair. We have classified various code repair problems in advance. "
        f" Please identify the {K} classifications that are most closely related to this description of the problem. That is, the {K} categories that this problem is most likely to belong to. "
        "Please respond strictly in JSON format with the following keys: 'categories'."
    )

    model_4_prompt = (
        "This problem_statement describes a code repair issue, and we have provided many hints of all possible correlations.  "
        f" Please summarize the key words of the problem based on the problem description, and then compare them with the key words of each hint, and select the {N} most important and most relevant hints from them "
        "Please respond strictly in JSON format with the following keys: 'hints'."
    )

    with open(initial_file, 'w') as file:
        for idx,entry in enumerate(data):
            response = call_model_api_1(
                prompt_model,
                base_url,
                api_key,
                entry["detailed_description"],
                model_1_prompt
            )
        
            code = response["choices"][0]["message"]["content"]
            hint = call_model_api_2(
                prompt_model,
                base_url,
                api_key,
                entry["detailed_description"],
                code,
                entry["ground_truth_class_body"],
                model_2_prompt
            )

            
           
            content_str = hint['choices'][0]['message']['content']
            content_dict = json.loads(content_str)
            hints = content_dict['hints']
        
        
            for hin in hints:
                file.write(f"{hin}\n")
            print(f"{idx + 1} entry has been completed")
    




