import json
import os
import requests

def call_model_api_1(prompt_model, base_url, api_key, task_description, student_output, hints_data, model_prompt):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    messages = [
        {"role": "system", "content": model_prompt},
        {"role": "user", "content": f"Task-description:\n{task_description}\n\nStudent-output:\n{student_output}\n\nHints-data:\n{hints_data}\n"}
    ]

    payload = {
        "model": prompt_model,
        "messages": messages,
        "max_tokens": 30000,
        "temperature": 0.7,
        "response_format": {"type": "json_object"}
    }

    response = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload)
    return response.json()



if __name__ == '__main__':
    prompt_model = "gpt-4.1"
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL") 


    task_description_file='your_task_description_file_name'
    student_output_file='your_student_model_output_file_name'
    hints_file='your_hints_file_name'

    teacher_hints_file='your_teacher_hints_file_name'

    with open(hints_file, 'r',encoding='utf-8') as f:
        hints_data = json.load(f)
    with open(task_description_file, 'r',encoding='utf-8') as taskf:
        task_description = taskf.read()
    with open(student_output_file, 'r',encoding='utf-8') as studentf:
        student_output = studentf.read()
    k = 20

    model_1_prompt = (
    "Here is the task description for the code repair task, the answer provided by the student model, as well as the prompt information used in the model's work."
    "The answers provided by the model have some issues. Now, please analyze the reasons for the errors based on the problems presented in the model's answer. "
    "And based on the errors that the model is most likely to make, re-determine the importance of each hint for the model.  "
    f"And after sorting the hints by their importance, select the f{k} most important hints.  "
    )
    
    with open(teacher_hints_file,'w',encoding='utf-8') as file:
        with open(hints_file, 'r',encoding='utf-8') as f:
            a = 1
            for line in f:
                line = line.strip()
                response = call_model_api_1(
                    prompt_model,
                    base_url,
                    api_key,
                    task_description,
                    student_output,
                    hints_data,
                    model_1_prompt
                )

                re = response["choices"][0]["message"]["content"]
                re = json.loads(re)
                re["hint"] = line
                file.write(f"{re}\n")
                print(f"{a} has been completed.")
                a = a + 1


    
   




