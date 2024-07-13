import ollama

res = ollama.chat(
    model='llava:13b',
    messages=[
        {
            'role': 'user',
            'content': 'Describe the narrative',
            'images': [r'.\spider.png']
        }

    ])

print(res['message']['content'])