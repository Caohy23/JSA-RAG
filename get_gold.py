import openai
import json
from tqdm import tqdm



api_key=''
base_url = ''
client = openai.OpenAI(
api_key=api_key )
def get_best_passage_id(data):
    # 构建向 GPT - 4o 发送的提示信息
    prompt = f"Question: {data['question']}, Provided Answers: {', '.join(data['answers'])}. Please select the ID of the passage that best answers the question from the following paragraphs. If there is no passage you think can generate the correct answer, select the ID of the passage that comes closest to answering the question. Note!!! Only return the value of the passage's id key."
    for i, passage in enumerate(data['passages']):
        prompt += f"\nparagraph {i + 1}: {passage}"

    try:
        # 调用 GPT - 4o API
        response = client.chat.completions.create(
            model="gpt-4o",  # 确保你有访问 GPT - 4o 的权限
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        # 提取 GPT - 4o 的回复
        gpt_response = response.choices[0].message.content
        #print(f"GPT的回复：{gpt_response}")
        # 解析回复，尝试找到最佳段落的 id
        for passage in data['passages']:
            if passage['id'] in gpt_response:
                return passage['id']

        return None

    except Exception as e:
        print(f"调用 API 时出现错误: {e}")
        return None

# 示例输入数据
input_data = {
    "question": "total number of death row inmates in the us",
    "answers": ["2,718"],
    "passages": [
        {
            "id": "24347960",
            "title": " ",
            "section": " ",
            "text": "List of death row inmates in the United States   As of December 7, 2018, there were 2,667 death row inmates in the United States. The number of death row inmates changes daily with new convictions, appellate decisions overturning conviction or sentence alone, commutations, or deaths (through execution or otherwise). Due to this fluctuation as well as lag and inconsistencies in inmate reporting procedures across jurisdictions, the information in this article may be out of date."
        },
        {
            "id": "24347968",
            "title": " ",
            "section": " ",
            "text": "List of death row inmates in the United States Federal Currently on death row: 62 ; Total number executed: 37 (1927–2003) List of federal death row inmates  Due to the high number of federal death row inmates, only prisoners with Wikipedia pages are listed on this page. A full list is externally linked: "
        }
    ]
}


best_passage_id = get_best_passage_id(input_data)
if best_passage_id:
    print("最佳段落的 id:")
    print(best_passage_id)
else:
    print("未找到最佳段落。")

input_filename = f'D:\data\\tasi\\train_data\\tasi_rag_platform-master\data\\test_with_top100.jsonl'
output_filename = f'D:\data\\tasi\\train_data\\tasi_rag_platform-master\data\\test_with_top100.jsonl'



i = 0
data = []
with open(input_filename, 'r', encoding='utf-8') as file:
    for line in file:
        # i +=1
        # if i ==10:
        #     break
        try:
            # 解析每行的 JSON 数据
            item = json.loads(line)
            data.append(item)
        except json.JSONDecodeError:
            print(f"解析行时出错: {line}")

# 提取 question 和 answer 对
question_answer_pairs = []

# 使用 tqdm 包装循环，添加进度条
for item in tqdm(data, desc="处理数据", unit="项"):
    best_passage_id = get_best_passage_id(item) 
    best_passage=None # 这里假设输入是 item 而不是 input_data
    for passage in item["passages"]:
        if passage["id"]== best_passage_id:
            best_passage = passage
            break
    if best_passage==None:
        continue
    new_json = {
        "question": item["question"],
        "answers": item["answers"],
        "gold_passage":best_passage,
        "gold_doc": best_passage_id
    }
    question_answer_pairs.append(new_json)


with open(output_filename, 'w', encoding='utf-8') as outfile:
    for item in tqdm(question_answer_pairs, desc="写入文件", unit="项"):
        outfile.write(json.dumps(item) + '\n')

print(f'数据已成功转换并保存到 {output_filename}')
# 读取JSON文件
    # with open('D:\data\\tasi\\train_data\\tasi_rag_platform-master\doc_name\doc_name\\rag_format\\test.json', 'r') as file:
    #     data = json.load(file)
    # print(data[0])
    # #assert 1==0
    # # 提取question和answer对
    # question_answer_pairs = []
    # for item in data:
    #     question = item["dialog"]
    #     print(question)
    #     assert 1==0
    #     answer = item["response"]
    #     new_json = {
    #         "question": question,
    #         "answers": [answer]
    #     }
    #     question_answer_pairs.append(new_json)
    # output_filename = 'D:\data\\tasi\\train_data\\tasi_rag_platform-master\doc_name\doc_name\\rag_format\\test.jsonl'

    # with open(output_filename, 'w', encoding='utf-8') as outfile:
    #     for item in question_answer_pairs:
    #         outfile.write(json.dumps(item) + '\n')
    # print(f'数据已成功转换并保存到 {output_filename}')


# 打印question和answer对
# for pair in question_answer_pairs:
#     print(f"Question: {pair[0]}")
#     print(f"Answer: {pair[1]}")
#     print()
