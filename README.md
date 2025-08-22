# RAG system design on People’s Daily corpus

## 1 Data collecting

Use Python web scraping to collect all articles from People’s Daily between May 2023 and April 2024.
Refer to the project: https://github.com/858399075/pachong.

Save the data in JSON format for easier reading.
An example format is shown below:

```json
[
    {
        "url": "http://paper.people.com.cn/rmrb/html/2023-05/01/nw.D110000renmrb_20230501_1-01.htm",
        "title": "在“五一”国际劳动节到来之际\n习近平向全国广大劳动群众致以节日的祝贺和诚挚的慰问",
        "content": "新华社北京4月30日电  在“五一”国际劳动节到来之际，中共中央总书记、国家主席、中央军委主席习近平代表党中央，向全国广大劳动群众致以节日的祝贺和诚挚的慰问。\n　　习近平强调，今年是全面贯彻党的二十大精神的开局之年，是实施“十四五”规划承前启后的关键之年。希望广大劳动群众大力弘扬劳模精神、劳动精神、工匠精神，诚实劳动、勤勉工作，锐意创新、敢为人先，依靠劳动创造扎实推进中国式现代化，在强国建设、民族复兴的新征程上充分发挥主力军作用。各级党委和政府要充分激发广大劳动群众的劳动热情和创新创造活力，切实保障广大劳动群众合法权益，用心帮助广大劳动群众排忧解难，推动全社会进一步形成崇尚劳动、尊重劳动者的良好氛围。"
    },
  ...
 ]
```

## 2 Data processing

Upon examining the collected data, the following issues were found:
	1.	Some reports contain only images without any text.
	2.	Pages such as “Editor of this page” and “Editor of month X” do not contain valid information.

All such news items should be removed.

```python
def clean_json_files(directory):
    for filename in tqdm(os.listdir(directory)):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            # 保留不符合条件的条目
            cleaned_data = [
                item for item in data if '责编' not in item['title'] and item['content'].strip() != '']

            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(cleaned_data, file, ensure_ascii=False, indent=4)
```

## 3 Document Retrieval Model Construction

In this stage, both traditional and neural retrieval approaches were explored. Specifically, I experimented with the classical BM25 algorithm as well as dense retrieval methods based on BERT. Two pretrained models, bert-base-chinese and sbert-base-chinese-nli, were evaluated. However, the performance of these dense retrieval models was consistently inferior to the baseline BM25. Consequently, BM25 was adopted as the final retrieval model.

To enhance retrieval effectiveness, separate corpora were constructed for article titles and article contents. Two BM25 models were trained accordingly, and a hierarchical retrieval strategy was employed: the union of the top 2 most relevant results based on titles and the top 5 most relevant results based on contents was selected as the final set of retrieved documents.

In addition, basic query preprocessing was introduced. When queries contained multiple, semantically divergent components (e.g., Reading Day and Maritime Community), recall performance deteriorated significantly. To address this, query splitting was considered. Two approaches were tested: (1) splitting queries by commas, and (2) delegating query decomposition to a large language model (LLM). Empirical results showed that LLM-generated sub-queries tended to be unnecessarily verbose, whereas comma-based splitting yielded more efficient results. Therefore, the latter approach was adopted.

## 4 Re-ranking

The next step focuses on finer-grained query matching: when returning retrieved news articles, can the system first identify the most relevant paragraphs? Moreover, can the content provided to the LLM be reduced to a manageable length (e.g., 512 characters)? To address these questions, additional processing was applied to the top-k retrieved articles.

For this step, I employed the **GTE sentence embedding model** (https://modelscope.cn/models/iic/nlp_gte_sentence-embedding_chinese-base/summary). 

Each article was divided into paragraphs, and both the query and each paragraph were vectorized using the embedding model. The similarity between the query and each paragraph was then computed. Based on these similarity scores, paragraphs were re-ranked, and the top-k paragraphs from each document were selected as the final content.

Considering the token length limitations of downstream models, the total length of the selected paragraphs was constrained to no more than 1,024 characters. Finally, the condensed text was aggregated and fed into the text generation model as input.

```python
def simplify_text(text, query, max_length):
    sentences = text.split('\n')

    # 获取每个句子的向量
    sentence_embeddings = get_sentence_embeddings(sentences)
    # 获取查询的向量
    query_embedding = get_sentence_embeddings([query])[0]
    # 计算查询和每个句子的相似度
    scores = np.dot(sentence_embeddings, query_embedding)
    # 按相似度排序
    sentence_scores = sorted(zip(scores, sentences),
                             key=lambda x: x[0], reverse=True)
    # 选择得分最高的句子，直到达到最大长度
    simplified_text = ""
    current_length = 0
    for score, sentence in sentence_scores:
        if current_length + len(sentence) > max_length:
            break
        simplified_text += sentence
        current_length += len(sentence)

    return simplified_text
```



## 5 Generation

nitially, I experimented with the **LLaMA-3-Chinese-8B-Instruct** model. However, due to the lack of a suitable dataset for fine-tuning, the model’s response quality was unsatisfactory, and it was eventually discarded.

As a replacement, I adopted **GPT-4-Turbo**, accessed via the OpenAI API, which demonstrated significantly better reasoning and generation capabilities.

On the test set, this approach achieved an **Exact Match (EM) score of 0.9**, indicating strong retrieval and answer generation performance.

---------

对于简单的问答题，以下是我调用模型的函数以及告诉他的prompt：

~~~python
def answer_question(question, context):
    prompt = f'''你是一个知识渊博且简洁的助手。你有很多基本的知识。
    请基于```内的内容简洁回答问题。
    ```内容格式为：doc1:\n url:\n title:\n content:\n doc2:... 每个doc都有url、title和content。
    你只需要回答问题的答案，不需要重复问题或提供额外的解释。例如，如果答案是数字，那么就只回答数字本身，不需要包含前后背景。
    现在是2024年，去年是2023年。doc的url格式如下：
    http://paper.people.com.cn/rmrb/html/yyyy-mm/dd/nw....
    如果报道中没有提到事件的日期，你可以通过url中的日期（yyyy-mm/dd）来判断报道的时间，进而推断报道中提到事件的具体日期。
    当问题问到到日期时，回答需要包含年份。
    在回答问题时，如果```内的内容不能支持你完全正确的回答问题，请使用你本身的知识回答。
    ```
    {context}
    ```
    我的问题是：{question}。
    '''
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            model="gpt-4-turbo",
            max_tokens=150,
            temperature=0.7,
            n=1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error: {e}")
        return "抱歉，我无法回答这个问题。"
~~~

----

对于开放性问题，我的思路是让gpt-4自己拆分query，分别搜索+整合结果。这个过程通过提供精准的prompt来实现。

~~~python
def split_query(query):
    prompt = f'''你是一个乐于助人的人民日报新闻问答小助手，现在你需要根据我的问题，将一个模糊的问题分解为几个可以精准搜索的子问题，最好为关键词组合的形式，但不要只有一个词。子问题不超过五个，方便我在语料库中进一步使用bm25搜索。
    我的问题是：{query}
    请严格按照python列表的格式返回生成的子查询，不要说其他的话，只要结果。也不要生成代码块，只要字符串格式的，我要将返回的内容直接转换为python列表。
    '''

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            model="gpt-4-turbo",
            max_tokens=150,
            temperature=0.7,
            n=1
        )
        print(response.choices[0])
        sub_queries = eval(response.choices[0].message.content.strip())
        print(sub_queries)
        return sub_queries
    except Exception as e:
        print(f"Error: {e}")
        return []


def answer_question(question, context):
    prompt = f'''你是一个乐于助人的人民日报新闻问答小助手。请你根据我提供的文本信息，帮我对以下开放性问题进行全面、清晰、连贯、流畅的解答。
    回答的内容不超过512个字，并提供你认为最有帮助的文档url。
    文本内容：```
    {context}
    ```
    问题：{question}。'''

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            model="gpt-4-turbo",
            max_tokens=1024,
            temperature=0.7,
            n=1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error: {e}")
        return "抱歉，我无法回答这个问题。"
~~~


## 6 Result
query spliting:
<img width="774" height="123" alt="image" src="https://github.com/user-attachments/assets/a8355636-bfd9-4f7b-9d7d-1d4e1e671223" />

answer：
<img width="770" height="423" alt="image" src="https://github.com/user-attachments/assets/1555209b-8d06-447f-9283-4914aaffd308" />



EM=0.7625 on eval set:
<img width="750" height="123" alt="image" src="https://github.com/user-attachments/assets/4c1ca91c-69c5-49d3-957b-0550adc20b48" />

