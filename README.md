# 人民日报问答系统实现报告

## 1 获得数据

使用python爬虫抓取2023.5-2024.4人民日报的全部文章。参考project：https://github.com/858399075/pachong

保存为json格式方便读取。样例如下：

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

## 2 数据预处理

观察得到的数据，发现存在：1. 纯图报道无文字；2. “本版责编”页与“x月责编”页，均没有有效信息，把这些新闻全部删除。

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

## 3 构建文档检索模型

这一步先后尝试了基础的BM25与使用bert-based的dense retrieval。预训练模型先后尝试了bert-base-chinese和sbert-base-chinese-nli，但检索的效果均不如最基础的bm25，因此最终还是选择了bm25作为检索模型。

又为了提升检索效果，我对于文章的标题和内容分别构建了corpus，训练了两个BM25模型让其进行分级检索，取top2个最相关标题的文章与top5最相关内容的文章的并集，得到最终选定的新闻内容。

这一步还涉及到简单的query处理：遇到两个不同方向的query，比如读书日和海洋共同体，召回效果也很差。考虑在第一步拆分query，可以按照逗号划分，也可以直接丢给大模型让他自己处理，分别进行搜索。进行各种尝试后发现大模型拆分的query过于冗长，于是决定直接按逗号分隔。

## 4 文档内重排序

接下来考虑更细粒度的查询：返回召回的新闻时，能否先找到最相关的段落？给大模型的内容能不能缩减至一定长度。（比如512个字）。因此对返回的前k个文章内容要再做一些处理。

这一步使用了gte文本嵌入模型（https://modelscope.cn/models/iic/nlp_gte_sentence-embedding_chinese-base/summary）。按段落划分文章，对于query和每一段文章进行向量化，进而计算query与文本每一段的相似度。通过相似度来对每个段落进行重排序，每个文档取前k个段落作为最终的内容，考虑到模型输入token限制，要求k个段落的字数加起来不超过1024.最终，将精简后的文本一起作为input输入给文本生成模型。

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



## 5 文本生成模型

此处模型最开始尝试使用llama-3-chinese-8b-instruct，但奈何没有合适的数据集进行微调，导致回答效果很差，最终弃用。

最后选择比较聪明的gpt-4-turbo（调用openai的api-key实现）。

最后在测试集上能够达到em=0.9

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

拆分query效果：
![image-20240603171302789](/Users/redpanda/Library/Application Support/typora-user-images/image-20240603171302789.png)

回答效果：
![image-20240603171612678](/Users/redpanda/Library/Application Support/typora-user-images/image-20240603171612678.png)

## 6 最终结果

在评测问题上的EM=0.7625

![image-20240603163755377](/Users/redpanda/Library/Application Support/typora-user-images/image-20240603163755377.png)
