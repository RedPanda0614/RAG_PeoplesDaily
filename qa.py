import os
import json
import glob
from rank_bm25 import BM25Okapi
import jieba
import pickle
from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import numpy as np
import openai
from openai import OpenAI
import tiktoken  # 新增的库

# 设置OpenAI API密钥
client = OpenAI(
    base_url="",
    api_key=''
)

# 禁用tokenizers的并行化
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 使用本地模型路径
model_path = "./nlp_gte_sentence-embedding_chinese-base"
pipeline_se = pipeline(Tasks.sentence_embedding,
                       model=model_path, sequence_length=512)


def load_data_from_json(directory):
    data = []
    files = glob.glob(os.path.join(directory, "*.json"))
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            month_data = json.load(f)
            data.extend(month_data)
    return data


def prepare_corpus(data):
    content_corpus = []
    title_corpus = []
    for item in data:
        title_text = item['title']
        combined_text = item['title'] + " " + item['content']
        segmented_title = jieba.lcut(title_text)
        segmented_text = jieba.lcut(combined_text)
        title_corpus.append(segmented_title)
        content_corpus.append(segmented_text)
    return title_corpus, content_corpus


def train_bm25(corpus):
    bm25 = BM25Okapi(corpus)
    return bm25


def search_bm25(bm25, query, data, n=5, max_length=1024):
    segmented_query = jieba.lcut(query)
    scores = bm25.get_scores(segmented_query)
    top_n = sorted(range(len(scores)),
                   key=lambda i: scores[i], reverse=True)[:n]
    results = [data[i] for i in top_n]

    for result in results:
        result['content'] = simplify_text(result['content'], query, max_length)

    results_with_query = [{"query": query, "url": result['url'],
                           "title": result['title'], "content": result['content']} for result in results]
    return results_with_query


def get_sentence_embeddings(sentences):
    inputs = {"source_sentence": sentences}
    result = pipeline_se(input=inputs)
    return result['text_embedding']


def simplify_text(text, query, max_length):
    sentences = text.split('\n')
    simplified_text = sentences[0]
    current_length = len(sentences[0])
    if len(sentences) > 1:
        sentence_embeddings = get_sentence_embeddings(sentences[1:])
        query_embedding = get_sentence_embeddings([query])[0]
        scores = np.dot(sentence_embeddings, query_embedding)
        sentence_scores = sorted(zip(scores, sentences[1:]),
                                 key=lambda x: x[0], reverse=True)
        for score, sentence in sentence_scores:
            if current_length + len(sentence) > max_length:
                break
            simplified_text += sentence
            current_length += len(sentence)
    return simplified_text


def save_model(bm25_title, bm25_content, title_corpus, content_corpus, model_path_title, model_path_content, corpus_path_title, corpus_path_content):
    with open(model_path_title, 'wb') as f:
        pickle.dump(bm25_title, f)
    with open(model_path_content, 'wb') as f:
        pickle.dump(bm25_content, f)
    with open(corpus_path_title, 'wb') as f:
        pickle.dump(title_corpus, f)
    with open(corpus_path_content, 'wb') as f:
        pickle.dump(content_corpus, f)
    print(f'Model and corpus saved')


def load_model(model_path_title, model_path_content, corpus_path_title, corpus_path_content):
    with open(model_path_title, 'rb') as f:
        bm25_title = pickle.load(f)
    with open(model_path_content, 'rb') as f:
        bm25_content = pickle.load(f)
    with open(corpus_path_title, 'rb') as f:
        title_corpus = pickle.load(f)
    with open(corpus_path_content, 'rb') as f:
        content_corpus = pickle.load(f)
    return bm25_title, bm25_content, title_corpus, content_corpus


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

# 开放性问题
def split_query(query):
    prompt = f'''你是一个乐于助人的人民日报新闻问答小助手，现在你需要根据我的问题，将一个模糊的问题分解为几个可以精准搜索的子问题，最好为关键词形式，越精简越好。方便我在语料库中进一步搜索。我的问题是：{query}'''

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            model="gpt-4",
            max_tokens=150,
            temperature=0.7,
            n=1
        )
        sub_queries = eval(response.choices[0].message.content.strip())
        return sub_queries
    except Exception as e:
        print(f"Error: {e}")
        return []
    

def answer_question(question, context):
    prompt = f'''你是一个乐于助人的人民日报新闻问答小助手。请你根据我提供的文本信息，帮我对以下开放性问题进行全面、清晰、连贯、流畅的解答。文本内容不超过512个字，并提供你认为最有帮助的文档url。
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
            model="gpt-4",
            max_tokens=512,
            temperature=0.7,
            n=1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error: {e}")
        return "抱歉，我无法回答这个问题。"


def search_complex_query(query, bm25_title, bm25_content, data, n=5):
    all_results = []
    subqueries = query.split('，')
    for subquery in subqueries:
        title_results = search_bm25(bm25_title, subquery, data, n=2)
        content_results = search_bm25(bm25_content, subquery, data, n=5)
        all_results.extend(title_results)
        all_results.extend(content_results)

    unique_results = {result['url']: result for result in all_results}
    return list(unique_results.values())


if __name__ == '__main__':
    directory = "./news_bm25"
    data = load_data_from_json(directory)
    title_corpus, content_corpus = prepare_corpus(data)

    # Uncomment the following lines to train and save the model
    # bm25_title = train_bm25(title_corpus)
    # bm25_content = train_bm25(content_corpus)
    # save_model(bm25_title, bm25_content, title_corpus, content_corpus, './bm25_title_model.pkl', './bm25_content_model.pkl', './title_corpus.pkl', './content_corpus.pkl')

    bm25_title, bm25_content, title_corpus, content_corpus = load_model(
        './bm25_title_model.pkl', './bm25_content_model.pkl', './title_corpus.pkl', './content_corpus.pkl')

    all_results = []
    while True:
        query = input("请输入查询：")
        if query == 'stop':
            break
        results = search_complex_query(
            query, bm25_title, bm25_content, data, n=2)

        combined_context = "\n\n".join(
            [f"doc{idx+1}:\nurl: {result['url']}\ntitle: {result['title']}\ncontent: {result['content']}" for idx, result in enumerate(results)])

        answer = answer_question(query, combined_context)
        print(f"Answer: {answer}")

        combined_result = {
            "query": query,
            "combined_context": combined_context,
            "answer": answer
        }
        all_results.append(combined_result)

    with open('results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
