import os

import gradio as gr
import torch
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration

# 设置代理（如果需要）
# os.environ["http_proxy"] = "http://127.0.0.1:7890"
# os.environ["https_proxy"] = "http://127.0.0.1:7890"

# 加载模型
# 加载分词器和模型
# pretrained_model = "IDEA-CCNL/Randeng-T5-77M-MultiTask-Chinese"
pretrained_model = r"D:\DeepLearningModels\2024_07_24\IDEA-CCNLRandeng-T5-77M-MultiTask-Chinese"

special_tokens = ["<extra_id_{}>".format(i) for i in range(100)]
tokenizer = T5Tokenizer.from_pretrained(
    pretrained_model,
    do_lower_case=True,
    max_length=512,
    truncation=True,
    additional_special_tokens=special_tokens,
)

config = T5Config.from_pretrained(pretrained_model)
model = T5ForConditionalGeneration.from_pretrained(pretrained_model, config=config)
model.resize_token_embeddings(len(tokenizer))
model.eval()

# 构造prompt的过程中，verbalizer这个占位key的内容，是通过 "/".join(choices) 拼接起来
dataset2instruction = {
    "情感分析": {
        "prompt": "{}任务：【{}】这篇文章的情感态度是什么？{}",
        "keys_order": ["subtask_type", "text_a", "verbalizer"],
        "data_type": "classification",
    },
    "文本分类": {
        "prompt": "{}任务：【{}】这篇文章的类别是什么？{}",
        "keys_order": ["subtask_type", "text_a", "verbalizer"],
        "data_type": "classification",
    },
    "新闻分类": {
        "prompt": "{}任务：【{}】这篇文章的类别是什么？{}",
        "keys_order": ["subtask_type", "text_a", "verbalizer"],
        "data_type": "classification",
    },
    "意图识别": {
        "prompt": "{}任务：【{}】这句话的意图是什么？{}",
        "keys_order": ["subtask_type", "text_a", "verbalizer"],
        "data_type": "classification",
    },
    # --------------------
    "自然语言推理": {
        "prompt": "{}任务：【{}】和【{}】，以上两句话的逻辑关系是什么？{}",
        "keys_order": ["subtask_type", "text_a", "text_b", "verbalizer"],
        "data_type": "classification",
    },
    "语义匹配": {
        "prompt": "{}任务：【{}】和【{}】，以上两句话的内容是否相似？{}",
        "keys_order": ["subtask_type", "text_a", "text_b", "verbalizer"],
        "data_type": "classification",
    },
    # -----------------------
    "指代消解": {
        "prompt": "{}任务：文章【{}】中{}{}",
        "keys_order": ["subtask_type", "text_a", "question", "verbalizer"],
        "data_type": "classification",
    },
    "多项选择": {
        "prompt": "{}任务：阅读文章【{}】问题【{}】？{}",
        "keys_order": ["subtask_type", "text_a", "question", "verbalizer"],
        "data_type": "classification",
    },
    # ------------------------
    "抽取式阅读理解": {
        "prompt": "{}任务：阅读文章【{}】问题【{}】的答案是什么？",
        "keys_order": ["subtask_type", "text_a", "question"],
        "data_type": "mrc",
    },
    "实体识别": {
        "prompt": "{}任务：找出【{}】这篇文章中所有【{}】类型的实体？",
        "keys_order": ["subtask_type", "text_a", "question"],
        "data_type": "ner",
    },
    # ------------------------
    "关键词抽取": {
        "prompt": "{}任务：【{}】这篇文章的关键词是什么？",
        "keys_order": ["subtask_type", "text_a"],
        "data_type": "keys",
    },
    "关键词识别": {
        "prompt": "{}任务：阅读文章【{}】问题【{}】{}",
        "keys_order": ["subtask_type", "text_a", "question", "verbalizer"],
        "data_type": "classification",
    },
    "生成式摘要": {
        "prompt": "{}任务：【{}】这篇文章的摘要是什么？",
        "keys_order": ["subtask_type", "text_a"],
        "data_type": "summ",
    },
}


def get_instruction(subtask_type: str, text_a: str, verbalizer: str) -> str:
    sample = {
        "subtask_type": subtask_type,
        "text_a": text_a,
        "verbalizer": verbalizer,
    }
    template = dataset2instruction[sample["subtask_type"]]
    sample["instruction"] = template["prompt"].format(*[
        sample[k] for k in template["keys_order"]
    ])
    print(sample["instruction"])
    return sample["instruction"]


def function(subtask_type, text_a, verbalizer):
    if not text_a:
        gr.Warning("请输入文本")
        return "# 请输入文本"
    verbalizer = "/".join(verbalizer.split())
    text = get_instruction(subtask_type, text_a, verbalizer)
    encode_dict = tokenizer(text, max_length=512, padding='max_length', truncation=True)

    inputs = {
        "input_ids": torch.tensor([encode_dict['input_ids']]).long(),
        "attention_mask": torch.tensor([encode_dict['attention_mask']]).long(),
    }
    # 生成答案
    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],  # 确保也提供了注意力掩码
        max_length=100,
        early_stopping=True,
    )
    # 解码输出
    predicted_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )
    print(predicted_text)
    return "# " + predicted_text


def function_sentiment(text_a, verbalizer):
    return function("情感分析", text_a, verbalizer)


def function_news(text_a, verbalizer):
    return function("新闻分类", text_a, verbalizer)


def function_getKeyWords(text_a):
    return function("关键词抽取", text_a, "")


def function_summary(text_a):
    return function("生成式摘要", text_a, "")


# 通过循环加载配置文件的方式，动态生成应用
application_config = [
    {
        "name": "情感分析",
        "function": function_sentiment,
        "inputs": [
            gr.TextArea(label="请输入情感分析的文本"),
            gr.Textbox(label="分类标签 使用空格分割", value="正面   负面"),
        ],
        "outputs": gr.Markdown(label="输出结果"),
        "description": "多场景文本分类",
        "examples": [
            ["这个酒店的服务很不错", "正面   负面"],
            ["这个酒店的服务很差，房间很脏", "正面   负面"],
        ]
    },
    {
        "name": "新闻分类",
        "function": function_news,
        "inputs": [
            gr.TextArea(label="请输入新闻分类的文本"),
            gr.Textbox(label="分类标签 使用空格分割", value="故事   文化  娱乐  体育  财经  房产  汽车  教育  科技"),
        ],
        "outputs": gr.Markdown(label="输出结果"),
        "description": "多场景文本分类",
        "examples": [
            ["中国女排3-0完胜美国女排，取得奥运会开门红", "故事   文化  娱乐  体育  财经  房产  汽车  教育  科技"],
            ["无人驾驶出租车在中国首次上路", "故事   文化  娱乐  体育  财经  房产  汽车  教育  科技"],
        ]
    },
    {
        "name": "关键词抽取",
        "function": function_getKeyWords,
        "inputs": [
            gr.TextArea(label="请输入关键词抽取的文本"),
        ],
        "outputs": gr.Markdown(label="输出结果"),
        "description": "多场景文本分类",
        "examples": [
            ["今儿在大众点评，找到了口碑不错的老茶故事私房菜。"],
            ["今天天气不错，适合出去玩。"],
        ]
    },
    {
        "name": "生成式摘要",
        "function": function_summary,
        "inputs": [
            gr.TextArea(label="请输入生成式摘要的文本"),
        ],
        "outputs": gr.Markdown(label="输出结果"),
        "description": "多场景文本分类",
        "examples": [
            "量化机器人对市场趋势的反应速度在当前金融市场中具有重要意义。随着算法和大数据技术的发展，量化机器人通过先进的计算能力和实时数据分析，能够迅速捕捉市场变化，做出及时的交易决策。这种快速反应能力，不仅提高了交易效率，也显著增强了投资者的收益和风险管理。",
            "请注意，function 函数在这里没有被定义，所以你需要确保它已经在你的代码中定义好了。此外，gr.Interface 不支持直接使用 gr.Blocks 中的布局元素，比如 gr.Row 和 gr.Column，所以这里只使用了基础的 gr.Interface 来定义输入输出。",
        ]
    },
    {
        "name": "关于",
        "function": None,
        "inputs": [
            gr.Markdown(
                """
                ## 模型来源
                [IDEA-CCNL/Randeng-T5-77M-MultiTask-Chinese](https://huggingface.co/IDEA-CCNL/Randeng-T5-77M-MultiTask-Chinese)                
                """
            )
        ],
        "outputs": gr.Markdown(label="关于"),
        "description": "多场景文本分类",
        "examples": [],
    }
]

application_ifaces = []
application_names = []
# 通过循环加载配置文件的方式，动态生成应用
for config in application_config:
    application_ifaces.append(
        gr.Interface(
            fn=config["function"],
            inputs=config["inputs"],
            outputs=config["outputs"],
            title=config["name"],
            description=config["description"],
            examples=config["examples"]
        )
    )
    application_names.append(config["name"])

# 使用 gr.TabbedInterface 将两个子应用组合起来
application = gr.TabbedInterface(
    interface_list=application_ifaces,
    tab_names=application_names,
    title="多场景文本分类",
)

application.launch()
