#!/usr/bin/env python
# coding: utf-8

# https://github.com/QwenLM/Qwen/blob/main/examples/langchain_tooluse.ipynb

# In[1]:


import os
import sys

import transformers

sys.path.append(os.path.abspath('.'))
import json
from typing import Dict, Tuple
import asyncio

from langchain_community.utilities import BingSearchAPIWrapper
from langchain_community.utilities import WolframAlphaAPIWrapper
from langchain_experimental.tools.python.tool import PythonAstREPLTool

from utils.chatbot import ChatBot

# In[2]:


os.environ["BING_SUBSCRIPTION_KEY"] = ""
os.environ["BING_SEARCH_URL"] = "https://api.bing.microsoft.com/v7.0/search"
os.environ["WOLFRAM_ALPHA_APPID"] = ""

# In[3]:


search = BingSearchAPIWrapper()
search

# In[4]:


search.run("hello")

# langchain的ai_agent的prompt：
# 
# 'Answer the following questions as best you can. You have access to the following tools:\n\n
# 
# Calculator(
# 	*args: Any, 
# 	callbacks: Union[List[langchain_core.callbacks.base.BaseCallbackHandler], langchain_core.callbacks.base.BaseCallbackManager, NoneType] = None, 
# 	tags: Optional[List[str]] = None, 
# 	metadata: Optional[Dict[str, Any]] = None, 
# 	**kwargs: Any) -> Any 
# - Useful for when you need to answer questions about math.\n
# 
# Search(query: str) -> str 
# - A wrapper around Search. Useful for when you need to answer questions about current events. Input should be a search query.\n\n
# 
# Use the following format:\n\n
# 
# Question: the input question you must answer\n
# Thought: you should always think about what to do\n
# Action: the action to take, should be one of [Calculator, Search]\n
# Action Input: the input to the action\nObservation: the result of the action\n
# ... (this Thought/Action/Action Input/Observation can repeat N times)\n
# Thought: I now know the final answer\n
# Final Answer: the final answer to the original input question\n\n
# 
# Begin!\n\n
# 
# Question: {input}\n
# Thought:{agent_scratchpad}'

# In[5]:


search = BingSearchAPIWrapper()
wolfram = WolframAlphaAPIWrapper()
python_ast = PythonAstREPLTool()

# In[6]:


print(wolfram.run("9894556*897898"))

# In[7]:


print(python_ast.run("print('hello')"))


# In[8]:


def tool_wrapper_for_langchain(tool):
    def tool_(query):
        try:
            query = json.loads(query)["query"]
            #         loop = asyncio.get_event_loop()
            #         results = loop.run_until_complete(tool.run(query))
            results = tool.run(query)
        except Exception as e:
            results = str(e)
        return results

    return tool_


# 以下是给模型看的工具描述：
tools_info = [
    {
        'name_for_human':
            'bing search',
        'name_for_model':
            'Search',
        'description_for_model':
            'Useful for when you need to answer questions about current events.',
        'parameters': [{
            "name": "query",
            "type": "string",
            "description": "search query of web",
            'required': True
        }],
        'tool_api': tool_wrapper_for_langchain(search)
    },
    {
        'name_for_human':
            'Wolfram Alpha',
        'name_for_model':
            'Math',
        'description_for_model':
            'Useful for when you need to answer questions about Math, Science, Technology, Culture, Society and Everyday Life.',
        'parameters': [{
            "name": "query",
            "type": "string",
            "description": "the problem to solved by Wolfram Alpha",
            'required': True
        }],
        'tool_api': tool_wrapper_for_langchain(wolfram)
    },
    {
        'name_for_human':
            'python',
        'name_for_model':
            'python',
        'description_for_model':
            "A Python shell. Use this to execute python commands. When using this tool, sometimes output is abbreviated - Make sure it does not look abbreviated before using it in your answer. "
            "Don't add comments to your python code.",
        'parameters': [{
            "name": "query",
            "type": "string",
            "description": "a valid python command.",
            'required': True
        }],
        'tool_api': tool_wrapper_for_langchain(python_ast)
    }

]

# In[9]:


TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters} Format the arguments as a JSON object."""

REACT_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

{tool_descs}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {query}\n"""


def build_planning_prompt(tools_info, query):
    tool_descs = []
    tool_names = []
    for info in tools_info:
        tool_descs.append(
            TOOL_DESC.format(
                name_for_model=info['name_for_model'],
                name_for_human=info['name_for_human'],
                description_for_model=info['description_for_model'],
                parameters=json.dumps(
                    info['parameters'], ensure_ascii=False),
            )
        )
        tool_names.append(info['name_for_model'])
    tool_descs = '\n\n'.join(tool_descs)
    tool_names = ','.join(tool_names)

    prompt = REACT_PROMPT.format(tool_descs=tool_descs, tool_names=tool_names, query=query)
    return prompt


# In[10]:


# prompt_1 = build_planning_prompt(tools_info[0:2], query="加拿大2023年人口统计数字是多少？")
# print(prompt_1)
# print("=====")

# In[11]:


chatbot = ChatBot()

# In[12]:


# message = [{"role": "system", "content": f"{prompt_1}"}, {"role": "assistant", "content": "Thought: "}]
# ans = ""
# for token in chatbot.chat(messages=message, temperature=0.0, stop=["Observation:", "Observation:\n"]):
#     # print(token, end='', flush=True)
#     ans += token


# print(ans)


# In[13]:


def parse_latest_plugin_call(text: str) -> Tuple[str, str]:
    i = text.rfind('\nAction:')
    if i == -1:
        i = text.rfind('Action:')
    j = text.rfind('\nAction Input:')
    k = text.rfind('\nObservation:')
    # print(i, j, k)
    if 0 <= i < j:  # If the text has `Action` and `Action input`,
        if k < j:  # but does not contain `Observation`,
            # then it is likely that `Observation` is ommited by the LLM,
            # because the output text may have discarded the stop word.
            text = text.rstrip() + '\nObservation:'  # Add it back.
            k = text.rfind('\nObservation:')
    if 0 <= i < j < k:
        plugin_name = text[i + len('\nAction:'):j].strip()
        plugin_args = text[j + len('\nAction Input:'):k].strip()
        return plugin_name, plugin_args
    return '', ''


def use_api(tools, response):
    use_toolname, action_input = parse_latest_plugin_call(response)
    if use_toolname == "":
        return "tool name is empty"

    used_tool_meta = list(filter(lambda x: x["name_for_model"] == use_toolname, tools))
    if len(used_tool_meta) == 0:
        return "no tool founds"

    api_output = used_tool_meta[0]["tool_api"](action_input)
    return api_output


# In[14]:


# api_output = use_api(tools_info[0:2], ans)
# print(api_output)


# In[ ]:

# message[-1]["content"] += f"{ans}\n"
# message.append({"role": "assistant", "content": f"Observation: \"\"\"{api_output}\"\"\"\n"})
# # print(message)
# tokenizer = transformers.AutoTokenizer.from_pretrained(
#     "/mnt/nfs/zsd_server/models/huggingface/llama-3-chinese-8b-instruct-v3/")
# template = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
# print("\n\n\n\n")
# print(template)

# In[ ]:


# ans = ""
# for token in chatbot.chat(messages=message, temperature=0.0, stop=["Observation:", "Observation:\n"]):
#     # print(token, end='', flush=True)
#     ans += token

# In[ ]:


# ans

# In[ ]:


# api_output = use_api(tools_info[0:2], ans)
# print(api_output)


# In[ ]:

def ai_agent_chatbot_chat(chatbot, messages):
    response = ""
    print("\033[32m", end='', flush=True)
    for token in chatbot.chat(messages=messages, temperature=0.0, stop=["Observation:", "Observation:\n"]):
        print(token, end='', flush=True)
        response += token
    if not response.endswith("\n"):
        response += "\n"
        print("\n", end='', flush=True)
    print("\033[0m", end='', flush=True)
    return response


def ai_agent_chat(choose_tools, query):
    prompt = build_planning_prompt(choose_tools, query)  # 组织prompt
    messages = [{"role": "system", "content": f"{prompt}"}, {"role": "assistant", "content": "Thought: "}]
    # template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    # print(template, flush=True)
    print(prompt, flush=True)
    print("Thought: ", flush=True)
    response = ai_agent_chatbot_chat(chatbot, messages)
    messages[-1]["content"] += f"{response}"

    while "Final Answer:" not in response:  # 出现final Answer时结束
        api_output = use_api(choose_tools, response)  # 抽取入参并执行api
        api_output = str(api_output)  # 部分api工具返回结果非字符串格式需进行转化后输出
        # if api_output in ["tool name is empty", "no tool founds"]:
        #     break
        response = f"Observation: \"\"\"{api_output}\"\"\"\n"
        print("\033[34m" + response + "\033[0m", flush=True)
        messages.append({"role": "assistant", "content": f"Observation: {response}"})

        messages.append({"role": "assistant", "content": "Thought: "})
        print("Thought: ", flush=True)
        response = ai_agent_chatbot_chat(chatbot, messages)  # 继续生成
        messages[-1]["content"] += f"{response}"


print("\n\n\n\n", flush=True)
ai_agent_chat(choose_tools=tools_info[0:1], query="金陵中学所在的行政区有多少人?")
