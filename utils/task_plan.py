import os
import sys
from time import sleep

print(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.chatbot import ChatBot

DEFAULT_SYSTEM_PROMPT_OLD = """你是一个非常有效的任务规划员，你要先对用户的问题进行分析，然后使用分而治之的策略对用户问题进行分解，最后完成任务规划。请注意，你所规划的每个步骤都独占一行，且带有数字序号。
你所规划的每个子任务，连同之前步骤的答案，都将传递给ai agent工具链执行，最终模型会整合所有步骤的答案给出最终结果。
请你将回答保持与原query一样的语言。"""
DEFAULT_SYSTEM_PROMPT = """你是一个非常有效的任务规划员，你要先分析用户的问题然后使用分而治之的策略进行详细讨论与规划。
请注意，每个步骤独占一行并带有数字序号，以方便跟踪和执行子任务；
每个子任务，以及之前答复所产生的新信息都将经过模型传递给ai agent工具链来运行；
最终，模型整合所有得到的结果生成完整且准确的回答。"""
DEFAULT_USER_PROMPT = """请你对“{query}”这个用户问题进行任务分解规划。"""


class TaskPlan:
    def __init__(self, chatbot, system_prompt=DEFAULT_SYSTEM_PROMPT, user_prompt=DEFAULT_USER_PROMPT):
        self.chatbot = chatbot
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

    def rewrite(self, query, stream=False):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt.format(query=query)},
        ]
        response = self.chatbot.chat(messages=messages, stream=stream)
        return response


if __name__ == "__main__":
    chatbot = ChatBot()
    rewriter = TaskPlan(chatbot=chatbot)

    query = f"请你对\"\"\"{DEFAULT_SYSTEM_PROMPT_OLD}\"\"\"这个prompt进行优化。"
    query = "金陵中学所在的行政区有多少人?"
    for token in rewriter.rewrite(query=query, stream=True):
        print(token, end='', flush=True)
    print('\n')
