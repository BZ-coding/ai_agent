import json
import os
import asyncio
import sys
from typing import Tuple


def tool_wrapper_for_langchain(tool):
    def tool_(args):
        try:
            kwargs = json.loads(args)
            # loop = asyncio.get_event_loop()
            # results = loop.run_until_complete(tool.run(**kwargs))
            results = tool.run(**kwargs)
        except Exception as e:
            results = str(e)
        return results

    return tool_


class DefaultTools:
    @classmethod
    def get_tool_describe_prompt(cls):
        return """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters} Format the arguments as a JSON object."""

    @classmethod
    def get_tool_web_search(cls):
        if hasattr(cls, 'web_search'):
            return cls.web_search

        from langchain_community.utilities import BingSearchAPIWrapper

        if os.getenv("BING_SUBSCRIPTION_KEY") is None:
            os.environ["BING_SUBSCRIPTION_KEY"] = ""
        if os.getenv("BING_SEARCH_URL") is None:
            os.environ["BING_SEARCH_URL"] = "https://api.bing.microsoft.com/v7.0/search"

        cls.web_search = BingSearchAPIWrapper()
        return cls.web_search

    @classmethod
    def get_tool_describe_web_search(cls):
        return {
            'name_for_human': 'bing search',
            'name_for_model': 'Search',
            'description_for_model': 'Useful for when you need to answer questions about current events.',
            'parameters': [{
                "name": "query",
                "type": "string",
                "description": "search query of web",
                'required': True
            }],
            'tool_api': tool_wrapper_for_langchain(cls.get_tool_web_search())
        }

    @classmethod
    def get_tool_math(cls):
        if hasattr(cls, 'math'):
            return cls.math

        from langchain_community.utilities import WolframAlphaAPIWrapper

        if os.getenv("WOLFRAM_ALPHA_APPID") is None:
            os.environ["WOLFRAM_ALPHA_APPID"] = ""

        cls.math = WolframAlphaAPIWrapper()
        return cls.math

    @classmethod
    def get_tool_describe_math(cls):
        return {
            'name_for_human': 'Wolfram Alpha',
            'name_for_model': 'Math',
            'description_for_model': 'Useful for when you need to answer questions about Math, Science, Technology, Culture, Society and Everyday Life.',
            'parameters': [{
                "name": "query",
                "type": "string",
                "description": "the problem to solved by Wolfram Alpha",
                'required': True
            }],
            'tool_api': tool_wrapper_for_langchain(cls.get_tool_math())
        }

    @classmethod
    def get_tool_python(cls):
        if hasattr(cls, 'python_ast'):
            return cls.python_ast

        from langchain_experimental.tools.python.tool import PythonAstREPLTool

        cls.python_ast = PythonAstREPLTool()
        return cls.python_ast

    @classmethod
    def get_tool_describe_python(cls):
        return {
            'name_for_human': 'Python',
            'name_for_model': 'Python',
            'description_for_model': "A Python shell. Use this to execute python commands. When using this tool, sometimes output is abbreviated - Make sure it does not look abbreviated before using it in your answer. Don't add comments to your python code.",
            'parameters': [{
                "name": "query",
                "type": "string",
                "description": "a valid python command.",
                'required': True
            }],
            'tool_api': tool_wrapper_for_langchain(cls.get_tool_python())
        }


class AIAgent:
    def __init__(self, chatbot, tools_info=None, tool_describe_prompt=None, react_prompt=None):
        self.chatbot = chatbot

        self.tool_describe_prompt = tool_describe_prompt
        if self.tool_describe_prompt is None:
            self.tool_describe_prompt = DefaultTools.get_tool_describe_prompt()

        self.react_prompt = react_prompt
        if self.react_prompt is None:
            self.react_prompt = """Answer the following questions as best you can. You have access to the following tools:\n\n{tool_descs}\n\nUse the following format:\n\nQuestion: the input question you must answer\nThought: you should always think about what to do\nAction: the action to take, should be one of [{tool_names}]\nAction Input: the input to the action\nObservation: the result of the action\n... (this Thought/Action/Action Input/Observation can be repeated zero or more times)\nThought: I now know the final answer\nFinal Answer: the final answer to the original input question\n\nBegin!\n\nQuestion: {query}\n"""

        self.tools_info = tools_info
        if self.tools_info is None:
            self.tools_info = [
                DefaultTools.get_tool_describe_web_search(),
                DefaultTools.get_tool_describe_math(),
                DefaultTools.get_tool_describe_python()
            ]

    def build_planning_prompt(self, query):
        tool_descs = []
        tool_names = []
        for info in self.tools_info:
            tool_descs.append(
                self.tool_describe_prompt.format(
                    name_for_model=info['name_for_model'],
                    name_for_human=info['name_for_human'],
                    description_for_model=info['description_for_model'],
                    parameters=json.dumps(info['parameters'], ensure_ascii=False),
                )
            )
            tool_names.append(info['name_for_model'])
        tool_descs = '\n\n'.join(tool_descs)
        tool_names = ','.join(tool_names)

        prompt = self.react_prompt.format(tool_descs=tool_descs, tool_names=tool_names, query=query)
        return prompt

    def parse_latest_plugin_call(self, text: str) -> Tuple[str, str]:
        action_index = text.rfind('\nAction:')
        if action_index == -1:
            action_index = text.rfind('Action:')
        action_input_index = text.rfind('\nAction Input:')
        observation_index = text.rfind('\nObservation:')
        if 0 <= action_index < action_input_index:  # If the text has `Action` and `Action input`,
            if observation_index < action_input_index:  # but does not contain `Observation`,
                # then it is likely that `Observation` is ommited by the LLM,
                # because the output text may have discarded the stop word.
                text = text.rstrip() + '\nObservation:'  # Add it back.
                observation_index = text.rfind('\nObservation:')
        if 0 <= action_index < action_input_index < observation_index:
            plugin_name = text[action_index + len('\nAction:'):action_input_index].strip()
            plugin_args = text[action_input_index + len('\nAction Input:'):observation_index].strip()
            return plugin_name, plugin_args
        return '', ''

    def use_api(self, response):
        use_toolname, action_input = self.parse_latest_plugin_call(response)
        if use_toolname == "":
            return "tool name is empty"

        used_tool_meta = list(filter(lambda x: x["name_for_model"] == use_toolname, self.tools_info))
        if len(used_tool_meta) == 0:
            return "no tool founds"

        api_output = used_tool_meta[0]["tool_api"](action_input)
        return api_output

    def ai_agent_chatbot_chat(self, messages, temperature, stop, is_print):
        response = ""
        if is_print:
            print("\033[32m", end='', flush=True)
        for token in self.chatbot.chat(messages=messages, temperature=temperature, stop=stop):
            if is_print:
                print(token, end='', flush=True)
            response += token
        if not response.endswith("\n"):
            response += "\n"
            if is_print:
                print("\n", end='', flush=True)
        if is_print:
            print("\033[0m", end='', flush=True)
        return response

    def ai_agent_chat(self, query, temperature=0.0, is_print=True, messages=None):
        if not messages:
            prompt = self.build_planning_prompt(query)  # 组织prompt
            if is_print:
                print(prompt, end='', flush=True)
            messages = [{"role": "system", "content": f"{prompt}"}]
            yield messages

        while True:
            response = self.ai_agent_chatbot_chat(messages=messages,
                                                  temperature=temperature,
                                                  stop=["Observation:", "Observation:\n"],
                                                  is_print=is_print)
            messages.append({"role": "assistant", "content": response})
            if "Final Answer:" in messages[-1]["content"]:
                break  # 出现final Answer时结束
            if is_print:
                print("\033[32m" + "Observation:\n" + "\033[0m", end='', flush=True)
            messages[-1]["content"] += "Observation:\n"

            api_output = self.use_api(messages[-1]["content"])  # 抽取入参并执行api
            api_output = str(api_output)  # 部分api工具返回结果非字符串格式需进行转化后输出
            # response = f"Observation: \"\"\"{api_output}\"\"\"\n"
            response = f"\"\"\"{api_output}\"\"\"\n"
            if is_print:
                print("\033[34m" + response + "\033[0m", end='', flush=True)
            messages.append({"role": "assistant", "content": response})
            yield messages
        yield messages


if __name__ == '__main__':
    print(os.path.dirname(os.path.dirname(__file__)))
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from utils.chatbot import ChatBot

    chatbot = ChatBot()
    ai_agent = AIAgent(chatbot=chatbot)
    is_print = True

    messages = ai_agent.ai_agent_chat(query="金陵中学所在的行政区有多少人?", temperature=0.0, is_print=is_print)

    print("\n\n\n\n")
    for current_messages in messages:
        if not is_print:
            print(current_messages[-1]["content"])
        pass
