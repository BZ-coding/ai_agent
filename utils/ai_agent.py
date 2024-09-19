import json
import os
import asyncio
import sys
from typing import Tuple

print(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.rewriter import ReWriter
from utils.fact_extraction import FactExtractor

REACT_PROMPT = \
"""Answer the following questions as best you can. You have access to the following tools:

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

Question: {query}
"""


def tool_wrapper_for_langchain(tool):
    def tool_(args):
        try:
            kwargs = json.loads(args, strict=False)
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
            # 'description_for_model': 'Useful for when you need to answer questions about Math, Science, Technology, Culture, Society and Everyday Life.',
            'description_for_model': 'Useful for when you need to answer questions about Math.',
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
                "name": "tool_input",
                "type": "string",
                "description": "a valid python command.",
                'required': True
            }],
            'tool_api': tool_wrapper_for_langchain(cls.get_tool_python())
        }


class AIAgent:
    def __init__(self, chatbot, tools_info=None, tool_describe_prompt=None, react_prompt=None):
        self.chatbot = chatbot
        self.rewriter = ReWriter(chatbot=chatbot)
        self.fact_extractor = FactExtractor(chatbot=chatbot)

        self.tool_describe_prompt = tool_describe_prompt
        if self.tool_describe_prompt is None:
            self.tool_describe_prompt = DefaultTools.get_tool_describe_prompt()

        self.react_prompt = react_prompt
        if self.react_prompt is None:
            self.react_prompt = REACT_PROMPT

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
        action_input_index = text.rfind('\nAction Input:')
        action_index = text.rfind('\nAction:', 0, None if action_input_index == -1 else action_input_index)
        if action_index == -1:
            action_index = text.rfind('Action:', 0, None if action_input_index == -1 else action_input_index)
        observation_index = text.rfind('\nObservation:',
                                       None if action_input_index == -1 else action_input_index)

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
            return "tool name is empty", (None, None)

        used_tool_meta = list(filter(lambda x: x["name_for_model"] == use_toolname, self.tools_info))
        if len(used_tool_meta) == 0:
            return "no tool founds", (None, None)

        api_output = used_tool_meta[0]["tool_api"](action_input)
        return api_output, (use_toolname, action_input)

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

    def rewrite_func(self, query, is_print=True):
        if not self.rewriter:
            return query
        new_query = self.rewriter.rewrite(query)
        if is_print:
            print(f"\033[32mquery: {query} --> {new_query}\033[0m")
        return new_query

    def fact_extract_func(self, action_input, api_output, is_print=True):
        if not self.fact_extractor or len(api_output) < 100:
            return api_output
        if is_print:
            print("\033[34m" + "old api_output : " + api_output + "\033[0m")
        api_output = self.fact_extractor.extract(query=action_input, context=api_output)
        return api_output

    def ai_agent_chat(self, query, temperature=0.0, is_print=True, messages=None, skip_chatbot=False, rewrite=False, fact_extract=True):
        if not messages:
            if rewrite:
                query = self.rewrite_func(query=query, is_print=False)
            prompt = self.build_planning_prompt(query)  # 组织prompt
            if is_print:
                print(prompt, end='', flush=True)
            messages = [{"role": "system", "content": f"{prompt}"}]
            yield messages

        while True:
            if not skip_chatbot:
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

            api_output, (use_toolname, action_input) = self.use_api(messages[-1]["content"])  # 抽取入参并执行api
            api_output = str(api_output)  # 部分api工具返回结果非字符串格式需进行转化后输出
            api_output = self.fact_extract_func(action_input=f"{action_input} {query}", api_output=api_output, is_print=False)
            response = f"\"\"\"{api_output}\"\"\"\n"
            if is_print:
                print("\033[34m" + response + "\033[0m", end='', flush=True)
            messages.append({"role": "assistant", "content": response})
            yield messages
        yield messages


if __name__ == '__main__':
    python_ast = DefaultTools.get_tool_describe_python()
    print(python_ast['tool_api']("{\"tool_input\": \"print('Hello World!')\"}"))

    from utils.chatbot import ChatBot

    chatbot = ChatBot(model_name="zhangsheng377/llama-3-chinese-8b-tools-f16-lora")
    ai_agent = AIAgent(chatbot=chatbot)
    is_print = True

    messages = ai_agent.ai_agent_chat(query="金陵中学赈灾义演的主持人是章泽天吗？", temperature=0.0, is_print=is_print)

    print("\n\n\n\n")
    for current_messages in messages:
        if not is_print:
            print(current_messages[-1]["content"])
        pass
