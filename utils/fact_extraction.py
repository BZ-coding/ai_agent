import os
import sys
from time import sleep

print(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.chatbot import ChatBot

DEFAULT_SYSTEM_PROMPT = """你是一个分析师。请从本文中提取与'{query}'相关的关键事实，不相关的可以舍弃。不要包括意见。给每个事实一个数字，并保持简短的句子。"""
DEFAULT_USER_PROMPT = """Context: {context}"""


class FactExtractor:
    def __init__(self, chatbot, system_prompt=DEFAULT_SYSTEM_PROMPT, user_prompt=DEFAULT_USER_PROMPT):
        self.chatbot = chatbot
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

    def extract(self, query, context, stream=False):
        messages = [
            {"role": "system", "content": self.system_prompt.format(query=query)},
            {"role": "user", "content": self.user_prompt.format(context=context)},
        ]
        response = self.chatbot.chat(messages=messages, stream=stream)
        return response


if __name__ == "__main__":
    chatbot = ChatBot()
    fact_extractor = FactExtractor(chatbot=chatbot)

    query = "金陵中学所在行政区的人口数量"
    content = """南京市<b>金陵中学</b>（Nanjing Jinling High School），简称“金中”，位于江苏省 南京市 鼓楼<b>区</b>，是南京市教育局直属公办高中，江苏省首批四星级高中、江苏省重点<b>中学</b>、国家级示范高中、江苏省首批创新拔尖人才培养试点学校、江苏省首批高品质示范高中，全国科技 ... 从<b>人口</b>质量来看，南京全市15岁及以上<b>人口的</b>平均受教育年限从2010年<b>的</b>10.99年提高至11.76年；每10万人中拥有大学文化程度<b>的</b>由2010年<b>的</b>2.61万人上升为3.52万人；文盲率从2010年<b>的</b>2.63%下降为2.13%。 <b>金陵中学</b>简介. 南京市<b>金陵中学</b>创建于1888年。. 现为首批国家级示范高中、江苏省首批四星级高中、江苏省首批普通高中创新拔尖人才培养试点学校、江苏省模范学校、江苏省德育先进学校、江苏省高品质示范高中首批建设立项学校；全国科技创新教育十佳学校 ... <b>金陵中学</b> (Jinling High School) (南京大学<b>金陵中学</b>)，简称“金中”，位于江苏省南京市鼓楼<b>区</b>，是南京市教育局直属公办高中，江苏省首批四星级高中、江苏省重点<b>中学</b>、国家级示范高中、江苏省首批创新拔尖人才培养试点学校、江苏省首批高品质示范高中 ... 南京市，简称“宁”，古称“ <b>金陵</b> ”“ 建康 ” [1]， 江苏省 辖 地级市 、 省会，是 副省级市 、 特大城市 [289]、 南京都市圈 核心城市 [2]，国务院批复确定<b>的</b> 中国东部 地区重要<b>的</b>中心城市，全国重要<b>的</b>科研教育基地和综合交通枢纽 [3]。. 截至2023年8月，南京 ... 开学第一课：成为新质校园生活<b>的</b>主动构建者 新质教育铸英才，馥郁书香润人生！. 9月2日上午，南京市<b>金陵中学</b>2024—2025学年第一学期开学典礼暨第十二届读书节开幕式分别在两个校区举行。. 身着正装踏上红毯，高一“萌新”们怀揣着憧憬与好奇，正式开启 ... 2017年中考人数：4300人左右. 建邺区. 高中：3所. 市直属1所，区四星2所. 初中：10所. 民办 ：金陵河西，河西外校. 市重点高中：中华中学. 区内知名初中：南师新城，新城黄山路，中华上新河. 金中仙林分校位于仙林大学城，由栖霞区人民政府、仙林大学城管委会、南京市教育局与南京大学四方合作创办，为区属公办学校，分为小学部和初中部，办学规模计划为小学8轨、初中16轨。 小学部2012年9月招生，原游府西街小学校长、南京知名教育专家林慧敏担任校长，并在全省遴选了优秀的管理和教学人才组成了经验丰富的教学团队。 初中部今年9月1日开办，金陵中学派出副校长夏广平担任分校校长，江苏省特级教师王苏豫、南京市物理学科带头人陈立其等6名本部名师入驻，占所有师资的1/3。 E 金陵中学实验小学. 最大亮点：河西中部优质公办小学，划分 学区. 关注人群：幼升小. 收费标准：公办体制，享受义务教育免学费政策. 上海市<b>金陵中学</b> 为 黄浦<b>区</b> <b>的</b>一所 完全<b>中学</b>，于1956年由上海市北郊<b>中学</b>分校（原 沪江大学 附中分校），与上海市建工初级<b>中学</b>合并而成，校址为原法国驻沪领事馆，曾是上海市楼层最多，最高<b>的</b>校舍，排名也曾一度位于黄浦<b>区</b>前三，仅次于大同，格致。 上海市 <b>金陵中学</b>（Shanghai Jinling Middle School）是一所位于中华人民共和国上海市 黄浦<b>区</b> 外滩附近<b>金陵</b>东路<b>的</b>公办 完全<b>中学</b>，学校原址是法国驻上海总领事馆。. [1]1956年， ..."""

    for token in fact_extractor.extract(query=query, context=content, stream=True):
        print(token, end='', flush=True)
    print('\n')
