# ai_agent

----------------------------------------

## TODO

- [ ] 自己用lora微调langchain agent格式的模型
  > 1. 普通的指令模型（llama3）面对使用工具的场景，表现一般（输出时不遵从工具名）。更为关键的是，langchain的代码太垃圾了（需要格式，指令关键字、标点完全匹配才能成功进入下一步，但llama3用中文时会输出中文标点）
  > 2. 参考toolformer的资料？
  > 3. Llama3使用 bfloat16 进行训练，但原始推理使用float16。
  > 4. tokenizer 自己就有模板能力，可以试试。 tokenizer_config.json  https://huggingface.co/docs/transformers/main/en/chat_templating#templates-for-chat-models
- [x] 多收集、想出一些query，100、200条就行。试试让大模型自己生成需要使用aiagent的问题？
- [ ] 找一个文本标注工具，有页面。显示query和模型的推理结果，人修改推理结果就行，prompt展示但不给修改。
