def get_conversation_token_fn(tokenizer):
    def _get_conversation_token(data_point):
        conversation = data_point["messages"]
        tokens = {'input_ids': [], 'attention_mask': [], 'labels': []}
        is_observation_before = False
        for i, chat in enumerate(conversation):
            chat_token = tokenizer.apply_chat_template(conversation=[chat], tokenize=True, add_generation_prompt=False,
                                                       return_tensors=None, return_dict=True)
            if i != 0:  # remove <|begin_of_text|>
                chat_token['input_ids'] = chat_token['input_ids'][1:]
                chat_token['attention_mask'] = chat_token['attention_mask'][1:]
            tokens['input_ids'] += chat_token['input_ids']
            tokens['attention_mask'] += chat_token['attention_mask']
            if chat["role"] == "assistant" and not is_observation_before:
                generation_prompt = '<|start_header_id|>' + chat['role'] + '<|end_header_id|>\n\n'
                generation_prompt_token = tokenizer.tokenize(generation_prompt)
                tokens['labels'] += [-100] * len(generation_prompt_token) + chat_token['input_ids'].copy()[
                                                                            len(generation_prompt_token):]
            else:
                tokens['labels'] += [-100] * len(chat_token['input_ids'])

            is_observation_before = chat["content"].endswith("Observation:\n") or chat["content"].endswith(
                "Observation:")
        return tokens

    return _get_conversation_token


def handle_multi_conversation_dataset(dataset, tokenizer):
    get_conversation_token = get_conversation_token_fn(tokenizer)
    for data_name in dataset:
        dataset[data_name] = dataset[data_name].map(get_conversation_token,
                                                    remove_columns=dataset[data_name].column_names, num_proc=2)
    return dataset
