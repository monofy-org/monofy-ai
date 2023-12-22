def convert_gr_to_openai(chat_list):
    formatted_chat = []

    for user_text, bot_text in chat_list:
        # Append user message
        formatted_chat.append({"role": "user", "content": user_text})
        
        # Append assistant message
        formatted_chat.append({"role": "assistant", "content": bot_text})

    return formatted_chat