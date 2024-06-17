from fastapi.responses import JSONResponse
from plugins.exllamav2 import ChatCompletionRequest
from modules.plugins import PluginBase, release_plugin, use_plugin


class CausalLMPlugin(PluginBase):
    name = "CausalLM"
    description = "Causal language model"
    instance = None

    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        super().__init__()

        model_name = "stabilityai/stablelm-2-1_6b"

        self.resources["model"] = AutoModelForCausalLM.from_pretrained(
            model_name,
        ).to(self.device, dtype=self.dtype)
        self.resources["tokenizer"] = AutoTokenizer.from_pretrained(model_name)
        self.resources["tokenizer"].chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}"


    def generate(self, req: ChatCompletionRequest) -> str:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model: AutoModelForCausalLM = self.resources["model"]
        tokenizer: AutoTokenizer = self.resources["tokenizer"]

        system_prefix = {
            "role": "system",
            "content": "You are a sassy chatbot responding to messages being relayed to you in chat. Read the following log and then give a single response.",
        }

        system_suffix = {
            "role": "system",
            "content": "End of log. Give your single response now.",
        }

        print(req.messages)

        inputs = tokenizer.apply_chat_template(
            [system_prefix] + req.messages + [system_suffix],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )

        print(tokenizer.decode(inputs[0]))

        outputs = model.generate(
            inputs.to(model.device),
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=0.95,
            do_sample=True,
        )

        return tokenizer.decode(outputs[0], skip_special_tokens=True)


@PluginBase.router.post("/causal_lm", tags=["Text Generation"])
async def causal_lm(
    req: ChatCompletionRequest,
):
    plugin: CausalLMPlugin = None
    try:
        plugin = await use_plugin(CausalLMPlugin)
        result = plugin.generate(req)
        return JSONResponse(content={"choices": [{"message": {"content": result}}]})
    finally:
        if plugin:
            release_plugin(plugin)
