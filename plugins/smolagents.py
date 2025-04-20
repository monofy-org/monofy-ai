from typing import Optional

from fastapi import Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# from submodules.smolagents.src.smolagents.models import InferenceClientModel
from smolagents import CodeAgent, DuckDuckGoSearchTool, TransformersModel

from modules.plugins import PluginBase, use_plugin_unsafe


class SmolAgentsRequest(BaseModel):
    prompt: str
    model_index: Optional[int] = 0


class SmolAgentsPlugin(PluginBase):
    def __init__(self):
        super().__init__()

    def load_model(self) -> CodeAgent:
        model = self.resources.get("model")
        agent = self.resources.get("agent")

        if model and agent:
            return agent

        # model = InferenceClientModel()
        model = TransformersModel(
            model_id="Qwen/Qwen2.5-Coder-7B-Instruct-GPTQ-Int8",
            max_new_tokens=4096,
            device_map="auto",
        )

        # model.tokenizer.chat_template="{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}",

        agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=model)

        self.resources["model"] = model
        self.resources["agent"] = agent

        return agent

    def generate(self, req: SmolAgentsRequest):
        agent: CodeAgent = self.load_model()

        response = agent.run(req.prompt)

        return response


@PluginBase.router.post("/smol")
async def smolagents(req: SmolAgentsRequest):
    plugin: SmolAgentsPlugin = use_plugin_unsafe(SmolAgentsPlugin)
    response = plugin.generate(req)
    return JSONResponse({"response": str(response)})


@PluginBase.router.get("/smol")
async def smolagents_get(req: SmolAgentsPlugin = Depends()):
    return await smolagents(req)
