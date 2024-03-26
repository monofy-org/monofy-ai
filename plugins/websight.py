from modules.plugins import PluginBase, release_plugin, use_plugin


class WebSightPlugin(PluginBase):

    name = "WebSight"
    description = "WebSight"
    instance = None

    def __init__(self):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        super().__init__()
        # Load tokenizer and model
        self.resources["tokenizer"] = AutoTokenizer.from_pretrained(
            "your_model_directory"
        )  # Load tokenizer from saved model
        self.resources["model"] = AutoModelForSequenceClassification.from_pretrained(
            "your_model_directory"
        )  # Load model from saved model

    def predict(self, text):
        import torch

        tokenizer = self.resources["tokenizer"]
        model = self.resources["model"]

        # Tokenize input text
        inputs = tokenizer(text, return_tensors="pt")

        # Perform inference
        outputs = model(**inputs)
        predictions = torch.softmax(outputs.logits, dim=1).tolist()[0]

        return predictions


@PluginBase.router.post("/websight")
async def websight(text: str):
    plugin = None
    try:
        plugin: WebSightPlugin = await use_plugin(WebSightPlugin)
        return plugin.predict(text)
    finally:
        if plugin is not None:
            release_plugin(plugin)
