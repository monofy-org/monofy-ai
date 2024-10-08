import logging
import gradio as gr

blocks: gr.Blocks = gr.Blocks(title="monofy-ai", analytics_enabled=False).queue()

with blocks:
    tabs = gr.Tabs()

def webui():

    def decorator(func):
        def wrapper(*args, **kwargs):
            with blocks:
                with tabs:       
                    func(*args, **kwargs)

        return wrapper

    return decorator

async def launch():
    logging.info("Launching Gradio...")
    blocks.launch(prevent_thread_lock=True, share=False)
    
