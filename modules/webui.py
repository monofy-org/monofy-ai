import gradio as gr

blocks: gr.Blocks = gr.Blocks(title="monofy-ai", analytics_enabled=False)

interfaces = [gr.Interface]

def webui(section: str):

    def decorator(func):

        def wrapper(*args, **kwargs):
            print(f"Adding block: {func.__name__}")
            with blocks:
                iface: gr.Interface = func(blocks, *args, **kwargs)
                iface.queue()
            
        return wrapper

    return decorator


def launch():
    #interfaces[0].launch(prevent_thread_lock=True)
    pass   
