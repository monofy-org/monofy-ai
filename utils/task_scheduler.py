import asyncio

# TODO implement this or something similar
class TaskScheduler:
    def __init__(self):
        self.task_queue = asyncio.Queue()        

    async def add(self, func, *args, **kwargs):
        task = (func, args, kwargs)
        await self.task_queue.put(task)

    async def run_scheduler(self):
        while True:
            task = await self.task_queue.get()

            func, args, kwargs = task
            await func(*args, **kwargs)

tasks = TaskScheduler()
loop = asyncio.get_event_loop()
loop.create_task(tasks.run_scheduler())
