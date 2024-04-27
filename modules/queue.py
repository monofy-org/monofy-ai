import asyncio
import datetime
import uuid
from fastapi import Request
from modules.plugins import router
from requests import post
from settings import HOST, PORT

# this addon offers a /queue/* endpoint that can be used to queue requests
# it accepts any request, creates an entry in the queue and returns the queue id
# the queue object is a simple list that holds the requests
# each request is a dictionary with the following keys:
# - id: asyncio task ID
# - request: the request itself
# - status: the status of the request (queued, processing, done, error)
# - result: the result of the request (if any)
# - error: the error message (if any)
# - created_at: the timestamp when the request was created
# - started_at: the timestamp when the request was started
# - finished_at: the timestamp when the request was finished
# - duration: the duration of the request

queue = []


def get_queue():

    return queue


def get_request(queue_id: str):

    for item in queue:
        if item["id"] == queue_id:
            return item

    return None


def add_request(request):
    item = {
        "id": str(uuid.uuid4()),
        "request": request,
        "status": "queued",
        "result": None,
        "error": None,
        "created_at": datetime.now(),
        "started_at": None,
        "finished_at": None,
        "duration": None,
    }
    queue.append(item)
    asyncio.create_task(process_next_item())
    return item["id"]


def update_request(queue_id: str, status: str, result=None, error=None):

    for item in queue:
        if item["id"] == queue_id:
            item["status"] = status
            item["result"] = result
            item["error"] = error
            item["finished_at"] = datetime.now()
            item["duration"] = item["finished_at"] - item["started_at"]
            return item

    return None


async def process_next_item():
    if len(queue) == 0:
        return

    item = queue.pop(0)
    post_request = item["request"]
    queue_id = item["id"]
    update_request(queue_id, "processing")

    async def process_request():
        response = post(
            f"http://{HOST}:{PORT}/api/{post_request['endpoint']}", json=post_request
        )

        if response.status_code == 200:
            update_request(queue_id, "done", result=response.json())
        else:
            update_request(queue_id, "error", error=response.text)

    task = asyncio.create_task(process_request())
    item["id"] = task.get_name()  # Update the item's ID to match the task's ID


@router.post("/queue/{endpoint:path}")
async def queue_request(
    request: Request,
):
    try:
        data = await request.json()
        queue_id = add_request(data)
        return {"queue_id": queue_id}
    except Exception as e:
        return {"error": str(e)}


@router.get("/queue/{queue_id}")
async def get_queue_item(queue_id: str):
    item = get_request(queue_id)
    return item
