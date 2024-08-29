from fastapi.responses import JSONResponse
import modules.plugins as plugins


@plugins.router.post("/unload", tags=["Administrative Tools"])
async def unload():
    plugins.unload_all()
    return JSONResponse(content={"message": "Unloaded all plugins"})


@plugins.router.get("/unload", tags=["Administrative Tools"])
async def unload_get():
    plugins.unload_all()
    return JSONResponse(content={"message": "Unloaded all plugins"})
