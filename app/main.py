from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.deeplabcut_routes import router as deeplabcut_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(deeplabcut_router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
