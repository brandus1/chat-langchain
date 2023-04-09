"""Main entrypoint for the app."""
import os
import logging
import pickle
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from langchain.vectorstores import VectorStore

from callback import QuestionGenCallbackHandler, StreamingLLMCallbackHandler
from query_data import get_chain
from schemas import ChatResponse

import yt_dlp # client to many multimedia portals

from pydantic import BaseModel
class VideoIngestRequest(BaseModel):
    url: str

#load dot env
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")
vectorstore: Optional[VectorStore] = None


@app.on_event("startup")
async def startup_event():
    logging.info("loading vectorstore")
    if not Path("vectorstore.pkl").exists():
        raise ValueError("vectorstore.pkl does not exist, please run ingest.py first")
    with open("vectorstore.pkl", "rb") as f:
        global vectorstore
        vectorstore = pickle.load(f)


@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    question_handler = QuestionGenCallbackHandler(websocket)
    stream_handler = StreamingLLMCallbackHandler(websocket)
    chat_history = []
    qa_chain = get_chain(vectorstore, question_handler, stream_handler)
    # Use the below line instead of the above line to enable tracing
    # Ensure `langchain-server` is running
    # qa_chain = get_chain(vectorstore, question_handler, stream_handler, tracing=True)

    while True:
        try:
            # Receive and send back the client message
            question = await websocket.receive_text()
            resp = ChatResponse(sender="you", message=question, type="stream")
            await websocket.send_json(resp.dict())

            # Construct a response
            start_resp = ChatResponse(sender="bot", message="", type="start")
            await websocket.send_json(start_resp.dict())

            result = await qa_chain.acall(
                {"question": question, "chat_history": chat_history}
            )
            chat_history.append((question, result["answer"]))

            end_resp = ChatResponse(sender="bot", message="", type="end")
            await websocket.send_json(end_resp.dict())
        except WebSocketDisconnect:
            logging.info("websocket disconnect")
            break
        except Exception as e:
            logging.error(e)
            resp = ChatResponse(
                sender="bot",
                message="Sorry, something went wrong. Try again.",
                type="error",
            )
            await websocket.send_json(resp.dict())

@app.get("/summarize")
async def summarize():
    from langchain.chains.summarize import load_summarize_chain
    from langchain.llms import OpenAI
    chain = load_summarize_chain(OpenAI(temperature=0), chain_type="refine")
    from ingest import load_docs    
    docs = load_docs()
    summary = chain.run(input_documents=docs, return_only_outputs=True) 
    # save summary to file
    with open("summary.txt", "w") as f:
        f.write(summary)
    return summary

@app.post("/video")
async def ingest_video(request: VideoIngestRequest):
    video_url = request.url
    mp3_file = download_audio(video_url, output_path="data")
    print(f"Downloaded {mp3_file}")
    txt_file = transcribe_audio(mp3_file)
    from ingest import ingest_docs
    ingest_docs()

# downloads yt_url to the same directory from which the script runs
def download_audio(yt_url, output_path="data"):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{output_path}/%(title)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }

    # Extract video information
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        video_info = ydl.extract_info(yt_url, download=False)
        video_title = video_info['title']
        video_ext = 'mp3'

    # Check if the file already exists
    output_filename = f'{output_path}/{video_title}.{video_ext}'
    if os.path.isfile(output_filename):
        print(f'Audio file for "{video_title}" already exists.')
        return output_filename

    # Download the video if it doesn't exist
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([yt_url])

    # return the file path
    return output_filename

def transcribe_audio(file_path):
    "Transcribe using Whisper "
    out_path = os.path.splitext(file_path)[0] + ".txt"
    if os.path.isfile(out_path):
        print(f"Transcription for {file_path} already exists.")
        return out_path

    import whisper
    model = whisper.load_model("tiny")
    result = model.transcribe(file_path)
    # save the result to a file
    with open(out_path, "w") as f:
        f.write(result['text'])
    return out_path

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000)
