# Transcription Segment Aligner

Align **transcript** to **audio or video** to get **word-level timings**.

Export as **JSON**, **SRT** or **CSV**.

![Demo](https://i.imgur.com/P4bv15M.gif)

## Setup

1. [Python 3.10+](https://www.python.org/) and [ffmpeg](https://ffmpeg.org) (on `PATH`).
2. Create a virtual environment, install dependencies and start the app:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

3. Open **http://127.0.0.1:8000**.

**macOS/Linux:** use `source venv/bin/activate` instead of `venv\Scripts\activate`.

**Windows:** after venv exists, running `start.bat` starts everything.

**Stack:** [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/) for local app, [WhisperX](https://github.com/m-bain/whisperX) and [PyTorch](https://pytorch.org/) for alignment (CUDA used automatically when available), [librosa](https://librosa.org/) for timing, [ffmpeg](https://ffmpeg.org/) on `PATH` for media decode.

## To-do

- Add manual segmentation fine-tuning
- Better language support
- Cross-file management

## License

[GNU General Public License v2.0](LICENSE) (GPL-2.0).
