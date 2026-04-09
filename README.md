# Word Aligner

Align **transcript** to **audio or video** to get **word-level timings**.

Export as **JSON**, **SRT** or **CSV**.

![Demo](https://i.imgur.com/P4bv15M.gif)

## Setup

1. [Python 3.10+](https://www.python.org/) and [ffmpeg](https://ffmpeg.org) (must be on your `PATH` — the server uses it to decode media).
2. Create a virtual environment, install dependencies, and start the app:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

3. Open **http://127.0.0.1:8000** in your browser (that is where the app listens).

**macOS/Linux:** use `source venv/bin/activate` instead of `venv\Scripts\activate`.

**Windows:** after the venv exists, you can run `start.bat` instead - starts the server and opens the browser when the app is ready.

## License

[GNU General Public License v2.0](LICENSE) (GPL-2.0).
