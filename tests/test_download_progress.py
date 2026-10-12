import threading

import gradio as gr
from gradio.context import LocalContext
from tqdm import tqdm

from app import track_model_download


def test_xet_download_shows_one_bar_that_advances_from_other_threads(monkeypatch):
    sent = []
    event_thread = threading.get_ident()
    # Gradio only finds its request context on the thread running the event.
    monkeypatch.setattr(
        gr.Progress,
        "_progress_callback",
        staticmethod(
            lambda: sent.append if threading.get_ident() == event_thread else None
        ),
    )
    progress = gr.Progress(track_tqdm=True)
    token = LocalContext.progress.set(progress)
    try:
        track_model_download(progress)
        written = tqdm(desc="model.safetensors: reconstructing file", total=100)
        network = tqdm(desc="model.safetensors: downloading bytes", total=100)

        def report():
            written.update(30)
            network.update(20)

        reporter = threading.Thread(target=report)
        reporter.start()
        reporter.join()

        assert [(bar.desc, bar.index) for bar in sent[-1]] == [
            ("model.safetensors: reconstructing file", 30)
        ]
        written.close()
        network.close()
    finally:
        LocalContext.progress.reset(token)
