from typing import List

import moduleconf
import numpy as np
import pydub
import torch
from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def __init__(self):
        self.device = None
        self.model = None

    @classmethod
    def readAudio(cls, path, normalize=True):
        audio = pydub.AudioSegment.from_file(path)
        y = np.array(audio.get_array_of_samples())
        y = y.reshape(-1, audio.channels)
        if normalize:
            y = np.float32(y) / 2 ** 15
        return audio.frame_rate, y

    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        confPath = "transkun/pretrained/2.0.conf"
        confManager = moduleconf.parseFromFile(confPath)
        TransKun = confManager["Model"].module.TransKun
        conf = confManager["Model"].config
        device = "cuda"

        checkpoint = torch.load("transkun/pretrained/2.0.pt", map_location=device)
        self.model = TransKun(conf=conf).to(device)

        if "best_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["best_state_dict"], strict=False)
        else:
            self.model.load_state_dict(checkpoint["state_dict"], strict=False)

        self.model.eval()
        self.device = device

    def predict(
            self,
            audio: Path = Input(description="Input audio file (wav, mp3, etc.)"),
            step_seconds: int = Input(description="Input audio step seconds.", default=8),
            segment_size: int = Input(description="Input audio segment size.", default=16),
    ) -> List:
        """Run a single prediction on the model"""
        fs, audio = Predictor.readAudio(audio)

        if fs != self.model.fs:
            import soxr
            audio = soxr.resample(
                audio,  # 1D(mono) or 2D(frames, channels) array input
                fs,  # input samplerate
                self.model.fs  # target samplerate
            )

        with torch.no_grad():
            x = torch.from_numpy(audio).to(self.device)
            notesEst = self.model.transcribe(x, stepInSecond=step_seconds, segmentSizeInSecond=segment_size, discardSecondHalf=False)

        return [note.__dict__ for note in notesEst]