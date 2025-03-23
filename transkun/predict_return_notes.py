import os.path

import moduleconf
import numpy as np
import pydub
import torch


def readAudio(path, normalize=True):
    audio = pydub.AudioSegment.from_file(path)
    y = np.array(audio.get_array_of_samples())
    y = y.reshape(-1, audio.channels)
    if normalize:
        y = np.float32(y) / 2 ** 15
    return audio.frame_rate, y


MODULE_PATH = os.path.abspath(__file__ + "/..")
confPath = os.path.join(MODULE_PATH, "pretrained/2.0.conf")
confManager = moduleconf.parseFromFile(confPath)
TransKun = confManager["Model"].module.TransKun
conf = confManager["Model"].config
device = "cuda"

checkpoint = torch.load(f"{MODULE_PATH}/pretrained/2.0.pt", map_location=device)
model = TransKun(conf=conf).to(device)

if "best_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["best_state_dict"], strict=False)
else:
    model.load_state_dict(checkpoint["state_dict"], strict=False)

model.eval()


def predict(audio, step_seconds=8, segment_size=16):
    fs, audio = readAudio(audio)

    if fs != model.fs:
        import soxr
        audio = soxr.resample(
            audio,  # 1D(mono) or 2D(frames, channels) array input
            fs,  # input samplerate
            model.fs  # target samplerate
        )

    with torch.no_grad():
        x = torch.from_numpy(audio).to(device)
        notesEst = model.transcribe(x, stepInSecond=step_seconds, segmentSizeInSecond=segment_size, discardSecondHalf=False)

    return [note.__dict__ for note in notesEst]
