import json
import subprocess

from anime_studio import device


def _fake_probe(gpus):
    def run(*args, **kwargs):
        return subprocess.CompletedProcess(args, 0, stdout=json.dumps(gpus) + "\n", stderr="")

    return run


def test_picks_graphics_card_over_ryzen_integrated_gpu(monkeypatch):
    env = {}
    monkeypatch.setattr(device.os, "environ", env)
    gpus = [
        # Ryzen 9000 built-in graphics shows up first and can report lots of shared memory
        [0, "AMD Radeon(TM) Graphics", 1, 1, 16 * 1024**3],
        [1, "AMD Radeon RX 7600 XT", 0, 16, 16 * 1024**3],
    ]
    monkeypatch.setattr(device.subprocess, "run", _fake_probe(gpus))
    assert device.isolate_discrete_gpu() == "AMD Radeon RX 7600 XT"
    assert env["HIP_VISIBLE_DEVICES"] == "1"


def test_single_gpu_or_user_choice_is_left_alone(monkeypatch):
    env = {}
    monkeypatch.setattr(device.os, "environ", env)
    monkeypatch.setattr(device.subprocess, "run", _fake_probe([[0, "AMD Radeon RX 7600 XT", 0, 16, 1]]))
    assert device.isolate_discrete_gpu() is None
    assert env == {}

    env["HIP_VISIBLE_DEVICES"] = "0"
    monkeypatch.setattr(device.subprocess, "run", _fake_probe([[0, "a", 1, 1, 1], [1, "b", 0, 16, 1]]))
    assert device.isolate_discrete_gpu() is None
    assert env == {"HIP_VISIBLE_DEVICES": "0"}
