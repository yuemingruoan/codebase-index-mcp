import code_index.device as device


def test_resolve_device_auto_prefers_cuda(monkeypatch):
    monkeypatch.setattr(device, "is_cuda_available", lambda: True)
    monkeypatch.setattr(device, "is_mps_available", lambda: True)
    assert device.resolve_device("auto") == "cuda"


def test_resolve_device_auto_falls_back_to_mps(monkeypatch):
    monkeypatch.setattr(device, "is_cuda_available", lambda: False)
    monkeypatch.setattr(device, "is_mps_available", lambda: True)
    assert device.resolve_device("auto") == "mps"


def test_resolve_device_auto_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(device, "is_cuda_available", lambda: False)
    monkeypatch.setattr(device, "is_mps_available", lambda: False)
    assert device.resolve_device("auto") == "cpu"


def test_resolve_device_explicit_cuda_falls_back(monkeypatch):
    monkeypatch.setattr(device, "is_cuda_available", lambda: False)
    assert device.resolve_device("cuda") == "cpu"


def test_resolve_device_explicit_mps_falls_back(monkeypatch):
    monkeypatch.setattr(device, "is_mps_available", lambda: False)
    assert device.resolve_device("mps") == "cpu"
