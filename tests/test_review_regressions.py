from pathlib import Path
import pytest
from utils.plate_utils import normalize_iran_plate, label_from_filename

def test_presentation_forms_and_suffix():
    assert normalize_iran_plate("12ﺏ34567") == "12ب34567"
    assert label_from_filename(Path("۱۲ب۳۴۵۶۷_2.jpg")) == "12ب34567"

def test_observation_does_not_register_vehicle(db):
    assert db.record_capture("2026-09-16T12:00:00+03:30", "12ب34567", "full.jpg", "raw.jpg") > 0
    assert db.lookup("12ب34567") is None
    row = db._conn.execute("SELECT * FROM captures").fetchone()
    assert row["plate"] == "12ب34567"
    assert row["timestamp"].endswith("+03:30")

def test_audit_reports_bad_names(tmp_path):
    from tools.audit_dataset import audit
    (tmp_path / "12ب34567.jpg").touch()
    (tmp_path / "bad.jpg").touch()
    report = audit(tmp_path)
    assert report["usable_ratio"] == .5
    assert report["rejected_examples"][0]["file"] == "bad.jpg"

def test_both_photos_persist_with_database_row(db, tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    from utils.plate_saver import PlateSaver
    frame = np.zeros((50, 160, 3), dtype=np.uint8)
    with PlateSaver(tmp_path / "captures", db=db) as saver:
        assert saver.save("12ب34567", raw_crop=frame, full_frame=frame)
    row = db._conn.execute("SELECT * FROM captures").fetchone()
    assert Path(row["full_frame"]).is_file() and Path(row["plate_raw"]).is_file()
    assert saver.written == 1 and saver.failed == 0

def test_failed_image_write_does_not_create_capture_record(db, tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    import utils.plate_saver as mod
    monkeypatch.setattr(mod, "imwrite_unicode", lambda *args: False)
    with mod.PlateSaver(tmp_path / "captures", db=db) as saver:
        assert saver.save("12ب34567", raw_crop=np.zeros((4, 4, 3), dtype=np.uint8))
    assert saver.failed == 1
    assert db._conn.execute("SELECT COUNT(*) FROM captures").fetchone()[0] == 0

def test_duplicate_sources_do_not_overwrite(tmp_path, voc_xml):
    from training.prepare_dataset import rename_images_using_xml
    out = tmp_path / "out"
    chars = [(c, i * 20, 0, i * 20 + 10, 20) for i, c in enumerate("12ب34567")]
    for n in (1, 2):
        source = tmp_path / str(n)
        source.mkdir()
        (source / "x.jpg").write_bytes(str(n).encode())
        voc_xml(chars, source / "x.xml")
        assert rename_images_using_xml(source, out) == (1, 0)
    assert {p.read_bytes() for p in out.glob("*.jpg")} == {b"1", b"2"}

def test_shaped_text_not_reordered_twice():
    pytest.importorskip("cv2")
    from PIL import ImageFont
    from utils.overlay import TextRenderer
    renderer = TextRenderer()
    if renderer.font_path is None:
        pytest.skip("No system font")
    assert renderer._font(24).layout_engine == ImageFont.Layout.BASIC

@pytest.mark.parametrize("gpu,precision", [(True, 16), (False, 32)])
def test_quantize_is_integer_even_on_cpu(monkeypatch, gpu, precision):
    pytest.importorskip("torch")
    pytest.importorskip("ultralytics")
    import models.detector as mod
    class FakeYOLO:
        def __init__(self, path): self.calls = []
        def to(self, device): return self
        def predict(self, **kwargs):
            self.calls.append(kwargs)
            return []
    monkeypatch.setattr(mod, "YOLO", FakeYOLO)
    monkeypatch.setattr(mod.torch.cuda, "is_available", lambda: gpu)
    monkeypatch.setattr(mod.PlateDetector, "_half_kwarg", staticmethod(lambda: "quantize"))
    detector = mod.PlateDetector("unused.pt")
    assert type(detector.model.calls[0]["quantize"]) is int
    assert detector.model.calls[0]["quantize"] == precision

def test_close_drains_full_queue(tmp_path, monkeypatch):
    import threading
    import time
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    from utils.plate_saver import PlateSaver
    entered, release = threading.Event(), threading.Event()
    saver = PlateSaver(tmp_path / "captures", queue_size=1, min_interval_seconds=0)
    original = saver._write
    def slow_write(job):
        entered.set()
        assert release.wait(3)
        original(job)
    monkeypatch.setattr(saver, "_write", slow_write)
    frame = np.zeros((10, 30, 3), dtype=np.uint8)
    saver.save("12ب34567", raw_crop=frame)
    assert entered.wait(2)
    saver.save("34ج67890", raw_crop=frame)
    closer = threading.Thread(target=saver.close)
    closer.start()
    release.set()
    closer.join(3)
    assert not closer.is_alive() and not saver._thread.is_alive()
    assert saver.written == 2
    saver.close()

def test_plate_visual_order_inside_persian_sentence():
    pytest.importorskip("cv2")
    reshaper = pytest.importorskip("arabic_reshaper")
    pytest.importorskip("bidi")
    from utils.overlay import shape_persian
    from utils.plate_utils import format_plate_display
    rendered = shape_persian("پلاک: " + format_plate_display("12ب34567", bidi=True))
    assert reshaper.reshape("12 ب 345 | 67") in rendered

def test_invalid_qt_font_directory_is_replaced(tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    from utils.overlay import configure_qt_fonts
    monkeypatch.setenv("QT_QPA_FONTDIR", str(tmp_path / "missing"))
    font = tmp_path / "font.ttf"
    font.touch()
    configure_qt_fonts(str(font))
    import os
    assert os.environ["QT_QPA_FONTDIR"] == str(tmp_path)
