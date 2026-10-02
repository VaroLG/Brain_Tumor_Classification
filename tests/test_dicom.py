import numpy as np
import pandas as pd
import pytest

pydicom = pytest.importorskip("pydicom")
from pydicom.dataset import Dataset, FileMetaDataset  # noqa: E402
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid  # noqa: E402

from btc.data.dicom import build_dicom_manifest  # noqa: E402


def _write_series(folder, patient_id, description, n_slices=20):
    """Escribe una serie DICOM MR mínima pero válida con un cuadrado brillante."""
    folder.mkdir(parents=True)
    series_uid = generate_uid()
    for i in range(1, n_slices + 1):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = MRImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = Dataset()
        ds.file_meta = meta
        ds.SOPClassUID, ds.SOPInstanceUID = MRImageStorage, meta.MediaStorageSOPInstanceUID
        ds.PatientID, ds.SeriesDescription = patient_id, description
        ds.SeriesInstanceUID, ds.InstanceNumber = series_uid, i
        arr = np.zeros((32, 32), np.uint16)
        arr[4:28, 4:28] = 300 + i
        ds.Rows, ds.Columns = arr.shape
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
        ds.PixelData = arr.tobytes()
        ds.save_as(folder / f"{i:03d}.dcm", enforce_file_format=True)


def test_build_dicom_manifest_filters_series_and_patients(tmp_path):
    root = tmp_path / "REMBRANDT"
    _write_series(root / "P1" / "t1post", "P1", "AX T1 POST GD")
    _write_series(root / "P1" / "t1pre", "P1", "AX T1 PRE")  # excluida por "PRE"
    _write_series(root / "P2" / "t1post", "P2", "AX T1 POST GD")
    _write_series(root / "P3" / "t1post", "P3", "AX T1 POST GD")  # sin etiqueta
    labels = pd.DataFrame({"patient_id": ["P1", "P2"], "label": ["II", "IV"]})

    df = build_dicom_manifest(root, labels, tmp_path / "png", ["T1"], ["PRE"], max_slices=4)
    assert set(df["patient_id"]) == {"P1", "P2"}
    assert df["series_uid"].nunique() == 2  # solo las series post-contraste
    assert (df.groupby("patient_id").size() == 4).all()
    # franja central (40 %) de 20 cortes -> instancias 7..14 aprox.
    assert df["instance_number"].between(6, 15).all()


def test_build_dicom_manifest_manual_selection(tmp_path):
    root = tmp_path / "R"
    _write_series(root / "P1" / "s", "P1", "T1 POST")
    labels = pd.DataFrame({"patient_id": ["P1"], "label": ["III"]})
    selected = pd.DataFrame({"patient_id": ["P1", "P1"], "instance_number": [2, 19]})
    df = build_dicom_manifest(root, labels, tmp_path / "png", selected_slices=selected)
    assert sorted(df["instance_number"]) == [2, 19]
