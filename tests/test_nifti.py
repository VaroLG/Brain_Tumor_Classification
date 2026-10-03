import numpy as np
import pandas as pd
import pytest

nib = pytest.importorskip("nibabel")

from btc.data.nifti import (  # noqa: E402
    SliceSelection,
    build_nifti_manifest,
    select_slice_indices,
    split_subject_id,
)


def test_split_subject_id():
    assert split_subject_id("UPENN-GBM-00001_11") == ("UPENN-GBM-00001", "11")
    assert split_subject_id("raro") == ("raro", "")


def test_select_slices_follows_tumor_and_caps():
    mask = np.zeros((20, 20, 50), np.uint8)
    mask[5:15, 5:15, 10:40] = 1  # tumor en los cortes axiales 10..39
    idx = select_slice_indices(mask, SliceSelection(axis="axial", max_slices=5))
    assert len(idx) == 5
    assert all(10 <= i < 40 for i in idx)
    assert select_slice_indices(np.zeros_like(mask), SliceSelection()) == []


def _write_subject(root, subject, with_mask=True):
    vol = np.zeros((40, 40, 30), np.float32)
    vol[5:35, 5:35, :] = np.random.default_rng(0).random((30, 30, 30)) * 100 + 50
    seg = np.zeros(vol.shape, np.uint8)
    seg[15:25, 15:25, 10:20] = 4
    d = root / "images_structural" / subject
    d.mkdir(parents=True)
    nib.save(nib.Nifti1Image(vol, np.eye(4)), d / f"{subject}_T1GD.nii.gz")
    if with_mask:
        s = root / "images_segm"
        s.mkdir(exist_ok=True)
        nib.save(nib.Nifti1Image(seg, np.eye(4)), s / f"{subject}_segm.nii.gz")


def test_build_nifti_manifest(tmp_path):
    root = tmp_path / "UPENN"
    _write_subject(root, "UPENN-GBM-00001_11")
    _write_subject(root, "UPENN-GBM-00001_21")  # seguimiento: excluido por defecto
    _write_subject(root, "UPENN-GBM-00002_11")
    _write_subject(root, "UPENN-GBM-00003_11", with_mask=False)  # sin máscara: se omite
    labels = pd.DataFrame(
        {
            "patient_id": ["UPENN-GBM-00001", "UPENN-GBM-00002", "UPENN-GBM-00003"],
            "label": ["short", "long", "long"],
        }
    )

    df = build_nifti_manifest(
        root, labels, tmp_path / "png", selection=SliceSelection(max_slices=3)
    )
    assert set(df["patient_id"]) == {"UPENN-GBM-00001", "UPENN-GBM-00002"}
    assert set(df["subject"]) == {"UPENN-GBM-00001_11", "UPENN-GBM-00002_11"}
    assert (df.groupby("patient_id").size() == 3).all()
    assert all(10 <= i < 20 for i in df["slice_idx"])
    assert all(p.endswith(".png") for p in df["image_path"])
