from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


SOURCES = (
    ("esm3", 1536),
    ("ontoprotein", 1024),
    ("proteinclip_t5", 128),
    ("protst", 512),
    ("protrek", 1024),
    ("proteindt", 1024),
)


def read_records(path, classification=False):
    convert = int if classification else float
    with Path(path).open() as handle:
        records = [(pid, convert(target)) for pid, target in
                   (line.rstrip("\n").split("\t") for line in handle)]
    if not records:
        raise ValueError(f"Empty dataset: {path}")
    return records


def load_embeddings(embedding_dir, dataset, protein_ids):
    paths = [Path(embedding_dir) / dataset / name / "protein_dictionary.pt"
             for name, _ in SOURCES]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Embeddings are not included in this package. "
                                "Provide --embeddings-dir. Missing files:\n" + "\n".join(missing))
    protein_ids = list(dict.fromkeys(protein_ids))
    matrices = []
    for path, (_, dim) in zip(paths, SOURCES):
        dictionary = torch.load(path, map_location="cpu", weights_only=True)
        absent = [pid for pid in protein_ids if pid not in dictionary]
        if absent:
            raise ValueError(f"{path}: missing {len(absent)} protein IDs; first: {absent[0]}")
        matrix = torch.stack([dictionary[pid] for pid in protein_ids]).float()
        if matrix.shape != (len(protein_ids), dim):
            raise ValueError(f"{path}: expected ({len(protein_ids)}, {dim}), got {tuple(matrix.shape)}")
        matrices.append(matrix)
    return dict(zip(protein_ids, torch.cat(matrices, dim=1)))


class BenchmarkDataset(Dataset):
    def __init__(self, records, embeddings, classification=False):
        self.records = records
        self.embeddings = embeddings
        self.dtype = torch.long if classification else torch.float32

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        pid, target = self.records[index]
        return pid, self.embeddings[pid], torch.tensor(target, dtype=self.dtype)


def make_loader(records, embeddings, classification, batch_size, shuffle=False):
    dataset = BenchmarkDataset(records, embeddings, classification)
    if shuffle and (batch_size < 2 or len(dataset) % batch_size == 1):
        raise ValueError("Training needs at least two samples per batch for BatchNorm and HSIC; "
                         "choose a batch size that leaves no single-sample batch.")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
