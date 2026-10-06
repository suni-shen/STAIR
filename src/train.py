import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import spearmanr

from data import SOURCES, load_embeddings, make_loader, read_records
from losses import stair_loss, task_loss
from model import STAIR


ROOT = Path(__file__).resolve().parents[1]
TASKS = ("aav", "gb1", "gfp", "location", "meltome", "stability")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train_epoch(model, loader, optimizer, training, device, classification, epoch):
    model.train()
    total = 0.0
    for _, protein, target in loader:
        protein, target = protein.to(device), target.to(device)
        prediction, mask, aux = model(protein, epoch, training["iteration"])
        loss = stair_loss(prediction, target, mask, aux, model.projectors, training, classification)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * len(target)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, classification):
    model.eval()
    total, ids, targets, predictions = 0.0, [], [], []
    for batch_ids, protein, target in loader:
        protein, target = protein.to(device), target.to(device)
        prediction, _, _ = model(protein)
        total += task_loss(prediction, target, classification).item() * len(target)
        output = prediction.argmax(dim=1) if classification else prediction.squeeze(-1)
        ids.extend(batch_ids)
        targets.append(target.cpu())
        predictions.append(output.cpu())
    targets = torch.cat(targets).numpy()
    predictions = torch.cat(predictions).numpy()
    metric = float((targets == predictions).mean()) if classification else float(spearmanr(targets, predictions).statistic)
    return {"loss": total / len(loader.dataset), "metric": metric}, list(zip(ids, targets.tolist(), predictions.tolist()))


def fit(model, loaders, config, model_config, dataset, seed, device, checkpoint_path):
    training = config["training"]
    classification = dataset == "location"
    optimizer = torch.optim.Adam(model.parameter_groups(training))
    best_loss = float("inf")
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, training["iteration"] + 1):
        loss = train_epoch(model, loaders["train"], optimizer, training, device, classification, epoch)
        validation, _ = evaluate(model, loaders["validation"], device, classification)
        if not np.isfinite(loss) or not np.isfinite(validation["loss"]):
            raise ValueError(f"Non-finite loss at epoch {epoch}")
        print(f"epoch={epoch} train_loss={loss:.6f} val_loss={validation['loss']:.6f} "
              f"val_metric={validation['metric']:.6f}", flush=True)
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            torch.save({
                "state_dict": model.state_dict(), "model_config": model_config,
                "config": config, "dataset": dataset, "seed": seed,
                "epoch": epoch, "validation": validation,
            }, checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description="STAIR training and inference")
    parser.add_argument("--mode", choices=("train", "inference"), required=True)
    parser.add_argument("--dataset", choices=TASKS, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--embeddings-dir", type=Path, default=ROOT / "embeddings")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topk", type=int, choices=range(1, 7))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.mode == "inference" and args.checkpoint is None:
        parser.error("--checkpoint is required for inference")
    if args.mode == "train" and args.checkpoint is not None:
        parser.error("--checkpoint is an inference input; training saves to --output-dir")
    if args.mode == "inference" and (args.topk is not None or args.epochs is not None):
        parser.error("Inference uses the checkpoint's model settings; omit --topk and --epochs")
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be positive")
    try:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; use --device cpu")
        if args.mode == "inference":
            checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
            if checkpoint["dataset"] != args.dataset:
                raise ValueError("The checkpoint dataset does not match --dataset")
            config = checkpoint["config"]
            model_config = checkpoint["model_config"]
            seed = checkpoint["seed"]
        else:
            with (ROOT / "configs" / f"{args.dataset}.yaml").open() as handle:
                config = yaml.safe_load(handle)
            if args.epochs is not None:
                config["training"]["iteration"] = args.epochs
            model_config = {
                "source_dims": [dim for _, dim in SOURCES],
                "hidden_dim": config["hidden_dimension"],
                "output_dim": 10 if args.dataset == "location" else 1,
                "topk": args.topk if args.topk is not None else 6,
            }
            seed = args.seed
        if args.batch_size is not None:
            config["training"]["batch_size"] = args.batch_size
        set_seed(seed)
        classification = args.dataset == "location"
        partitions = ("train", "validation", "test") if args.mode == "train" else ("test",)
        records = {part: read_records(args.data_dir / config["directories"][part], classification)
                   for part in partitions}
        ids = [pid for part in records.values() for pid, _ in part]
        embeddings = load_embeddings(args.embeddings_dir, args.dataset, ids)
        loaders = {part: make_loader(rows, embeddings, classification,
                                    config["training"]["batch_size"], part == "train")
                   for part, rows in records.items()}
        model = STAIR(**model_config).to(device)
        output_dir = args.output_dir / args.dataset / f"seed_{seed}"
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.mode == "train":
            checkpoint = fit(model, loaders, config, model_config, args.dataset, seed,
                             device, output_dir / "best.pt")
        else:
            model.load_state_dict(checkpoint["state_dict"])
        test, predictions = evaluate(model, loaders["test"], device, classification)
        metrics = {
            "dataset": args.dataset, "seed": seed, "best_epoch": checkpoint["epoch"],
            "metric_name": "accuracy" if classification else "spearman",
            "validation": checkpoint["validation"], "test": test,
        }
        with (output_dir / "metrics.json").open("w") as handle:
            json.dump(metrics, handle, indent=2)
        with (output_dir / "predictions.tsv").open("w", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(("protein_id", "target", "prediction"))
            writer.writerows(predictions)
        print(json.dumps(metrics, indent=2))
        print(f"Output: {output_dir}")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
