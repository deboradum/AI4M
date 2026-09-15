import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader

from main import img_transform, gt_transform
from dataset import SliceDataset
from ENet import ENet
from losses import CrossEntropy
from utils import probs2one_hot, probs2class, dice_coef, save_images


def write_json(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def save_checkpoint(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(data, tmp)
    tmp.replace(path)


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def synchronize():
    torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--workers', type=int, default=5)
    parser.add_argument('--dest', type=Path, required=True)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    if args.epochs < 1 or args.workers < 0:
        raise ValueError('Invalid epoch or worker count')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: run in a GPU allocation, with ai4mi activated.')
    args.dest.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {}
    patients = {}
    for split in ('train', 'val'):
        ds = SliceDataset(split, Path('data/SEGTHOR'), img_transform=img_transform,
                          gt_transform=partial(gt_transform, 5), debug=args.debug)
        if not len(ds):
            raise ValueError(f'No {split} slices in data/SEGTHOR')
        pairs = ds.files
        if any(gt is None or img.stem != gt.stem for img, gt in pairs):
            raise ValueError('Image/label mismatch')
        patients[split] = sorted({p.stem.rsplit('_', 1)[0] for p, _ in pairs})
        loaders[split] = DataLoader(ds, batch_size=8, shuffle=split == 'train',
            num_workers=args.workers, persistent_workers=args.workers > 0,
            worker_init_fn=seed_worker, generator=generator)
    if set(patients['train']) & set(patients['val']):
        raise ValueError('Patient leakage between training and validation')
    net = ENet(1, 5, kernels=8, factor=2).cuda()
    net.init_weights()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0005, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
    loss_fn = CrossEntropy(idk=list(range(5)))
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    source_hashes = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                     for name in ('train_16.py', 'main.py', 'ENet.py', 'dataset.py', 'losses.py', 'utils.py')}
    config = dict(exp_id=f'E006_cosine_{args.epochs}', status='training',
        change='Longer training with cosine learning-rate decay', epochs=args.epochs,
        seed=args.seed, model='ENet', kernels=8, factor=2, in_slices=1,
        loss='CrossEntropy', optimizer='Adam', learning_rate=0.0005,
        betas=[0.9, 0.999], batch_size=8, scheduler='CosineAnnealingLR',
        eta_min=0, resolution=[256, 256], dataset='data/SEGTHOR',
        patients=patients, classes=['background','esophagus','heart','trachea','aorta'],
        workers=args.workers, persistent_workers=args.workers > 0,
        git_commit=commit, source_sha256=source_hashes, torch_version=torch.__version__,
        gpu=torch.cuda.get_device_name(0), job_id=os.getenv('SLURM_JOB_ID'),
        assertions_enabled=__debug__, debug=args.debug,
        model_selection='Mean 2D validation Dice over classes 1..4, matching starter',
        comparison_note='Previous personal E000 was unseeded; comparison changes seed as well as schedule. Group E001 uses seed 123.',
        metrics_3d=None)
    write_json(args.dest / 'config.json', config)
    history = []
    best = -1.0
    best_epoch = -1
    torch.cuda.reset_peak_memory_stats()
    synchronize()
    started = time.perf_counter()
    for epoch in range(args.epochs):
        record = {'epoch': epoch, 'learning_rate': optimizer.param_groups[0]['lr']}
        for split, loader in loaders.items():
            net.train(split == 'train')
            synchronize()
            phase_start = time.perf_counter()
            losses, dices = [], []
            with torch.set_grad_enabled(split == 'train'):
                for batch_index, batch in enumerate(loader):
                    img, gt = batch['images'].cuda(), batch['gts'].cuda()
                    if img.shape[1:] != (1, 256, 256):
                        raise ValueError(f'Expected one 256x256 slice: {img.shape}')
                    if split == 'train':
                        optimizer.zero_grad()
                    probs = net(img).softmax(dim=1)
                    loss = loss_fn(probs, gt)
                    with torch.no_grad():
                        dices.append(dice_coef(probs2one_hot(probs), gt).cpu())
                    if split == 'train':
                        loss.backward()
                        optimizer.step()
                    losses.append(loss.item())
                    if batch_index % 50 == 0:
                        print(f'Epoch {epoch+1}/{args.epochs} {split}: batch {batch_index+1}/{len(loader)}', flush=True)
            synchronize()
            record[split + '_seconds'] = time.perf_counter() - phase_start
            record[split + '_loss'] = float(np.mean(losses))
            per_class = torch.cat(dices).mean(dim=0)
            record[split + '_dice_per_class'] = per_class.tolist()
            record[split + '_dice_2d'] = per_class[1:].mean().item()
        if record['val_dice_2d'] > best:
            best, best_epoch = record['val_dice_2d'], epoch
            save_checkpoint(args.dest / 'bestweights.pt', net.state_dict())
            save_checkpoint(args.dest / 'bestmodel.pkl', net)
            (args.dest / 'best_epoch.txt').write_text(f'Epoch {epoch}: 2D validation Dice {best:.6f}\n')
        scheduler.step()
        history.append(record)
        save_checkpoint(args.dest / 'last.pt', dict(epoch=epoch, model=net.state_dict(),
            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), best_epoch=best_epoch,
            best_dice=best, history=history, config=config, python_rng=random.getstate(),
            numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(), loader_rng=generator.get_state()))
        stats = dict(epochs_done=epoch+1, best_epoch=best_epoch, best_val_dice_2d=best,
            total_minutes=(time.perf_counter()-started)/60,
            peak_gpu_memory_mb=torch.cuda.max_memory_allocated()/2**20,
            parameter_count=sum(p.numel() for p in net.parameters()), history=history)
        write_json(args.dest / 'stats.json', stats)
        print(f"Epoch {epoch+1}: train {record['train_seconds']:.1f}s, val {record['val_seconds']:.1f}s, Dice {record['val_dice_2d']:.4f}", flush=True)
    net.load_state_dict(torch.load(args.dest / 'bestweights.pt', map_location='cuda', weights_only=True))
    net.eval()
    synchronize()
    inference_start = time.perf_counter()
    with torch.no_grad():
        for batch in loaders['val']:
            classes = probs2class(net(batch['images'].cuda()).softmax(dim=1))
            save_images(classes * 63, batch['stems'], args.dest / 'best_epoch/val')
    synchronize()
    config.update(status='training_and_prediction_complete', results=stats,
        prediction_seconds=time.perf_counter()-inference_start,
        prediction_timing_note='Includes PNG output and loading; excludes stitching and 3D metrics')
    write_json(args.dest / 'config.json', config)
    print(f'Finished. Model, predictions and records: {args.dest}', flush=True)


if __name__ == '__main__':
    main()
