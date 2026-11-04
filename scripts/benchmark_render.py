"""Benchmark reproducible: python scripts/benchmark_render.py [--seconds 180 --tracks 8]."""
import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from osky_remix.core.config import workspace
from osky_remix.core.assets import AssetStore
from osky_remix.core.cache import CacheStore
from osky_remix.core.models import AssetKind, Project, Provenance
from osky_remix.core.operations import Context, RegisterAsset
from osky_remix.core.transactions import apply_ops, parse_ops
from osky_remix.audio.renderer import Renderer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=int, default=180)
    parser.add_argument('--tracks', type=int, default=8)
    args = parser.parse_args()
    root = workspace() / 'benchmarks' / f'{args.seconds}s-{args.tracks}tracks'
    root.mkdir(parents=True, exist_ok=True)
    sr = 44100
    path = root / 'source.wav'
    block = .02 * np.sin(2*np.pi*330*np.arange(sr)/sr).astype(np.float32)
    with sf.SoundFile(path, 'w', samplerate=sr, channels=2, subtype='FLOAT') as f:
        for _ in range(args.seconds):
            f.write(np.stack([block, block], 1))
    assets, cache = AssetStore(root/'assets'), CacheStore(root/'cache')
    asset = assets.ingest(path, AssetKind.sample, Provenance(tool='benchmark'))
    ops = []
    for i in range(args.tracks):
        ops.extend([{'op':'add_track', 'id':f't{i}', 'name':f'T{i}', 'kind':'audio'},
                    {'op':'add_clip','track':f't{i}', 'clip':{'type':'audio','id':f'c{i}',
                     'asset':asset.sha256, 'start_tick':0, 'source_length':args.seconds*sr}}])
    project = apply_ops(Project(name='benchmark',sample_rate=sr), [RegisterAsset(asset=asset)]+parse_ops(ops), Context(author='user'))
    started = time.monotonic()
    report = Renderer(project, assets, cache).render(root/'render', 'streaming', stems=False)
    samples, rate = sf.read(report.files['master'], frames=sr)
    assert rate == sr and np.isfinite(samples).all() and np.max(np.abs(samples)) > 0
    assert report.mastering_converged and abs(report.lufs + 14) <= .3
    metrics = {'seconds':args.seconds,'tracks':args.tracks,'elapsed_s':round(time.monotonic()-started,2),
               'peak_rss_mib':round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,1),
               'lufs':report.lufs,'true_peak_dbtp':report.true_peak_dbtp}
    (root/'metrics.json').write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics))


if __name__ == '__main__':
    main()
