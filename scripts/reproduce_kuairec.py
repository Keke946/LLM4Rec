"""Fixed, auditable KuaiRec reproduction. Test is scored only after both models fit."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', choices=['historical', 'strict'], default='strict')
    parser.add_argument('--raw-dir', default='data/raw/kuairec_official/extracted')
    parser.add_argument('--skip-prepare', action='store_true')
    parser.add_argument('--evaluate-only', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    protocol = args.protocol
    suffix = 'kuairec' if protocol == 'historical' else 'kuairec_strict'
    processed = f'data/processed/{suffix}'
    features = f'data/features/{suffix}_trend1500'
    base_features = f'data/features/{suffix}'
    retrievers = f'models/{suffix}/retrievers'
    models = f'models/{suffix}_trend1500'
    reports = Path(f'reports/reproduction/{protocol}')
    reports.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, 'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2',
           'PYTHONHASHSEED': '42', 'HF_HUB_OFFLINE': '1'}
    commands = []

    def run(script, *arguments):
        command = [sys.executable, f'scripts/{script}.py', *map(str, arguments)]
        started = time.time()
        log = reports / f'{len(commands):02d}-{script}.log'
        print(shlex.join(command), flush=True)
        with log.open('w') as output:
            result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, env=env)
        commands.append(dict(command=command, exit_code=result.returncode,
                             elapsed_seconds=time.time()-started, log=str(log)))
        (reports / 'commands.json').write_text(json.dumps(commands, indent=2))
        if result.returncode:
            raise RuntimeError(f'{script} failed; see {log}')

    if not args.evaluate_only:
        if not args.skip_prepare:
            run('prepare_dataset', '--dataset', 'kuairec', '--raw_dir', args.raw_dir,
                '--output_dir', processed, '--kuairec_matrix', 'big',
                '--positive_watch_ratio_threshold', 1.0, '--sample_users', 1000,
                '--max_interactions', 300000, '--max_interactions_strategy', 'random',
                '--max_interactions_seed', 42, '--no_sample_fallback',
                '--min_users', 1000, '--min_items', 5000, '--min_interactions', 300000,
                '--split_strategy', 'per_user_temporal' if protocol == 'historical' else 'global_temporal')
        run('build_features', '--dataset', 'kuairec', '--processed_dir', processed,
            '--output_dir', base_features, '--max_history_len', 50)
        common = ['--dataset', 'kuairec', '--processed_dir', processed,
                  '--features_dir', base_features, '--output_dir', features,
                  '--retrievers', 'trend', '--candidate_pool_k', 1500,
                  '--retriever_top_k', 1500, '--max_history_len', 50,
                  '--trend_half_life_days', 2.0, '--trend_category_weight', 0.4,
                  '--trend_tag_weight', 0.2, '--trend_popularity_floor', 0.02,
                  '--trend_candidate_cache_k', 2000, '--source_weights', 'trend=1.6']
        run('build_ranker_candidates', *common, '--retriever_dir', retrievers, '--splits', 'val,test', '--force_retrain_retrievers')
        run('build_temporal_ranker_train', *common, '--holdout_quantile', 0.8)
        run('train_gbdt_ranker', '--dataset', 'kuairec', '--features_dir', features,
            '--output_dir', f'{models}/gbdt', '--train_split', 'train', '--val_split', 'val',
            '--max_iter', 120, '--learning_rate', 0.07, '--max_leaf_nodes', 31,
            '--l2_regularization', 0.05, '--max_train_rows', 700000,
            '--max_val_rows', 500000, '--sample_seed', 42)
        run('train_lambdarank_ranker', '--dataset', 'kuairec', '--features_dir', features,
            '--output_dir', f'{models}/lambdarank', '--train_split', 'train', '--val_split', 'val',
            '--n_estimators', 120, '--learning_rate', 0.05, '--num_leaves', 31,
            '--max_train_rows', 700000, '--max_val_rows', 500000, '--random_state', 42)
    run('evaluate_reproduction', '--processed-dir', processed, '--features-dir', features,
        '--models', models, '--retrievers', retrievers, '--output-dir', reports)
    run('diagnose_candidate_recall', '--dataset', 'kuairec', '--processed_dir', processed,
        '--features_dir', features, '--output_dir', reports, '--split', 'test')
    import importlib.metadata as metadata
    packages = ['numpy', 'pandas', 'scikit-learn', 'torch', 'pyarrow', 'lightgbm']
    artifacts = {}
    for directory in [Path(processed), Path(features), Path(models)]:
        for path in sorted(directory.rglob('*')):
            if path.is_file():
                with path.open('rb') as stream:
                    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                artifacts[str(path)] = dict(bytes=path.stat().st_size, sha256=digest)
    meta = dict(protocol=protocol, seed=42, python=sys.version, platform=platform.platform(),
                packages={name: metadata.version(name) for name in packages},
                env={name: env[name] for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTHONHASHSEED']},
                artifacts=artifacts)
    (reports/'run_manifest.json').write_text(json.dumps(meta, indent=2))
    print(f'Completed {protocol}: {reports}', flush=True)


if __name__ == '__main__':
    main()
