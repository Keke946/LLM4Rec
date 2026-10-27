"""Finish all predeclared real-data module checks, excluding LLM APIs and GitHub."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['all','neural','retrievers','negative','semantic','business','docker'],default='all')
    parser.add_argument('--semantic-python',default=sys.executable)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();os.chdir(ROOT)
    import yaml
    spec=yaml.safe_load(Path('configs/kuairec_completion.yaml').read_text())
    out=Path('reports/reproduction/completion');out.mkdir(parents=True,exist_ok=True)
    (out/'frozen_spec.yaml').write_text(Path('configs/kuairec_completion.yaml').read_text())
    record=out/'execution.json'
    records=json.loads(record.read_text()) if record.exists() else []
    env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','PYTHONHASHSEED':'42','HF_HUB_OFFLINE':'1'}

    def run(stage,script,arguments,python=sys.executable):
        command=[python,'-u',f'scripts/{script}.py',*map(str,arguments)]
        path=out/stage;path.mkdir(parents=True,exist_ok=True)
        log=path/f'run-{len(records):02d}-{script}.log'
        print('RUN',command,flush=True);start=time.time()
        with log.open('w') as stream:
            result=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
        records.append(dict(stage=stage,command=command,log=str(log),exit_code=result.returncode,
                            elapsed_seconds=time.time()-start,finished_utc=datetime.now(timezone.utc).isoformat()))
        record.write_text(json.dumps(records,indent=2))
        if result.returncode:raise RuntimeError(f'Failed: {log}')

    stages=['neural','retrievers','negative','semantic','business','docker'] if args.stage=='all' else [args.stage]
    for stage in stages:
        if stage=='neural':
            cfg=spec['neural_rankers']
            for name in cfg['models']:
                destination=Path('models/kuairec_completion')/name
                if args.resume and (destination/'config.json').exists():
                    existing=json.loads((destination/'config.json').read_text())
                    assert existing['exclude_injected_positives'] and existing['dense_transform']=='log1p'
                    assert existing['epochs_requested']==cfg['epochs']
                    print('REUSE',destination,flush=True)
                    continue
                run(stage,'train_ranker',['--dataset','kuairec_strict','--model',name,'--features_dir',spec['features'],
                    '--output_dir',destination,'--epochs',cfg['epochs'],'--patience',cfg['patience'],
                    '--lr',cfg['learning_rate'],'--batch_size',cfg['batch_size'],'--embedding_dim',cfg['embedding_dim'],
                    '--max_history_len',cfg['max_history_len'],'--max_train_rows',cfg['max_train_rows'],
                    '--max_val_rows',cfg['max_val_rows'],'--sample_seed',spec['seed'],
                    '--tensor_cache','--exclude_injected_positives','--dense_transform',cfg['dense_transform']])
            run(stage,'evaluate_completion',['--stage','neural'])
            run(stage,'verify_neural_serving',[])
            run(stage,'verify_neural_api',[])
        elif stage=='retrievers':
            cfg=spec['neural_retrievers']
            for name in cfg['models']:
                dest=Path('models/kuairec_completion/retrievers')/name
                if args.resume and (dest/'config.json').exists():
                    existing=json.loads((dest/'config.json').read_text())
                    assert existing['init_kwargs']['epochs']==cfg['epochs']
                    continue
                run(stage,'train_retriever',['--dataset','kuairec_strict','--processed_dir','data/processed/kuairec_strict',
                    '--output_dir','models/kuairec_completion/retrievers','--model',name,'--epochs',cfg['epochs'],
                    '--embedding_dim',cfg['embedding_dim'],'--batch_size',cfg['batch_size'],
                    '--lr',cfg.get('learning_rate',.01),'--max_history_len',cfg['max_history_len'],'--device','cpu'])
            run(stage,'verify_retrievers',['--models',','.join(cfg['models']),'--epochs',cfg['epochs'],
                '--model-dir','models/kuairec_completion/retrievers','--output-dir',out/'retrievers','--load-only'])
        elif stage=='negative':
            cfg=spec['negative_sampling']
            for name in cfg['strategies']:
                features=Path('data/features/kuairec_completion_negatives')/name
                model=Path('models/kuairec_completion/negatives')/name/'lambdarank'
                if args.resume and (model/'config.json').exists():continue
                run(stage,'build_exposure_ranker_train',['--dataset','kuairec_strict','--input_dir',spec['features'],
                    '--output_dir',features,'--processed_dir','data/processed/kuairec_strict','--report_dir',out/'negative'/name,
                    '--strategy',name,'--negatives_per_positive',cfg['negatives_per_positive'],'--seed',spec['seed']])
                run(stage,'train_lambdarank_ranker',['--dataset','kuairec_strict','--features_dir',features,
                    '--output_dir',model,'--n_estimators',cfg['n_estimators'],'--max_train_rows',700000,
                    '--max_val_rows',500000,'--random_state',spec['seed']])
            run(stage,'evaluate_completion',['--stage','negative'])
        elif stage=='semantic':
            run(stage,'verify_semantic_retrieval',[],python=args.semantic_python)
        elif stage=='business':
            run(stage,'evaluate_completion',['--stage','business'])
        elif stage=='docker':
            runtime=shutil.which('docker')
            status=dict(status='blocked',reason='Docker executable/engine unavailable; rootless user namespaces are denied.')
            if runtime:
                result=subprocess.run([runtime,'info'],capture_output=True,text=True,timeout=30)
                if result.returncode==0:
                    log=out/'docker-build.log'
                    with log.open('w') as stream:
                        result=subprocess.run([runtime,'build','-t','llm4rec:acceptance','.'],stdout=stream,stderr=subprocess.STDOUT)
                    if result.returncode:raise RuntimeError(f'Docker build failed: {log}')
                    result=subprocess.run([runtime,'run','--rm','-e','OMP_NUM_THREADS=2','llm4rec:acceptance',
                                           'python','-m','unittest','discover','-s','tests'],capture_output=True,text=True)
                    (out/'docker-tests.log').write_text(result.stdout+result.stderr)
                    if result.returncode:raise RuntimeError('Container tests failed')
                    status=dict(status='passed',image='llm4rec:acceptance',unit_tests=True)
                else:status['reason']=result.stderr.strip()[:1000]
            (out/'docker_status.json').write_text(json.dumps(status,indent=2))
    if args.stage=='all':
        run('report','report_completion',[])
    print('Completion stage finished:',args.stage,flush=True)


if __name__=='__main__':main()
