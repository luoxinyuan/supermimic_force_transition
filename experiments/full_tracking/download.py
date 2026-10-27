"""Download the pinned full G1 dataset and verify SHA-256; safely rerunnable."""
import hashlib
import json
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

ROOT=Path(__file__).resolve().parents[2]
DEST=ROOT/'data/raw/bones_seed'
REVISION='2f59b2077b9da34dd4e43618e705c7cb962c9a66'
REPO='bones-studio/seed'
FILES=['g1.tar.gz','metadata/seed_metadata_v004.parquet',
       'metadata/seed_metadata_v004.csv','metadata/seed_metadata_v002_temporal_labels.jsonl',
       'README.md','LICENSE.md']

def main():
    DEST.mkdir(parents=True,exist_ok=True)
    info=HfApi().dataset_info(REPO,revision=REVISION,files_metadata=True)
    byname={f.rfilename:f for f in info.siblings}
    manifest={'repo':REPO,'revision':REVISION,'verified':False,'files':[]}
    status=ROOT/'reports/full_tracking/download.json'
    for name in FILES:
        f=byname[name]
        item={'path':name,'size':f.size,'expected_sha256':f.lfs.sha256 if f.lfs else None,'verified':False}
        manifest['files'].append(item)
    status.write_text(json.dumps(manifest,indent=2)+'\n')
    for item in manifest['files']:
        print('Downloading:',item['path'],flush=True)
        path=Path(hf_hub_download(REPO,item['path'],repo_type='dataset',revision=REVISION,local_dir=DEST))
        if path.stat().st_size!=item['size']:raise RuntimeError('Size mismatch: '+str(path))
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
        item['sha256']=digest.hexdigest()
        if item['expected_sha256'] and item['sha256']!=item['expected_sha256']:raise RuntimeError('SHA256 mismatch: '+str(path))
        item['verified']=True
        status.write_text(json.dumps(manifest,indent=2)+'\n')
        print('Verified:',item['path'],item['sha256'],flush=True)
    manifest['verified']=True
    status.write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':main()
