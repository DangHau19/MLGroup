"""Refresh hashes in the DRAFT final manifest after receiving handovers. Never approves or scores test."""
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src import config


def main():
    file=ROOT/'docs/evaluation/final_manifest.json'
    manifest=json.loads(file.read_text(encoding='utf-8-sig'))
    if manifest.get('ready_for_final_test') is True:
        raise SystemExit('Refusing to modify a ready/frozen manifest. Review the handover first.')
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    for rel in manifest['source_sha256']:
        manifest['source_sha256'][rel]=sha(ROOT/rel)
    for model in manifest['models']:
        model['sha256']=sha(ROOT/model['path'])
        model['cv_evidence_sha256']=sha(ROOT/model['cv_evidence'])
        decision=model['decision']
        decision['evidence_sha256']=sha(ROOT/decision['evidence'])
        # Training provenance must be confirmed by the model owner; hashing does not establish it.
    file.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Draft hashes refreshed. Readiness, decision rules and training confirmations unchanged. No test scored.')


if __name__=='__main__':main()
