"""Seven-file CPU install/restore/tamper gates; never touches the real package."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile

ROOT=Path(__file__).resolve().parents[1]


def main(installer=None):
    source=Path(__file__).with_name('install_overlay.py')
    spec=importlib.util.spec_from_file_location('iluvatar_overlay',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if installer is None:installer=module
    new_file='patches/iluvatar_index1d.py'
    checks=[]
    six=installer.BASE_FILES+(installer.PREFILL_FILE,installer.DECODE_FILE,installer.CONFIG_FILE)
    seven=six+(new_file,)
    for schema in [installer.BASE_FILES,installer.BASE_FILES+(installer.PREFILL_FILE,),
                   installer.BASE_FILES+(installer.PREFILL_FILE,installer.GEMM_FILE),
                   installer.BASE_FILES+(installer.PREFILL_FILE,installer.DECODE_FILE),
                   installer.BASE_FILES+(installer.PREFILL_FILE,installer.CONFIG_FILE),six]:
        assert installer.selected_files(schema)==schema
    checks.append('all_six_legacy_schema_variants_preserved')
    for tamper in (False,True):
        with tempfile.TemporaryDirectory(prefix='eggking-index7-cpu-') as temp:
            root=Path(temp);src,pkg=root/'source',root/'package'
            candidate={name:('candidate '+name).encode() for name in seven}
            base={name:('accepted '+name).encode() for name in seven[:4]}
            for name,data in candidate.items():
                p=src/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
            for name,data in base.items():
                p=pkg/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
            manifest=dict(candidate_commit='CPU-fixture-not-a-candidate',
                runtime_file_hashes={n:hashlib.sha256(v).hexdigest() for n,v in candidate.items()},
                previous_runtime_file_hashes={n:hashlib.sha256(v).hexdigest() for n,v in base.items()},
                base_plugin_init_sha256=hashlib.sha256(base['__init__.py']).hexdigest())
            result=installer.install(src,pkg,manifest,root/'backups')
            assert all((pkg/n).read_bytes()==data for n,data in candidate.items())
            if tamper:
                p=pkg/new_file;p.write_bytes(b'unknown replacement')
                before={str(p.relative_to(pkg)):p.read_bytes() for p in pkg.rglob('*.py')}
                try:installer.restore(Path(result['backup']),pkg)
                except RuntimeError:pass
                else:raise AssertionError('Unknown bytes overwritten')
                assert {str(p.relative_to(pkg)):p.read_bytes() for p in pkg.rglob('*.py')}==before
                checks.append('unknown_seventh_module_refuses_entire_restore_before_any_mutation')
            else:
                installer.restore(Path(result['backup']),pkg)
                assert all((pkg/n).read_bytes()==data for n,data in base.items())
                assert all(not (pkg/n).exists() for n in seven[4:])
                checks.append('seven_file_install_and_exact_four_file_restore_remove_owned_extras')
    try:installer.selected_files(seven+('patches/unknown.py',))
    except RuntimeError:pass
    else:raise AssertionError('Unknown file schema allowed')
    checks.append('unknown_additional_file_schema_refused')
    return dict(checks=checks,passed=len(checks),validation_errors=[],new_gpu_work=False,
        real_package_modified=False,installer_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        all_seven_file_runners_ready=False)


if __name__=='__main__':print(json.dumps(main(),indent=2))
