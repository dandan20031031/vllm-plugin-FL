"""Install the manifest's tested Python overlay without rebuilding extensions.

Default mode verifies and prints a plan. Use --apply in the matching official
Linux image after stopping the model service. This tool never starts GPU work.
"""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import tempfile

BASE_FILES = ('__init__.py', 'patches/iluvatar_sampler.py', 'patches/iluvatar_attention.py')
PREFILL_FILE = 'patches/iluvatar_prefill.py'
GEMM_FILE = 'patches/iluvatar_gemm.py'
DECODE_FILE = 'patches/iluvatar_decode_pages.py'
CONFIG_FILE = 'patches/iluvatar_config_cache.py'


def selected_files(hashes):
    for files in (BASE_FILES, BASE_FILES + (PREFILL_FILE,), BASE_FILES + (PREFILL_FILE, GEMM_FILE),
                  BASE_FILES + (PREFILL_FILE, DECODE_FILE),
                  BASE_FILES + (PREFILL_FILE, CONFIG_FILE),
                  BASE_FILES + (PREFILL_FILE, DECODE_FILE, CONFIG_FILE),
                  BASE_FILES + (PREFILL_FILE, DECODE_FILE, CONFIG_FILE, 'patches/iluvatar_index1d.py')):
        if set(hashes) == set(files):
            return files
    raise RuntimeError('Unexpected source file list')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def python_fingerprint(root, excludes=()):
    values = {path.relative_to(root).as_posix(): digest(path)
              for path in root.rglob('*.py')
              if path.relative_to(root).as_posix() not in excludes and path.name != '_version.py'}
    return {'python_files': len(values),
            'sha256': hashlib.sha256(json.dumps(values, sort_keys=True, separators=(',', ':')).encode()).hexdigest()}


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = path.stat().st_mode & 0o777 if path.exists() else 0o644
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.eggking-', delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
    try:
        temporary.chmod(mode)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def plan(source, package, manifest):
    hashes = manifest['runtime_file_hashes']
    files = selected_files(hashes)
    items = []
    for relative in files:
        src, dst = source / relative, package / relative
        if digest(src) != hashes[relative]:
            raise RuntimeError('Source differs from tested candidate: ' + relative)
        current = digest(dst) if dst.exists() else None
        allowed = {hashes[relative], None}
        if relative == '__init__.py':
            allowed = {hashes[relative], manifest['base_plugin_init_sha256']}
        previous = manifest.get('previous_runtime_file_hashes', {}).get(relative)
        if previous is not None:
            allowed.add(previous)
        if current not in allowed:
            raise RuntimeError('Installed file is neither the expected base nor candidate: ' + relative)
        items.append({'relative': relative, 'source_sha256': hashes[relative],
                      'original_sha256': current, 'change_required': current != hashes[relative]})
    return items


def restore(backup, package):
    receipt = json.loads((backup / 'backup.json').read_text())
    if Path(receipt['package']).resolve() != package.resolve():
        raise RuntimeError('Backup belongs to a different installed package')
    items = receipt['files']
    files = selected_files([item['relative'] for item in items])
    if [item['relative'] for item in items] != list(files):
        raise RuntimeError('Unexpected backup file list')
    for item in items:
        if not item['change_required']:
            continue
        dst = package / item['relative']
        if not dst.exists() or digest(dst) != item['source_sha256']:
            raise RuntimeError('Runtime changed after installation; refusing to overwrite: ' + str(dst))
        if item['original_sha256'] is not None:
            original = backup / item['relative'].replace('/', '__')
            if digest(original) != item['original_sha256']:
                raise RuntimeError('Backup digest mismatch: ' + str(original))
    for item in items:
        if not item['change_required']:
            continue
        dst = package / item['relative']
        if item['original_sha256'] is None:
            dst.unlink()
        else:
            atomic_write(dst, (backup / item['relative'].replace('/', '__')).read_bytes())
    return {'restored': True, 'package': str(package), 'backup': str(backup)}


def install(source, package, manifest, backup_root):
    items = plan(source, package, manifest)
    if not any(item['change_required'] for item in items):
        return {'already_installed': True, 'package': str(package)}
    backup_root.mkdir(parents=True, exist_ok=True)
    backup = Path(tempfile.mkdtemp(prefix='overlay-', dir=backup_root))
    receipt = {'package': str(package.resolve()), 'created_at_utc': datetime.now(timezone.utc).isoformat(),
               'candidate_commit': manifest['candidate_commit'], 'files': items}
    for item in items:
        dst = package / item['relative']
        if dst.exists():
            (backup / item['relative'].replace('/', '__')).write_bytes(dst.read_bytes())
    (backup / 'backup.json').write_text(json.dumps(receipt, indent=2))
    changed = []
    try:
        for item in items:
            if item['change_required']:
                dst = package / item['relative']
                atomic_write(dst, (source / item['relative']).read_bytes())
                changed.append(item)
                if digest(dst) != item['source_sha256']:
                    raise RuntimeError('Post-install digest mismatch: ' + str(dst))
    except BaseException:
        for item in reversed(changed):
            dst = package / item['relative']
            if item['original_sha256'] is None:
                if dst.exists():
                    dst.unlink()
            else:
                atomic_write(dst, (backup / item['relative'].replace('/', '__')).read_bytes())
        raise
    return {'installed': True, 'package': str(package), 'backup': str(backup), 'files': items}


def no_active_server():
    processes = subprocess.check_output(['ps', '-eo', 'args'], text=True)
    if any('vllm serve ' in line or 'VLLM::EngineCore' in line for line in processes.splitlines()):
        raise RuntimeError('Stop the vLLM service before changing runtime files')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--apply', action='store_true')
    mode.add_argument('--rollback', type=Path)
    parser.add_argument('--backup-root', type=Path, default=Path.cwd() / 'eggking-overlay-backups')
    args = parser.parse_args()
    manifest = json.loads((args.root / 'docs/iluvatar-minicpm/manifest.json').read_text())
    package = Path(metadata.distribution('vllm-plugin-fl').locate_file('vllm_fl')).resolve()
    if args.rollback:
        no_active_server()
        print(json.dumps(restore(args.rollback.resolve(), package), indent=2))
        return
    if os.name != 'posix':
        raise RuntimeError('Apply/verify this overlay inside the official Linux image')
    for name, expected in manifest['packages'].items():
        if metadata.version(name) != expected:
            raise RuntimeError('Package version differs from the tested image: ' + name)
    vllm_root = Path(metadata.distribution('vllm').locate_file('vllm'))
    for relative, expected in manifest['native_vllm_file_hashes'].items():
        if digest(vllm_root / relative) != expected:
            raise RuntimeError('Vendor attention code differs from the tested image: ' + relative)
    excludes = manifest['plugin_fingerprint_excludes']
    if python_fingerprint(package, excludes) != manifest['plugin_base_python_fingerprint']:
        raise RuntimeError('Installed plugin base Python code differs from the tested image')
    gems = Path(metadata.distribution('flag_gems').locate_file('flag_gems'))
    if python_fingerprint(gems) != manifest['flaggems_python_fingerprint']:
        raise RuntimeError('FlagGems code differs despite its version label; install the bundled exact source first')
    # The fork supplies only the seven-file overlay; installed base and
    # FlagGems fingerprints above still guard the tested environment.
    source = args.root / 'vllm_fl'
    items = plan(source, package, manifest)
    if args.apply:
        no_active_server()
        result = install(source, package, manifest, args.backup_root.resolve())
    else:
        result = {'dry_run': True, 'package': str(package), 'files': items,
                  'note': 'No runtime files changed and no GPU work started'}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
