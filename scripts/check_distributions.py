"""Check actual release archives without importing System1 or optional libraries."""
import argparse
from pathlib import Path
import tarfile
import zipfile


def main(directory):
    wheels = list(directory.glob('*.whl'))
    sources = list(directory.glob('*.tar.gz'))
    assert len(wheels) == len(sources) == 1, 'Expected one wheel and one source archive'
    wheel, source = wheels[0], sources[0]
    # These archives contain code/examples, not the full research evidence corpus.
    assert wheel.stat().st_size < 2 * 1024**2, 'Wheel unexpectedly exceeds 2 MiB'
    assert source.stat().st_size < 25 * 1024**2, 'Source archive unexpectedly exceeds 25 MiB'
    with zipfile.ZipFile(wheel) as archive:
        wheel_names = set(archive.namelist())
    assert {'system1/compiler.py', 'system1/proto/system1.proto', 'reflex/__init__.py',
            'reflex/proto/system1.proto'} <= wheel_names
    assert not any(name.startswith(('benchmarks/', 'examples/')) for name in wheel_names)
    with tarfile.open(source, 'r:gz') as archive:
        source_names = {member.name.split('/', 1)[1] for member in archive.getmembers() if '/' in member.name}
    assert {'src/system1/compiler.py', 'examples/teaching_by_doing/app.py',
            'examples/teaching_by_doing/index.html', 'examples/teaching_by_doing/documents.json',
            'benchmarks/quality/n8n_gauntlet/prepare.py',
            'benchmarks/quality/n8n_gauntlet/boundary-teaching-requests.json'} <= source_names
    assert not any(name.startswith('benchmarks/quality/n8n_gauntlet/artifacts/') or
                   (name.startswith('benchmarks/quality/n8n_gauntlet/results/') and name.endswith(('.json', '.png')))
                   for name in source_names), 'Research payload leaked into the source archive'
    print(f'Checked wheel ({wheel.stat().st_size:,} bytes) and source archive ({source.stat().st_size:,} bytes).')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    main(parser.parse_args().directory)
