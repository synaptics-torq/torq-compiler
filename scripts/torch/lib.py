from os import environ
from pathlib import Path
from subprocess import run

from iree.turbine import aot
from torch import arange, int16, bfloat16, logical_not, isnan, no_grad


def test_and_save(model, reference, bittol=0):

    # test consistency againtst reference
    inp = arange(2**16).to(int16).view(bfloat16).reshape(2**13, 2**3)
    computed, real = model(inp), reference(inp)
    safe = logical_not(isnan(real))
    assert all(isnan(computed[~safe]))
    diff = computed[safe].view(int16) - real[safe].view(int16)
    assert all(diff.abs() <= bittol)

    # obtain sane base filename for outputs
    name = model.__class__.__name__

    # generate torch mlir
    print('exporting to', f"{name}.mlir")
    aot.export(model, inp).save_mlir(f"{name}.mlir")

    # generate linalg mlir from torch mlir and compile llvm-cpu vmfb
    project = Path(__file__).parent.parent.parent.parent
    build_dir = environ.get('IREE_BUILD_DIR', str(project / 'build-latest'))
    for target, args in (('vmfb', ('--iree-hal-target-backends=llvm-cpu',
                                   '--iree-llvmcpu-target-cpu=generic')),
                         ('linalg.mlir', ('--compile-to=input',))):
        full_args = [f'{build_dir}/tools/torq-compile', f'{name}.mlir',
                     '-o', f'{name}.{target}',
                     #'--mlir-print-ir-after-all',
                     *args]
        with open(f'{name}.{target}.log', 'w') as log:
            print(*full_args, '>', log.name)
            run(full_args, stderr=log)
