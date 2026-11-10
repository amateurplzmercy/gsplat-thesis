"""Check GPU computation and prefetch primitives; does not measure real overlap."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('reports/cuda_primitives.json'))
    args = parser.parse_args()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available')
    with torch.inference_mode():
        source = torch.empty(1024 * 1024, dtype=torch.float32, pin_memory=True)
        source.copy_(torch.linspace(0, 1, source.numel()))
        destination = torch.empty_like(source, device='cuda')
        torch.cuda.synchronize()
        transfer_stream = torch.cuda.Stream()
        ready = torch.cuda.Event()
        with torch.cuda.stream(transfer_stream):
            destination.copy_(source, non_blocking=True)
            ready.record()
        torch.cuda.current_stream().wait_event(ready)
        observed = destination.square().sum().item()
        expected = source.square().sum().item()
        error = abs(observed - expected) / max(abs(expected), 1)
        if error > 1e-5:
            raise RuntimeError(f'Transfer/event correctness failed: relative error {error}')
        matrix = torch.arange(16, device='cuda', dtype=torch.float32).reshape(4, 4)
        product = matrix @ torch.eye(4, device='cuda')
        if not torch.equal(matrix, product):
            raise RuntimeError('CUDA matrix operation failed')
        report = {
            'status': 'passed',
            'gpu': torch.cuda.get_device_name(0),
            'compute_capability': list(torch.cuda.get_device_capability(0)),
            'torch': torch.__version__,
            'cuda_runtime': torch.version.cuda,
            'pinned_host_memory': source.is_pinned(),
            'nonblocking_copy_with_event_dependency': True,
            'transfer_bytes': source.numel() * source.element_size(),
            'relative_error': error,
            'gpu_matrix_operation': True,
            'transfer_compute_overlap_measured': False,
            'note': 'Primitive correctness only; no prefetch speedup claim.'
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
