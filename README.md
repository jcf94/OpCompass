# 🧭 OpCompass

**SOL (Speed of Light) theoretical peak performance estimator for GPU operators.**

Given an operator (matmul, convolution, attention, …), concrete input shapes, and a target hardware spec, OpCompass estimates the theoretical lower-bound execution time by analyzing data movement, compute, and pipeline constraints.

---

## Quick Start

```bash
# Install
cd opcompass
pip install -e .

# List available operators & hardware
compass list operators
compass list hardware

# Analyze matmul 4096³ FP16 on A100
compass analyze matmul --hardware a100 --dtype fp16 --M 4096 --N 4096 --K 4096

# Sweep over shapes
compass sweep matmul --hardware a100,h100 --M 1024,2048,4096 --K 1024,2048,4096
```

## Web UI

```bash
uvicorn opcompass.server:app --reload
# Open http://127.0.0.1:8000
```

## Project Structure

```
opcompass/
├── opcompass/                # Core Python package
│   ├── models.py             # Shared data models
│   ├── registry.py           # Auto-discovery for operators & hardware
│   ├── operators/            # One file per operator (pluggable)
│   │   ├── base.py           #   Operator abstract base class
│   │   ├── matmul.py
│   │   ├── convolution.py
│   │   ├── flash_attention.py
│   │   ├── layernorm.py
│   │   ├── elementwise.py
│   │   └── reduction.py
│   ├── hardware/             # One file per hardware target (pluggable)
│   │   ├── base.py           #   Hardware abstract base class
│   │   ├── nvidia_a100.py
│   │   └── nvidia_h100.py
│   ├── engine/               # Analysis engine
│   │   ├── analyzer.py       #   Main SOL analysis orchestrator
│   │   ├── memory_model.py   #   Multi-tier memory hierarchy model
│   │   ├── compute_model.py  #   Peak compute throughput model
│   │   ├── pipeline_model.py #   Pipeline stage-level analysis
│   │   └── result.py         #   Result formatting
│   ├── cli.py                # CLI (click)
│   └── server.py             # FastAPI server
├── web/                      # Web frontend (vanilla JS + Chart.js)
│   ├── index.html
│   ├── css/style.css
│   └── js/{app,charts,api}.js
└── tests/
```

## Adding a New Operator

Create a file in `opcompass/operators/`, e.g. `my_op.py`:

```python
from opcompass.models import DataType
from opcompass.operators.base import Operator

class MyOp(Operator):
    name = "my_op"
    description = "My custom operator"

    @property
    def param_dims(self):
        return {"M": "batch", "N": "dim"}

    def compute_flops(self, M=0, N=0, **kwargs):
        return 2 * M * N

    def compute_io_bytes(self, dtype, M=0, N=0, **kwargs):
        bs = dtype.byte_size
        return (M * N * bs, M * N * bs)
```

It's automatically discovered — no registration needed.

## Adding a New Hardware Target

Create a file in `opcompass/hardware/`, e.g. `my_gpu.py`:

```python
from opcompass.hardware.base import Hardware
from opcompass.models import ComputeUnit, DataType, MemoryHierarchy, MemoryTier

class MyGPU(Hardware):
    name = "my_gpu"
    vendor = "Vendor"
    description = "My custom GPU"

    memory = MemoryHierarchy(
        tiers=[MemoryTier("HBM", 80e9, 3.0e12)],
        can_overlap_with_compute={"HBM"},
    )
    compute_unit = ComputeUnit(
        name="SM", count=128, clock_mhz=2000,
        peak_flops={DataType.FP16: 500e12, DataType.FP32: 62e12},
    )
```

## Analysis Modes

| Mode | Description | When to Use |
|------|-------------|-------------|
| `simple` | Pure roofline: max(FLOPs/peak, bytes/bandwidth) | Quick rough estimate |
| `hierarchy` | Multi-tier memory hierarchy (HBM → L2 → SRAM) | Moderate accuracy (default) |
| `pipeline` | Pipeline stage-level modelling | Detailed analysis (requires op breakdown) |

## License

MIT
