# MoriTransferEngineConnector

## When to Use

Currently supports intra-node deployment with Mori.

Inter-node support will come in a future refactor.

## Mechanism

Uses Mori's `IOEngine` / `MemoryDesc` API for zero-copy RDMA transfers.

- Data Plane: RDMA (InfiniBand/RoCE) with managed memory pool.
- Control Plane: ZMQ for pull-request handshake and async completion.

## Installation

See the [Mori repository](https://github.com/ROCm/mori) for installation instructions.

## Configuration

Define the connector in runtime:

```yaml
runtime:
  connectors:
    mori_connector:
      name: MoriTransferEngineConnector
      extra:
        host: "auto"
        zmq_port: 50051
        device_name: ""
        memory_pool_size: 536870912
        memory_pool_device: "cpu"
```

Wire stages to the connector:

```yaml
stage_args:
  - stage_id: 0
    output_connectors:
      to_stage_1: mori_connector

  - stage_id: 1
    input_connectors:
      from_stage_0: mori_connector
```

Parameters:

- host: local RDMA IP (`"auto"` for auto-detect).
- zmq_port: ZMQ base port for control-plane communication.
- device_name: RDMA device (e.g., `"mlx5_0"`), empty for auto-detect.
- memory_pool_size: RDMA memory pool size in bytes.
- memory_pool_device: `"cpu"` (pinned) or `"cuda"` (GPUDirect RDMA).

For more details, refer to the
[Mori repository](https://github.com/ROCm/mori).
