"""Device policy; simulation never calls or impersonates torch.cuda."""


class BackendUnavailable(RuntimeError):
    pass


class Backend:
    def __init__(self, requested, torch):
        self.torch = torch
        self.requested = requested
        self.simulated = requested == "cuda-sim"
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        if self.simulated:
            requested = "cpu"
        if requested.startswith("cuda"):
            if not torch.cuda.is_available():
                raise BackendUnavailable("Real CUDA requested, but PyTorch reports CUDA unavailable. Install a matching CUDA-enabled PyTorch and NVIDIA driver on a GPU host, or select cpu/cuda-sim.")
            index = int(requested.partition(":")[2] or 0)
            if index >= torch.cuda.device_count():
                raise BackendUnavailable(f"CUDA device ordinal {index} is unavailable")
            requested = f"cuda:{index}"
            torch.cuda.set_device(index)
        self.device = torch.device(requested)

    def synchronize(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def describe(self):
        return dict(requested=self.requested, actual_device=str(self.device), simulated=self.simulated,
                    execution_kind="interface_simulation" if self.simulated else "hardware",
                    cuda_hardware_available=self.torch.cuda.is_available(),
                    timing_enabled=not self.simulated,
                    simulation_scope="CPU execution of device-independent interface/correctness cases; no CUDA kernels, device/stream emulator or GPU performance prediction" if self.simulated else None)
