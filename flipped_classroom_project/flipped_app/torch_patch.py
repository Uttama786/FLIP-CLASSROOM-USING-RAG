"""
Workaround for PyTorch compatibility bugs with newer Hugging Face transformers.
Specifically fixes:
1. AttributeError: module 'torch' has no attribute 'float8_e8m0fnu'
2. AttributeError: module 'torch' has no attribute 'accelerator'
   (transformers >= 4.49 accesses torch.accelerator which only exists in PyTorch >= 2.6.0)
"""


def patch_torch_compat():
    try:
        import torch

        # 1. Patch missing float8_e8m0fnu
        if not hasattr(torch, "float8_e8m0fnu"):
            setattr(torch, "float8_e8m0fnu", torch.float32)

        # 2. Patch missing torch.accelerator namespace
        if not hasattr(torch, "accelerator"):
            class _DummyAccelerator:
                @staticmethod
                def current_accelerator():
                    return None

                @staticmethod
                def is_available():
                    return False

                @staticmethod
                def device_count():
                    return 0

                @staticmethod
                def is_initialized():
                    return False

                @staticmethod
                def get_device_name(device=None):
                    return ""

                @staticmethod
                def set_device_index(idx):
                    pass

            setattr(torch, "accelerator", _DummyAccelerator)
    except ImportError:
        pass


# Run immediately on module import
patch_torch_compat()
