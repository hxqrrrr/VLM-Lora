import importlib.metadata
import importlib.util
import logging
from typing import Optional, Tuple, Union

import torch
from packaging import version


def copy_parameters(source: torch.nn.Module, dest: torch.nn.Module):
    # 获取目标模块的设备
    device = next(dest.parameters()).device if list(dest.parameters()) else torch.device('cpu')
    # 将源模块的状态字典复制到目标设备上
    state_dict = {k: v.to(device) for k, v in source.state_dict().items()}
    dest.load_state_dict(state_dict)
    dest.requires_grad_(False)


def setup_logging(log_level: str = "WARN", log_file: str = None):
    # set the logger
    log_handlers = [logging.StreamHandler()]
    if log_file is not None:
        log_handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        format="[%(asctime)s] MoE-PEFT: %(message)s",
        level=log_level,
        handlers=log_handlers,
        force=True,
    )


def is_package_available(
    pkg_name: str, pkg_version: Optional[str] = None
) -> Union[Tuple[bool, str], bool]:
    # Check we're not importing a "pkg_name" directory somewhere but the actual library by trying to grab the version
    package_exists = importlib.util.find_spec(pkg_name) is not None
    package_version = "N/A"
    if package_exists:
        try:
            package_version = importlib.metadata.version(pkg_name)
            package_exists = True
        except importlib.metadata.PackageNotFoundError:
            package_exists = False
        logging.debug(f"Detected {pkg_name} version {package_version}")
    if pkg_version is not None:
        return package_exists and version.parse(package_version) >= version.parse(
            pkg_version
        )
    else:
        return package_exists


class Unsubscribable:
    def __init__(self) -> None:
        raise RuntimeError(f"Instant unsubscribable class {__class__}")


# Class Placeholder for Bitsandbytes
class Linear8bitLt(Unsubscribable):
    def __init__(self) -> None:
        super().__init__()


class Linear4bit(Unsubscribable):
    def __init__(self) -> None:
        super().__init__()


class BitsAndBytesConfig:
    def __init__(self, **kwargs) -> None:
        raise RuntimeError("Quantization not supported.")


class NoneContexts(object):
    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass
