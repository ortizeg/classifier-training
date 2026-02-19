"""Hydra ConfigStore registration utilities."""

from __future__ import annotations

from typing import Any

from hydra.core.config_store import ConfigStore
from loguru import logger


def register(
    cls: type[Any] | None = None,
    *,
    group: str | None = None,
    name: str | None = None,
    **kwargs: Any,
) -> type[Any] | Any:
    """Decorator to register a class with Hydra's ConfigStore.

    Automatically creates a configuration entry with the correct ``_target_``
    pointing to the decorated class and registers it in the ConfigStore.

    If *group* is not provided, it tries to infer it from the module path
    by taking the second-to-last element of the module path.

    Arguments:
        cls: The class to register.
        group: The ConfigStore group. If ``None``, inference is attempted.
        name: The name for the config. Defaults to class name.
        **kwargs: Default values for the configuration node.
    """

    def _process_class(target_cls: type[Any]) -> type[Any]:
        nonlocal group, name

        # Determine the target path (module + class name)
        target_path = f"{target_cls.__module__}.{target_cls.__name__}"

        # Use provided name or class name as the config name
        config_name = name or target_cls.__name__

        # Infer group if not provided
        if group is None:
            module_parts = target_cls.__module__.split(".")
            group = module_parts[-2]

        # Register in ConfigStore
        cs = ConfigStore.instance()
        logger.debug(
            f"Registering {target_cls.__name__} as '{config_name}' in group '{group}'"
        )
        # Build the configuration node
        node = {"_target_": target_path}
        node.update(kwargs)
        cs.store(group=group, name=config_name, node=node)

        return target_cls

    if cls is None:
        return _process_class
    return _process_class(cls)
