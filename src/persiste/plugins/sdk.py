"""Plugin development kit - decorators and utilities."""

from typing import Callable, Any
from functools import wraps


def register_state_space(name: str) -> Callable:
    """
    Decorator to register a state space factory in a plugin.
    
    Args:
        name: State space identifier
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        wrapper._persiste_state_space = name
        return wrapper
    
    return decorator


def register_baseline(name: str) -> Callable:
    """
    Decorator to register a baseline process factory in a plugin.
    
    Args:
        name: Baseline identifier
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        wrapper._persiste_baseline = name
        return wrapper
    
    return decorator


def register_analysis(name: str) -> Callable:
    """
    Decorator to register an analysis workflow in a plugin.
    
    Args:
        name: Analysis identifier
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        wrapper._persiste_analysis = name
        return wrapper
    
    return decorator
