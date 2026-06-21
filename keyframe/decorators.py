"""D: timing / validation 데코레이터 + lru_cache 이미지 로더."""
import functools
import time
from functools import lru_cache

import numpy as np
from PIL import Image


def timing(func):
    """실행 시간 측정 후 출력."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        print(f"[timing] {func.__name__}: {time.perf_counter() - t0:.3f}s")
        return result
    return wrapper


def validate_features(func):
    """첫 번째 인자(features)가 2-D ndarray인지 검증."""
    @functools.wraps(func)
    def wrapper(features, *args, **kwargs):
        if not isinstance(features, np.ndarray) or features.ndim != 2:
            raise ValueError(
                f"{func.__name__}: features must be 2-D ndarray, "
                f"got {type(features).__name__} ndim={getattr(features, 'ndim', '?')}"
            )
        if len(features) == 0:
            raise ValueError(f"{func.__name__}: features array is empty")
        return func(features, *args, **kwargs)
    return wrapper


# 배치 모드에서 동일 경로를 중복 열지 않도록 캐싱.
# 반환 배열은 write=False로 설정 — 캐시 오염 방지.

@lru_cache(maxsize=512)
def load_mask(path: str) -> np.ndarray:
    arr = np.array(Image.open(path))
    arr.flags.writeable = False
    return arr


@lru_cache(maxsize=256)
def load_image(path: str) -> np.ndarray:
    arr = np.array(Image.open(path).convert("RGB"))
    arr.flags.writeable = False
    return arr
