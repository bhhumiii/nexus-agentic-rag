import logging
import os
import sys
from functools import lru_cache
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

log = logging.getLogger("rag_store")

BASE = Path(__file__).resolve().parent

CHROMA_PATH = Path(
    os.getenv("CHROMA_PATH", str(BASE / "chroma_db"))
)

if not CHROMA_PATH.is_absolute():
    CHROMA_PATH = BASE / CHROMA_PATH

COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "company_knowledge")

# ChromaDB 1.x (Rust core with C++ hnswlib) and onnxruntime are built with
# Visual Studio 2022 17.10+. With an older msvcp140.dll they crash the whole
# process with an access violation, which Python cannot catch.
MIN_MSVCP140 = (14, 40)


def _msvcp140_version() -> tuple[str, tuple[int, int]] | None:
    """Return (path, (major, minor)) of the msvcp140.dll this process uses."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetModuleHandleW.restype = wintypes.HMODULE
    kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
    kernel32.GetModuleFileNameW.argtypes = [wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
    handle = kernel32.GetModuleHandleW("msvcp140.dll")
    if handle:
        buf = ctypes.create_unicode_buffer(1024)
        kernel32.GetModuleFileNameW(handle, buf, 1024)
        path = buf.value
    else:  # Not loaded yet: Windows will load the System32 copy.
        path = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"),
                            "System32", "msvcp140.dll")

    version = ctypes.WinDLL("version")
    version.GetFileVersionInfoSizeW.argtypes = [wintypes.LPCWSTR, wintypes.LPVOID]
    version.GetFileVersionInfoW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD,
                                            wintypes.DWORD, wintypes.LPVOID]
    version.VerQueryValueW.argtypes = [wintypes.LPCVOID, wintypes.LPCWSTR,
                                       ctypes.POINTER(ctypes.c_void_p),
                                       ctypes.POINTER(wintypes.UINT)]
    size = version.GetFileVersionInfoSizeW(path, None)
    if not size:
        return None
    data = ctypes.create_string_buffer(size)
    if not version.GetFileVersionInfoW(path, 0, size, data):
        return None
    info, length = ctypes.c_void_p(), wintypes.UINT()
    if not version.VerQueryValueW(data, "\\", ctypes.byref(info), ctypes.byref(length)):
        return None
    # VS_FIXEDFILEINFO: signature, struct version, FileVersionMS, FileVersionLS, ...
    file_version_ms = ctypes.cast(info, ctypes.POINTER(wintypes.DWORD * 4)).contents[2]
    return path, (file_version_ms >> 16, file_version_ms & 0xFFFF)


def check_native_runtime() -> None:
    """Raise a clear error instead of letting ChromaDB crash the process."""
    if sys.platform != "win32" or os.getenv("SKIP_VC_RUNTIME_CHECK") == "1":
        return
    try:
        found = _msvcp140_version()
    except OSError as exc:
        log.warning("Could not check the Visual C++ runtime version: %s", exc)
        return
    if found and found[1] < MIN_MSVCP140:
        path, (major, minor) = found
        raise RuntimeError(
            f"Microsoft Visual C++ runtime is too old ({path} is {major}.{minor}; "
            f"ChromaDB needs {MIN_MSVCP140[0]}.{MIN_MSVCP140[1]} or newer). Install the latest "
            "'Microsoft Visual C++ Redistributable for Visual Studio 2015-2022 (x64)' from "
            "https://aka.ms/vs/17/release/vc_redist.x64.exe, then restart the backend."
        )


@lru_cache(maxsize=1)
def get_model():
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cpu"
    )


@lru_cache(maxsize=1)
def get_collection():
    check_native_runtime()
    log.info("Opening ChromaDB at %s (collection %s)", CHROMA_PATH, COLLECTION_NAME)
    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH)
    )

    # Prevent ChromaDB from initializing its default ONNX embedder.
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=None
    )


def search_knowledge(query: str, top_k: int = 4) -> list[dict]:
    query = query.strip()

    if not query:
        raise ValueError("Query cannot be empty")

    top_k = max(1, min(int(top_k), 8))

    col = get_collection()

    count = col.count()
    if count == 0:
        return []

    vector = get_model().encode(
        query,
        normalize_embeddings=True
    ).tolist()

    result = col.query(
        query_embeddings=[vector],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"]
    )

    return [
        {
            "text": doc,
            "source": (result["metadatas"][0][i] or {}).get(
                "source", "unknown"
            ),
            "distance": float(result["distances"][0][i])
        }
        for i, doc in enumerate(result["documents"][0])
    ]
