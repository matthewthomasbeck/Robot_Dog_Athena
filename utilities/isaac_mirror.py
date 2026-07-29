##################################################################################
# Copyright (c) 2025 Matthew Thomas Beck                                         #
#                                                                                #
# Licensed under the Creative Commons Attribution-NonCommercial 4.0              #
# International (CC BY-NC 4.0). Personal and educational use is permitted.       #
# Commercial use by companies or for-profit entities is prohibited.              #
##################################################################################

"""LAN server: receive absolute joint targets from Isaac Lab (tai-chi / mirror)."""

from __future__ import annotations

import json
import logging
import socket
import struct
import threading
import time
from typing import Any

import utilities.config as config


_latest_lock = threading.Lock()
_latest_msg: dict[str, Any] | None = None
_latest_time: float = 0.0
_server_sock: socket.socket | None = None
_stop_event = threading.Event()


def _recv_exact(conn: socket.socket, n: int) -> bytes | None:
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def _handle_client(conn: socket.socket, addr) -> None:
    global _latest_msg, _latest_time
    logging.info(f"(isaac_mirror.py): Desktop connected from {addr}\n")
    try:
        while not _stop_event.is_set():
            length_bytes = _recv_exact(conn, 4)
            if length_bytes is None:
                logging.warning("(isaac_mirror.py): Client closed connection.\n")
                break
            length = struct.unpack(">I", length_bytes)[0]
            if length <= 0 or length > 1_000_000:
                logging.error(f"(isaac_mirror.py): Invalid frame length {length}\n")
                break
            payload = _recv_exact(conn, length)
            if payload is None:
                logging.warning("(isaac_mirror.py): Incomplete payload; client gone.\n")
                break
            try:
                msg = json.loads(payload.decode("utf-8"))
            except json.JSONDecodeError as e:
                logging.error(f"(isaac_mirror.py): Bad JSON: {e}\n")
                continue

            msg_type = msg.get("type", "joint_targets")
            if msg_type == "ping":
                continue

            with _latest_lock:
                _latest_msg = msg
                _latest_time = time.time()
    except Exception as e:
        logging.error(f"(isaac_mirror.py): Client handler error: {e}\n")
    finally:
        try:
            conn.close()
        except Exception:
            pass
        logging.info(f"(isaac_mirror.py): Desktop disconnected ({addr})\n")


def _accept_loop(server: socket.socket) -> None:
    while not _stop_event.is_set():
        try:
            server.settimeout(1.0)
            conn, addr = server.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            threading.Thread(target=_handle_client, args=(conn, addr), daemon=True).start()
        except socket.timeout:
            continue
        except Exception as e:
            if not _stop_event.is_set():
                logging.error(f"(isaac_mirror.py): Accept error: {e}\n")
            break


def start_isaac_mirror_server() -> bool:
    """Bind TCP server and start accept thread. Returns True on success."""
    global _server_sock
    cfg = config.ISAAC_MIRROR_CONFIG
    host = cfg["BIND_HOST"]
    port = int(cfg["PORT"])
    _stop_event.clear()

    try:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(1)
        _server_sock = server
        threading.Thread(target=_accept_loop, args=(server,), daemon=True).start()
        logging.info(f"(isaac_mirror.py): Listening for Isaac Lab on {host}:{port}\n")
        return True
    except Exception as e:
        logging.error(f"(isaac_mirror.py): Failed to start server on {host}:{port}: {e}\n")
        _server_sock = None
        return False


def get_latest_message(max_age_s: float | None = None) -> dict[str, Any] | None:
    """Return latest message if fresh enough, else None."""
    if max_age_s is None:
        max_age_s = float(config.ISAAC_MIRROR_CONFIG["TIMEOUT_S"])
    with _latest_lock:
        if _latest_msg is None:
            return None
        age = time.time() - _latest_time
        if age > max_age_s:
            return None
        return dict(_latest_msg)


def message_age_s() -> float | None:
    with _latest_lock:
        if _latest_msg is None:
            return None
        return time.time() - _latest_time
