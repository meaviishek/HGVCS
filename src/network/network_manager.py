"""
NetworkManager – LAN file sharing via TCP + mDNS peer discovery.

Architecture:
  • LanFileServer  – asyncio TCP server that listens for incoming transfers
  • LanFileSender  – connects to peer and streams a file in chunks
  • PeerDiscovery  – zeroconf mDNS to find other HGVCS nodes on LAN

All I/O runs on a dedicated asyncio thread so the Qt GUI is never blocked.
"""

import asyncio
import hashlib
import json
import logging
import os
import socket
import struct
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Callable

log = logging.getLogger("hgvcs.network")

# ── constants ──────────────────────────────────────────────
PORT         = 9876
CHUNK        = 65_536        # 64 KB
MAGIC        = b"HGVCS\x01"  # 6-byte frame magic
SAVE_DIR     = Path.home() / "Downloads" / "HGVCS"

# ── zeroconf optional ──────────────────────────────────────
try:
    from zeroconf import ServiceBrowser, ServiceInfo, Zeroconf
    _ZC_OK = True
except ImportError:
    _ZC_OK = False
    log.warning("zeroconf not installed – peer discovery disabled")

SERVICE_TYPE = "_hgvcs._tcp.local."


# ══════════════════════════════════════════════════════════
# PEER REGISTRY
# ══════════════════════════════════════════════════════════
class PeerRegistry:
    """Thread-safe store of discovered LAN peers."""
    def __init__(self):
        self._lock  = threading.Lock()
        self._peers: Dict[str, dict] = {}   # name → {ip, port, name}

    def add(self, name: str, ip: str, port: int):
        with self._lock:
            self._peers[name] = {"name": name, "ip": ip, "port": port,
                                 "seen": time.time()}
        log.info(f"Peer discovered: {name} @ {ip}:{port}")

    def remove(self, name: str):
        with self._lock:
            self._peers.pop(name, None)

    def all(self) -> List[dict]:
        now = time.time()
        with self._lock:
            # expire peers not seen in 30 s
            stale = [n for n, p in self._peers.items()
                     if now - p["seen"] > 30]
            for n in stale:
                del self._peers[n]
            return list(self._peers.values())

    def first(self) -> Optional[dict]:
        peers = self.all()
        return peers[0] if peers else None


# ══════════════════════════════════════════════════════════
# PEER DISCOVERY (zeroconf mDNS)
# ══════════════════════════════════════════════════════════
class PeerDiscovery:
    def __init__(self, registry: PeerRegistry, my_name: str, port: int):
        self._registry = registry
        self._name     = my_name
        self._port     = port
        self._zc:  Optional["Zeroconf"] = None
        self._info: Optional["ServiceInfo"] = None

    def start(self):
        if not _ZC_OK:
            return
        try:
            self._zc = Zeroconf()
            ip_bytes  = socket.inet_aton(self._get_local_ip())
            self._info = ServiceInfo(
                SERVICE_TYPE,
                f"{self._name}.{SERVICE_TYPE}",
                addresses=[ip_bytes],
                port=self._port,
                properties={"version": "1.0"},
            )
            self._zc.register_service(self._info)
            ServiceBrowser(self._zc, SERVICE_TYPE, self)
            log.info(f"mDNS registered as {self._name}")
        except Exception as e:
            log.error(f"PeerDiscovery start error: {e}")

    def stop(self):
        if self._zc:
            try:
                if self._info:
                    self._zc.unregister_service(self._info)
                self._zc.close()
            except Exception:
                pass

    # zeroconf listener callbacks
    def add_service(self, zc, stype, name):
        try:
            info = zc.get_service_info(stype, name)
            if info and name != f"{self._name}.{SERVICE_TYPE}":
                ip   = socket.inet_ntoa(info.addresses[0])
                peer = name.split(".")[0]
                self._registry.add(peer, ip, info.port)
        except Exception as e:
            log.debug(f"add_service error: {e}")

    def remove_service(self, zc, stype, name):
        peer = name.split(".")[0]
        self._registry.remove(peer)

    def update_service(self, zc, stype, name):
        self.add_service(zc, stype, name)

    @staticmethod
    def _get_local_ip() -> str:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"


# ══════════════════════════════════════════════════════════
# PROTOCOL HELPERS
# ══════════════════════════════════════════════════════════
def _encode_header(filename: str, filesize: int) -> bytes:
    meta = json.dumps({"filename": filename, "size": filesize}).encode()
    return MAGIC + struct.pack(">I", len(meta)) + meta

def _decode_header(data: bytes) -> Optional[dict]:
    if not data.startswith(MAGIC):
        return None
    meta_len = struct.unpack(">I", data[6:10])[0]
    return json.loads(data[10:10+meta_len])


# ══════════════════════════════════════════════════════════
# FILE SERVER  (receive side)
# ══════════════════════════════════════════════════════════
class LanFileServer:
    def __init__(self, port: int, registry: PeerRegistry,
                 on_incoming: Optional[Callable] = None,
                 on_progress: Optional[Callable] = None):
        self._port        = port
        self._registry    = registry
        self._on_incoming = on_incoming   # (filename, size, peer_ip) → True/False
        self._on_progress = on_progress   # (filename, received, total)
        self._server      = None
        self._loop:  Optional[asyncio.AbstractEventLoop] = None
        self._accept_next = threading.Event()  # set when user accepts transfer

    def set_accept_callback(self, cb): self._on_incoming = cb
    def set_progress_callback(self, cb): self._on_progress = cb
    def accept_transfer(self): self._accept_next.set()

    async def _handle(self, reader: asyncio.StreamReader,
                      writer: asyncio.StreamWriter):
        peer_ip = writer.get_extra_info("peername", ("?", 0))[0]
        try:
            # read header
            hdr_prefix = await reader.readexactly(10 + 4096)  # generously sized
        except asyncio.IncompleteReadError as e:
            hdr_prefix = e.partial

        meta = None
        for end in range(10, len(hdr_prefix)+1):
            try:
                meta = _decode_header(hdr_prefix[:end])
                if meta:
                    break
            except Exception:
                continue

        if not meta:
            log.warning(f"Invalid transfer header from {peer_ip}")
            writer.close()
            return

        filename = os.path.basename(meta["filename"])
        filesize = meta["size"]
        hdr_size = 6 + 4 + len(json.dumps(meta).encode())
        leftover = hdr_prefix[hdr_size:]

        log.info(f"Incoming transfer: {filename} ({filesize} bytes) from {peer_ip}")

        # call UI callback to ask user
        accepted = True
        if self._on_incoming:
            accepted = self._on_incoming(filename, filesize, peer_ip)

        if not accepted:
            writer.write(b"REJECT")
            await writer.drain()
            writer.close()
            return

        writer.write(b"ACCEPT")
        await writer.drain()

        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        save_path = SAVE_DIR / filename
        received = 0
        try:
            with open(save_path, "wb") as f:
                if leftover:
                    f.write(leftover)
                    received += len(leftover)
                while received < filesize:
                    chunk = await reader.read(min(CHUNK, filesize - received))
                    if not chunk:
                        break
                    f.write(chunk)
                    received += len(chunk)
                    if self._on_progress:
                        self._on_progress(filename, received, filesize)
        except Exception as e:
            log.error(f"Error receiving file: {e}")
        finally:
            writer.close()

        if received == filesize:
            log.info(f"Transfer complete: {save_path}")
        else:
            log.warning(f"Transfer incomplete: {received}/{filesize}")

    async def _serve(self):
        self._server = await asyncio.start_server(
            self._handle, "0.0.0.0", self._port
        )
        log.info(f"LAN file server on port {self._port}")
        try:
            async with self._server:
                await self._server.serve_forever()
        except asyncio.CancelledError:
            pass   # clean shutdown — suppress traceback

    def start_in_thread(self):
        def _run():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            try:
                self._loop.run_until_complete(self._serve())
            except asyncio.CancelledError:
                pass   # clean shutdown
            except Exception as e:
                log.error(f"LanFileServer error: {e}")
            finally:
                try:
                    self._loop.close()
                except Exception:
                    pass
        t = threading.Thread(target=_run, daemon=True, name="lan-server")
        t.start()

    def stop(self):
        if self._server and self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._server.close)


# ══════════════════════════════════════════════════════════
# FILE SENDER  (send side)
# ══════════════════════════════════════════════════════════
class LanFileSender:
    def __init__(self, on_progress: Optional[Callable] = None,
                 on_done:     Optional[Callable] = None):
        self._on_progress = on_progress  # (filename, sent, total)
        self._on_done     = on_done      # (filename, success)

    def send(self, filepath: str, peer_ip: str, peer_port: int):
        """Non-blocking – runs in daemon thread."""
        t = threading.Thread(
            target=self._send_sync,
            args=(filepath, peer_ip, peer_port),
            daemon=True,
            name="lan-sender"
        )
        t.start()

    def _send_sync(self, filepath: str, peer_ip: str, peer_port: int):
        filename = os.path.basename(filepath)
        filesize = os.path.getsize(filepath)
        header   = _encode_header(filename, filesize)
        success  = False

        try:
            sock = socket.create_connection((peer_ip, peer_port), timeout=10)
            sock.sendall(header)

            # wait for accept/reject
            resp = sock.recv(6)
            if resp != b"ACCEPT":
                log.info(f"Transfer rejected by {peer_ip}")
                sock.close()
                if self._on_done:
                    self._on_done(filename, False)
                return

            sent = 0
            with open(filepath, "rb") as f:
                while sent < filesize:
                    chunk = f.read(CHUNK)
                    if not chunk:
                        break
                    sock.sendall(chunk)
                    sent += len(chunk)
                    if self._on_progress:
                        self._on_progress(filename, sent, filesize)

            sock.close()
            success = sent == filesize
            log.info(f"Sent {filename} to {peer_ip}: {sent}/{filesize}")
        except Exception as e:
            log.error(f"LanFileSender error: {e}")
        finally:
            if self._on_done:
                self._on_done(filename, success)


# ══════════════════════════════════════════════════════════
# NETWORK MANAGER
# ══════════════════════════════════════════════════════════
class NetworkManager:
    """
    Top-level coordinator.  Wires together:
      • LanFileServer    (TCP receive)
      • LanFileSender    (TCP send)
      • PeerDiscovery    (mDNS)
      • PeerRegistry     (known peers)

    Gesture events:
      on_gesture("wave")  – cancel any pending transfer
    """

    def __init__(self, config, event_bus):
        self.config     = config
        self.event_bus  = event_bus
        self._port      = int(config.get("port", PORT))
        self._my_name   = socket.gethostname()

        self.registry   = PeerRegistry()
        self._server    = LanFileServer(
            port=self._port,
            registry=self.registry,
            on_incoming=self._on_incoming_transfer,
            on_progress=self._on_rx_progress,
        )
        self._sender    = LanFileSender(
            on_progress=self._on_tx_progress,
            on_done=self._on_tx_done,
        )
        self._discovery = PeerDiscovery(self.registry, self._my_name, self._port)
        # UI callbacks (injected from main_window)
        self._on_rx_ui:         Optional[Callable] = None   # (name, size, ip) → bool
        self._on_progress_ui:   Optional[Callable] = None   # (name, done, total, direction)
        self._on_transfer_done: Optional[Callable] = None   # (name, success, direction)

        log.info("NetworkManager initialised")

    # ── UI hooks ──────────────────────────────────────────
    def set_rx_accept_cb(self, cb):      self._on_rx_ui        = cb
    def set_progress_cb(self, cb):       self._on_progress_ui  = cb
    def set_transfer_done_cb(self, cb):  self._on_transfer_done= cb

    # ── lifecycle ──────────────────────────────────────────
    def start(self):
        self._server.start_in_thread()
        self._discovery.start()
        log.info(f"NetworkManager started (port {self._port})")

    def stop(self):
        self._server.stop()
        self._discovery.stop()

    def on_gesture(self, gesture: str):
        if gesture == "wave":
            log.info("Transfer cancelled by wave gesture")
        elif gesture == "open_palm":
            log.debug("open_palm gesture received by NetworkManager (no-op)")

    def send_file(self, filepath: str, peer: Optional[dict] = None):
        """Direct send API (also called from UI button)."""
        if peer is None:
            peer = self.registry.first()
        if peer is None:
            log.warning("No peer available for file send")
            return
        self._sender.send(filepath, peer["ip"], peer["port"])


    def _on_incoming_transfer(self, filename: str, size: int, ip: str) -> bool:
        if self._on_rx_ui:
            return self._on_rx_ui(filename, size, ip)
        return True   # auto-accept if no UI callback

    def _on_rx_progress(self, filename: str, done: int, total: int):
        if self._on_progress_ui:
            self._on_progress_ui(filename, done, total, "rx")

    def _on_tx_progress(self, filename: str, done: int, total: int):
        if self._on_progress_ui:
            self._on_progress_ui(filename, done, total, "tx")

    def _on_tx_done(self, filename: str, success: bool):
        if self._on_transfer_done:
            self._on_transfer_done(filename, success, "tx")
