# -*- coding: utf-8 -*-
"""
CH341 I2C 调试 UI - V1.7
首个器件页面：MLX90393 三轴磁传感器（Legacy 2015 ABA-011 / Current 2023 全系列）

运行环境：Windows 11 / Python 3.10+
Python 第三方依赖：无（仅标准库）
硬件驱动依赖：WCH 官方 CH341PAR 驱动 + 与 Python 位数匹配的 CH341 DLL
    64-bit Python：优先 CH341DLLA64.DLL
    32-bit Python：CH341DLL.DLL

设计：
1) CH341 官方 DLL -> I2C 通信封装（ctypes）
2) 器件页面插件式结构（ttk.Notebook）
3) MLX90393 专用页面：支持 Legacy 2015 与 Current 2023 两套器件 Profile（配置档）
4) Legacy 2015 默认按 MLX90393S-LW(QFN16)-ABA-011-RE：0x0C~0x0F + 2015 灵敏度表
5) ACK 探测严格保留 WCH 官方特殊模式：OUT 长度=0（0x80）时发送 1 字节并返回 ACK；0x81 不是该 ACK 采样方式
6) 全总线被动扫描 + MLX90393 协议指纹识别 + 复位、配置读写、单次/连续采样、曲线、软件置零
7) 新增 5×MLX90393 同步采集页面：默认地址 0x0C/0x0D/0x0E/0x0F/0x10，可独立修改 ID；采用“组触发 SM -> 统一等待 -> 逐个 RM”以降低采样时间偏差
8) 支持“全部 XYZ（15 曲线）”或指定 X/Y/Z/|B| 轴的 5 模块同步曲线
9) 原始 I2C 调试页面：任意地址的 Write / Write-Read(Repeated Start)

注意：
- MLX90393 的 HS(Store) 非易失存储命令本程序默认不提供，避免误写 NVRAM。
- “应用配置”只写易失 RAM，断电/复位后会按芯片 NVRAM 恢复。
"""

from __future__ import annotations

import ctypes
import math
import os
import queue
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

# ============================================================
# CH341 DLL -> I2C 底层
# ============================================================


class CH341Error(RuntimeError):
    pass


class I2CTransferError(CH341Error):
    pass


@dataclass
class CH341DeviceInfo:
    index: int
    ic_version: int

    @property
    def chip_name(self) -> str:
        return {0x10: "CH341", 0x20: "CH341A", 0x30: "CH341A3"}.get(
            self.ic_version, f"CH341 系列(VerIC=0x{self.ic_version:02X})"
        )

    @property
    def display(self) -> str:
        return f"设备 #{self.index}  |  {self.chip_name}"


class CH341Bridge:
    """
    Windows 下通过 WCH 官方 CH341 DLL 调用 CH341StreamI2C。

    CH341StreamI2C 的 iWriteBuffer 首字节是 8-bit I2C 地址字节。
    当同时存在写数据和读数据时，DLL 会生成 Repeated START（重复起始），
    并把首地址字节的 R/W 位切换为读方向。
    """

    MAX_DEVICES = 16
    INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

    # CH341A I2C stream command bytes（WCH CH341DLL.H）
    _I2C_STREAM = 0xAA
    _I2C_STA = 0x74
    _I2C_STO = 0x75
    _I2C_OUT = 0x80
    _I2C_OUT_ACK1 = 0x80  # WCH 特殊模式：长度字段=0，仍发送后续1字节，并把ACK状态回传
    _I2C_END = 0x00
    _I2C_STM_MAX = 32  # min(0x3F, mCH341_PACKET_LENGTH=32)

    def __init__(self) -> None:
        self.dll = None
        self.dll_path = ""
        self.index: Optional[int] = None
        self.handle = None
        self.clock_hz = 100_000
        self._lock = threading.RLock()
        # V1.6：CH341 的 ACK 原始回读在部分 DLL/硬件组合上可能固定返回某个状态字。
        # 因此 ACK 只作诊断，不再默认阻断 I2C 读写。MLX90393 由协议状态字/寄存器/测量序列确认。
        self.enforce_ack = False

    @property
    def connected(self) -> bool:
        return self.index is not None and self.handle not in (None, 0, self.INVALID_HANDLE_VALUE)

    @staticmethod
    def _candidate_dll_names() -> list[str]:
        is_64 = struct.calcsize("P") == 8
        if is_64:
            return ["CH341DLLA64.DLL", "CH341DLLA64.dll", "CH341DLL.DLL", "CH341DLL.dll"]
        return ["CH341DLL.DLL", "CH341DLL.dll"]

    @classmethod
    def _candidate_dll_paths(cls) -> list[str]:
        names = cls._candidate_dll_names()
        here = os.path.abspath(os.path.dirname(__file__))
        cwd = os.path.abspath(os.getcwd())
        windir = os.environ.get("WINDIR", r"C:\Windows")
        system_dir = os.path.join(windir, "System32")
        syswow_dir = os.path.join(windir, "SysWOW64")
        dirs = [here, cwd, system_dir, syswow_dir]
        out: list[str] = []
        seen = set()
        for d in dirs:
            for name in names:
                p = os.path.join(d, name)
                key = p.lower()
                if key not in seen:
                    seen.add(key)
                    out.append(p)
        # 最后再允许 Windows DLL 搜索路径按名称查找
        for name in names:
            key = name.lower()
            if key not in seen:
                seen.add(key)
                out.append(name)
        return out

    def _load_dll(self):
        if self.dll is not None:
            return self.dll
        if os.name != "nt":
            raise CH341Error("CH341 官方 DLL 方式仅支持 Windows。当前系统不是 Windows。")

        loader = getattr(ctypes, "WinDLL", ctypes.CDLL)
        errors = []
        for p in self._candidate_dll_paths():
            # 对绝对路径，文件不存在就不尝试；仅名称让 Windows 自己搜索
            if os.path.isabs(p) and not os.path.exists(p):
                continue
            try:
                dll = loader(p)
                self._bind_functions(dll)
                self.dll = dll
                self.dll_path = os.path.abspath(p) if os.path.exists(p) else p
                return dll
            except Exception as e:
                errors.append(f"{p}: {e}")

        bits = struct.calcsize("P") * 8
        wanted = "CH341DLLA64.DLL" if bits == 64 else "CH341DLL.DLL"
        detail = "\n".join(errors[-3:]) if errors else "未在程序目录、当前目录或 Windows 系统目录发现可加载 DLL。"
        raise CH341Error(
            f"未能加载 WCH CH341 DLL。当前 Python 为 {bits}-bit。\n"
            f"请安装官方 CH341PAR 驱动，并把 {wanted} 放到本程序 .py 同目录。\n"
            f"加载详情：{detail}"
        )

    @staticmethod
    def _bind_functions(dll) -> None:
        c_ulong = ctypes.c_ulong
        c_void_p = ctypes.c_void_p

        dll.CH341OpenDevice.argtypes = [c_ulong]
        dll.CH341OpenDevice.restype = c_void_p
        dll.CH341CloseDevice.argtypes = [c_ulong]
        dll.CH341CloseDevice.restype = None
        dll.CH341GetVerIC.argtypes = [c_ulong]
        dll.CH341GetVerIC.restype = c_ulong
        dll.CH341SetStream.argtypes = [c_ulong, c_ulong]
        dll.CH341SetStream.restype = ctypes.c_int
        dll.CH341StreamI2C.argtypes = [c_ulong, c_ulong, c_void_p, c_ulong, c_void_p]
        dll.CH341StreamI2C.restype = ctypes.c_int
        # 用于读取 CH341 I2C 命令流返回的 ACK 位。CH341StreamI2C 的 BOOL
        # 只代表 USB/DLL 事务执行成功，不能单独作为“从机 ACK”判断。
        if hasattr(dll, "CH341WriteRead"):
            dll.CH341WriteRead.argtypes = [
                c_ulong, c_ulong, c_void_p, c_ulong, c_ulong,
                ctypes.POINTER(c_ulong), c_void_p
            ]
            dll.CH341WriteRead.restype = ctypes.c_int
        if hasattr(dll, "CH341WriteData"):
            dll.CH341WriteData.argtypes = [c_ulong, c_void_p, ctypes.POINTER(c_ulong)]
            dll.CH341WriteData.restype = ctypes.c_int

        # 以下函数在常见 WCH DLL 中存在；没有也不影响核心 I2C。
        if hasattr(dll, "CH341FlushBuffer"):
            dll.CH341FlushBuffer.argtypes = [c_ulong]
            dll.CH341FlushBuffer.restype = ctypes.c_int
        if hasattr(dll, "CH341GetVersion"):
            dll.CH341GetVersion.argtypes = []
            dll.CH341GetVersion.restype = c_ulong
        if hasattr(dll, "CH341GetDrvVersion"):
            dll.CH341GetDrvVersion.argtypes = []
            dll.CH341GetDrvVersion.restype = c_ulong

    @staticmethod
    def _valid_handle(handle) -> bool:
        return handle not in (None, 0, CH341Bridge.INVALID_HANDLE_VALUE)

    def enumerate_devices(self) -> list[CH341DeviceInfo]:
        dll = self._load_dll()
        result: list[CH341DeviceInfo] = []
        # WCH DLL 的设备序号范围通常为 0~15。逐个尝试打开最可靠。
        for idx in range(self.MAX_DEVICES):
            handle = dll.CH341OpenDevice(idx)
            if not self._valid_handle(handle):
                continue
            try:
                ver = int(dll.CH341GetVerIC(idx))
                result.append(CH341DeviceInfo(index=idx, ic_version=ver))
            finally:
                dll.CH341CloseDevice(idx)
        return result

    def open(self, index: int = 0, clock_hz: int = 100_000) -> None:
        with self._lock:
            self.close()
            dll = self._load_dll()
            index = int(index)
            handle = dll.CH341OpenDevice(index)
            if not self._valid_handle(handle):
                raise CH341Error(
                    f"无法打开 CH341 设备 #{index}。请检查 CH341PAR 驱动、USB 连接，并关闭其它占用 CH341 的烧录软件。"
                )
            self.index = index
            self.handle = handle
            try:
                self.configure_i2c(clock_hz)
                if hasattr(dll, "CH341FlushBuffer"):
                    dll.CH341FlushBuffer(index)
            except Exception:
                self.close()
                raise

    def close(self) -> None:
        with self._lock:
            if self.index is not None and self.dll is not None:
                try:
                    self.dll.CH341CloseDevice(int(self.index))
                except Exception:
                    pass
            self.index = None
            self.handle = None

    def _require(self):
        if not self.connected or self.dll is None or self.index is None:
            raise CH341Error("CH341 尚未连接")
        return self.dll, int(self.index)

    def configure_i2c(self, clock_hz: int = 100_000) -> None:
        dll, idx = self._require()
        clock_hz = int(clock_hz)
        # CH341SetStream: bit1..0 = 00/01/10/11 -> 20/100/400/750 kHz
        speed_map = {20_000: 0x00, 100_000: 0x01, 400_000: 0x02, 750_000: 0x03}
        if clock_hz not in speed_map:
            raise ValueError("CH341 I²C 时钟请选择 20 / 100 / 400 / 750 kHz")
        if not dll.CH341SetStream(idx, speed_map[clock_hz]):
            raise CH341Error(f"CH341SetStream 设置 I²C={clock_hz//1000} kHz 失败")
        self.clock_hz = clock_hz

    def _issue_i2c_stop(self) -> None:
        """按 WCH 官方示例单独产生 STOP，避免把 STOP 混进 ACK 回读命令流。"""
        dll, idx = self._require()
        cmd = bytes([self._I2C_STREAM, self._I2C_STO, self._I2C_END])
        wbuf = (ctypes.c_ubyte * len(cmd))(*cmd)
        if hasattr(dll, "CH341WriteData"):
            n = ctypes.c_ulong(len(cmd))
            dll.CH341WriteData(idx, ctypes.cast(wbuf, ctypes.c_void_p), ctypes.byref(n))
        else:
            rlen = ctypes.c_ulong(0)
            dummy = (ctypes.c_ubyte * 1)()
            dll.CH341WriteRead(
                idx, len(cmd), ctypes.cast(wbuf, ctypes.c_void_p),
                self._I2C_STM_MAX, 0, ctypes.byref(rlen), ctypes.cast(dummy, ctypes.c_void_p)
            )

    def probe_address(self, address7: int) -> tuple[bool, int | None]:
        """
        对 7-bit 地址做 ACK 探测。

        V1.6 严格复现 WCH 官方 IIC_OutByteCheckAck / X76F640_AckPolling：
        AA, START, OUT(length=0), address, END。
        注意：CH341 的 I2C_STM_OUT 有特殊定义：length=0（命令字0x80）并不是“不发送”，
        而是“发送紧随其后的1个字节，并返回该字节的ACK状态”。因此这里必须是0x80，不能改成0x81。
        先读取 ACK，再单独发送 STOP；返回字节 bit7=0 表示 ACK。
        """
        dll, idx = self._require()
        if not hasattr(dll, "CH341WriteRead"):
            raise CH341Error(
                "当前 CH341 DLL 缺少 CH341WriteRead，无法可靠读取 I2C ACK。"
                "请使用 WCH 官方 CH341DLL/CH341DLLA64。"
            )

        addr8w = self._addr_byte(address7)
        cmd = bytes([
            self._I2C_STREAM,
            self._I2C_STA,
            self._I2C_OUT_ACK1,
            addr8w,
            self._I2C_END,
        ])
        wbuf = (ctypes.c_ubyte * len(cmd))(*cmd)
        rbuf = (ctypes.c_ubyte * self._I2C_STM_MAX)()
        rlen = ctypes.c_ulong(0)
        try:
            ok = dll.CH341WriteRead(
                idx,
                len(cmd),
                ctypes.cast(wbuf, ctypes.c_void_p),
                self._I2C_STM_MAX,
                1,
                ctypes.byref(rlen),
                ctypes.cast(rbuf, ctypes.c_void_p),
            )
            if not ok or rlen.value < 1:
                return False, None
            ack_byte = int(rbuf[rlen.value - 1])
            return (ack_byte & 0x80) == 0, ack_byte
        finally:
            try:
                self._issue_i2c_stop()
            except Exception:
                pass

    def _require_slave_ack(self, address7: int) -> None:
        ack, ack_byte = self.probe_address(address7)
        if not ack:
            detail = "无返回" if ack_byte is None else f"ACK状态=0x{ack_byte:02X}"
            raise I2CTransferError(
                f"I2C 从机 0x{int(address7):02X} 未产生 ACK（{detail}）。"
                "请检查地址、供电、共地、SDA/SCL、上拉以及 MLX90393 的 MS/CS。"
            )

    def _stream_i2c(self, write_buffer: bytes, read_len: int = 0) -> bytes:
        dll, idx = self._require()
        write_buffer = bytes(write_buffer)
        read_len = int(read_len)
        if not write_buffer:
            raise ValueError("CH341StreamI2C 至少需要 1 字节地址")
        if not (0 <= read_len <= 4096):
            raise ValueError("CH341 单次读取长度限制为 0~4096 字节")
        if len(write_buffer) > 4096:
            raise ValueError("CH341 单次写入长度过长（>4096 字节）")

        wbuf = (ctypes.c_ubyte * len(write_buffer))(*write_buffer)
        if read_len > 0:
            rbuf = (ctypes.c_ubyte * read_len)()
            rptr = ctypes.cast(rbuf, ctypes.c_void_p)
        else:
            rbuf = None
            rptr = None

        ok = dll.CH341StreamI2C(
            idx,
            len(write_buffer),
            ctypes.cast(wbuf, ctypes.c_void_p),
            read_len,
            rptr,
        )
        if not ok:
            raise I2CTransferError(
                "CH341 I²C 事务执行失败：请检查 USB/DLL、SDA/SCL 接线、上拉或总线状态。"
            )
        if rbuf is None:
            return b""
        return bytes(rbuf)

    @staticmethod
    def _addr_byte(address7: int) -> int:
        address7 = int(address7)
        if not (0x01 <= address7 <= 0x77):
            raise ValueError("I2C 7-bit 地址必须在 0x01~0x77")
        return (address7 << 1) & 0xFE

    def write_i2c_unchecked(self, address7: int, tx_data: bytes) -> None:
        """不依赖 ACK 判定的原始写。用于协议指纹/诊断。"""
        tx_data = bytes(tx_data)
        if not tx_data:
            raise ValueError("I2C 写数据不能为空")
        with self._lock:
            self._stream_i2c(bytes([self._addr_byte(address7)]) + tx_data, 0)

    def read_i2c_unchecked(self, address7: int, rx_len: int) -> bytes:
        """不依赖 ACK 判定的原始读。无从机时 CH341 常读回全 0xFF。"""
        rx_len = int(rx_len)
        if rx_len <= 0:
            raise ValueError("I2C 读取长度必须 > 0")
        with self._lock:
            return self._stream_i2c(bytes([self._addr_byte(address7)]), rx_len)

    def write_read_i2c_unchecked(self, address7: int, tx_data: bytes, rx_len: int) -> bytes:
        """不依赖 ACK 判定的写后读（Repeated START）。"""
        tx_data = bytes(tx_data)
        rx_len = int(rx_len)
        if not tx_data:
            raise ValueError("Write-Read 的写入部分不能为空")
        if rx_len <= 0:
            raise ValueError("Write-Read 的读取长度必须 > 0")
        with self._lock:
            return self._stream_i2c(bytes([self._addr_byte(address7)]) + tx_data, rx_len)

    def passive_probe(self, address7: int) -> dict:
        """
        双路、低风险地址诊断：
        1) WCH 命令流 ACK 回读（仅诊断）
        2) 不写寄存器，仅直接读 1 字节；无器件时常见 0xFF
        注意：raw_read!=0xFF 只是“疑似有器件”，不是通用 I2C 标准 ACK 证明。
        """
        ack = False
        ack_raw = None
        ack_error = None
        try:
            ack, ack_raw = self.probe_address(address7)
        except Exception as e:
            ack_error = str(e)
        read1 = None
        read_error = None
        try:
            r = self.read_i2c_unchecked(address7, 1)
            read1 = r[0] if r else None
        except Exception as e:
            read_error = str(e)
        return {
            "address": int(address7),
            "ack": bool(ack),
            "ack_raw": ack_raw,
            "ack_error": ack_error,
            "read1": read1,
            "read_error": read_error,
            "possible": bool(ack) or (read1 is not None and read1 != 0xFF),
        }

    def write_i2c(self, address7: int, tx_data: bytes) -> None:
        tx_data = bytes(tx_data)
        if not tx_data:
            raise ValueError("I2C 写数据不能为空")
        with self._lock:
            if self.enforce_ack:
                self._require_slave_ack(address7)
            # [8-bit address(W)] + payload
            self._stream_i2c(bytes([self._addr_byte(address7)]) + tx_data, 0)

    def read_i2c(self, address7: int, rx_len: int) -> bytes:
        rx_len = int(rx_len)
        if rx_len <= 0:
            raise ValueError("I2C 读取长度必须 > 0")
        with self._lock:
            if self.enforce_ack:
                self._require_slave_ack(address7)
            # 只有地址字节 + read_len 时，CH341StreamI2C 自动以读方向发地址。
            return self._stream_i2c(bytes([self._addr_byte(address7)]), rx_len)

    def write_read_i2c(self, address7: int, tx_data: bytes, rx_len: int) -> bytes:
        """写后读：CH341StreamI2C 在写与读之间自动生成 Repeated START。"""
        tx_data = bytes(tx_data)
        rx_len = int(rx_len)
        if not tx_data:
            raise ValueError("Write-Read 的写入部分不能为空")
        if rx_len <= 0:
            raise ValueError("Write-Read 的读取长度必须 > 0")
        with self._lock:
            if self.enforce_ack:
                self._require_slave_ack(address7)
            return self._stream_i2c(bytes([self._addr_byte(address7)]) + tx_data, rx_len)

    def dll_info(self) -> str:
        if self.dll is None:
            try:
                self._load_dll()
            except Exception:
                return "DLL 未加载"
        parts = [os.path.basename(self.dll_path) if self.dll_path else "CH341 DLL"]
        try:
            if hasattr(self.dll, "CH341GetVersion"):
                parts.append(f"DLL Ver=0x{int(self.dll.CH341GetVersion()):X}")
        except Exception:
            pass
        try:
            if hasattr(self.dll, "CH341GetDrvVersion"):
                parts.append(f"Driver Ver=0x{int(self.dll.CH341GetDrvVersion()):X}")
        except Exception:
            pass
        return " | ".join(parts)


# ============================================================
# MLX90393 器件驱动
# ============================================================


class MLX90393Error(RuntimeError):
    pass


class MLX90393:
    CMD_SB = 0x10
    CMD_SW = 0x20
    CMD_SM = 0x30
    CMD_RM = 0x40
    CMD_RR = 0x50
    CMD_WR = 0x60
    CMD_EX = 0x80
    CMD_HR = 0xD0
    CMD_HS = 0xE0
    CMD_RT = 0xF0

    AXIS_ALL = 0x0E  # Z,Y,X = 1; T = 0
    TEMP = 0x01
    ALL = 0x0F

    PROFILE_LEGACY_2015 = "Legacy 2015 / ABA-011 / QFN16"
    PROFILE_CURRENT_2023 = "Current 2023 / 全系列"

    # 2015 Rev.002（用户成品模块配套资料），HALLCONF=0xC, TC off，单位 µT/LSB
    # 表3：GAIN_SEL 0~7；每个分辨率 RES=0~3；(XY, Z)
    SENS_2015_HALLCONF_C = (
        ((0.805, 1.468), (1.610, 2.936), (3.220, 5.872), (6.440, 11.744)),
        ((0.644, 1.174), (1.288, 2.349), (2.576, 4.698), (5.152, 9.395)),
        ((0.483, 0.881), (0.966, 1.762), (1.932, 3.523), (3.864, 7.046)),
        ((0.403, 0.734), (0.805, 1.468), (1.610, 2.936), (3.220, 5.872)),
        ((0.322, 0.587), (0.644, 1.174), (1.288, 2.349), (2.576, 4.698)),
        ((0.268, 0.489), (0.537, 0.979), (1.073, 1.957), (2.147, 3.915)),
        ((0.215, 0.391), (0.429, 0.783), (0.859, 1.566), (1.717, 3.132)),
        ((0.161, 0.294), (0.322, 0.587), (0.644, 1.174), (1.288, 2.349)),
    )

    # 2023 Rev.010, HALLCONF=0xC, TC off, 25°C，单位 µT/LSB
    SENS_25C_HALLCONF_C = (
        ((0.751, 1.210), (1.502, 2.420), (3.004, 4.840), (6.009, 9.680)),
        ((0.601, 0.968), (1.202, 1.936), (2.403, 3.872), (4.840, 7.744)),
        ((0.451, 0.726), (0.901, 1.452), (1.803, 2.904), (3.605, 5.808)),
        ((0.376, 0.605), (0.751, 1.210), (1.502, 2.420), (3.004, 4.840)),
        ((0.300, 0.484), (0.601, 0.968), (1.202, 1.936), (2.403, 3.872)),
        ((0.250, 0.403), (0.501, 0.807), (1.001, 1.613), (2.003, 3.227)),
        ((0.200, 0.323), (0.401, 0.645), (0.801, 1.291), (1.602, 2.581)),
        ((0.150, 0.242), (0.300, 0.484), (0.601, 0.968), (1.202, 1.936)),
    )

    # 表 19：典型 TXYZ 转换时间(ms)，行=DIG_FILT 0~7，列=OSR 0~3，OSR2=0
    TCONV_MS = (
        (1.27, 1.84, 3.00, 5.30),
        (1.46, 2.23, 3.76, 6.84),
        (1.84, 3.00, 5.30, 9.91),
        (2.61, 4.53, 8.37, 16.05),
        (4.15, 7.60, 14.52, 28.34),
        (7.22, 13.75, 26.80, 52.92),
        (13.36, 26.04, 51.38, 102.07),
        (25.65, 50.61, 100.53, 200.37),
    )

    GAIN_NAMES = (
        "0 = 5×",
        "1 = 4×",
        "2 = 3×",
        "3 = 2.5×",
        "4 = 2×",
        "5 = 1.67×",
        "6 = 1.33×",
        "7 = 1×",
    )
    RES_NAMES = ("0 = 16-bit", "1 = 17-bit", "2 = 18-bit", "3 = 19-bit")

    def __init__(self, bridge: CH341Bridge, address: int = 0x0C, profile: str = PROFILE_LEGACY_2015) -> None:
        self.bridge = bridge
        self.address = address
        self.profile = profile
        self.gain = 7
        self.res_x = 0
        self.res_y = 0
        self.res_z = 0
        self.filter = 0
        self.osr = 0
        self.osr2 = 0
        self.tcmp = False
        self.hallconf = 0x0C
        self.tref = None
        self.last_status = 0

    @staticmethod
    def decode_status(status: int) -> str:
        parts = []
        if status & 0x80:
            parts.append("BURST")
        if status & 0x40:
            parts.append("WOC")
        if status & 0x20:
            parts.append("SM")
        if status & 0x10:
            parts.append("ERROR")
        if status & 0x08:
            parts.append("SED")
        if status & 0x04:
            parts.append("RS")
        parts.append(f"D={status & 0x03}")
        return " | ".join(parts)

    def _check_status(self, status: int, context: str = "") -> None:
        self.last_status = status
        if status & 0x10:
            raise MLX90393Error(
                f"MLX90393 状态字 ERROR=1（0x{status:02X}，{self.decode_status(status)}）"
                + (f"；操作={context}" if context else "")
            )

    def transceive(self, payload: bytes, data_len: int = 0, check_status: bool = True) -> bytes:
        """data_len 不含强制返回的 1 字节 Status。"""
        rx = self.bridge.write_read_i2c(self.address, bytes(payload), data_len + 1)
        if not rx:
            raise MLX90393Error("MLX90393 未返回状态字")
        self.last_status = rx[0]
        if check_status:
            self._check_status(rx[0], payload.hex(" "))
        return rx

    def transceive_unchecked(self, payload: bytes, data_len: int = 0) -> bytes:
        """协议诊断专用：绕过 CH341 ACK 判定，直接执行写后读。"""
        return self.bridge.write_read_i2c_unchecked(self.address, bytes(payload), int(data_len) + 1)

    @classmethod
    def fingerprint(cls, bridge: CH341Bridge, address: int, profile: str = PROFILE_LEGACY_2015) -> dict:
        """
        MLX90393 协议指纹识别。
        不依赖 CH341 的 ACK 判定，而检查：
          EX 状态字 -> RR(0x00/0x02) -> SM(XYZ) -> RM(XYZ)
        强指纹：RR 返回 16-bit 配置、HALLCONF 合理；RM 返回 6 字节且 Status.D=2。
        该流程只建议用于 MLX90393 合法地址范围 0x0C~0x1B，避免对未知 I2C 器件发送写命令。
        """
        s = cls(bridge, int(address), profile=profile)
        out = {
            "address": int(address), "score": 0, "confidence": "NO",
            "ex": None, "rr0": None, "rr2": None, "sm": None, "rm": None,
            "reg0": None, "reg2": None, "hallconf": None, "gain": None,
            "reason": [],
        }

        def valid_status(st):
            return st is not None and st != 0xFF

        # 1) EX：无器件时常见 0xFF；真实 MLX90393 应返回结构化状态字。
        try:
            rx = s.transceive_unchecked(bytes([s.CMD_EX]), 0)
            st = rx[0] if rx else None
            out["ex"] = st
            if valid_status(st):
                out["score"] += 1
                out["reason"].append(f"EX=0x{st:02X}")
                if not (st & 0x10):
                    out["score"] += 1
        except Exception as e:
            out["reason"].append(f"EX异常:{e}")

        # 2) RR register 0：默认/常用 HALLCONF 为 0xC 或 0x0，是较强特征。
        try:
            rx = s.transceive_unchecked(bytes([s.CMD_RR, 0x00]), 2)
            out["rr0"] = rx.hex(" ") if rx else None
            if len(rx) == 3 and valid_status(rx[0]):
                reg0 = (rx[1] << 8) | rx[2]
                hall = reg0 & 0x0F
                gain = (reg0 >> 4) & 0x07
                out.update(reg0=reg0, hallconf=hall, gain=gain)
                out["score"] += 2
                if not (rx[0] & 0x10):
                    out["score"] += 1
                if hall in (0x0C, 0x00):
                    out["score"] += 2
                    out["reason"].append(f"RR0=0x{reg0:04X},HALLCONF=0x{hall:X},GAIN={gain}")
                else:
                    out["reason"].append(f"RR0=0x{reg0:04X},HALLCONF=0x{hall:X}")
        except Exception as e:
            out["reason"].append(f"RR0异常:{e}")

        # 3) RR register 2：读取 OSR/DIG_FILT，用于估计后续转换等待时间。
        delay_s = 0.25
        try:
            rx = s.transceive_unchecked(bytes([s.CMD_RR, (0x02 << 2) & 0xFC]), 2)
            out["rr2"] = rx.hex(" ") if rx else None
            if len(rx) == 3 and valid_status(rx[0]):
                reg2 = (rx[1] << 8) | rx[2]
                out["reg2"] = reg2
                out["score"] += 1
                filt = (reg2 >> 2) & 0x07
                osr = reg2 & 0x03
                try:
                    delay_s = max(0.004, min(0.30, (cls.TCONV_MS[filt][osr] * 1.25 + 1.0) / 1000.0))
                except Exception:
                    delay_s = 0.25
                out["reason"].append(f"RR2=0x{reg2:04X},FILT={filt},OSR={osr}")
        except Exception as e:
            out["reason"].append(f"RR2异常:{e}")

        # 4) 单次 XYZ 测量。SM 的 SM_MODE 位、RM 的 D=2(6 data bytes)构成最强协议指纹。
        try:
            rx = s.transceive_unchecked(bytes([s.CMD_SM | s.AXIS_ALL]), 0)
            st = rx[0] if rx else None
            out["sm"] = st
            if valid_status(st):
                out["score"] += 1
                if st & 0x20:
                    out["score"] += 2
                    out["reason"].append(f"SM=0x{st:02X}(SM_MODE=1)")
                else:
                    out["reason"].append(f"SM=0x{st:02X}")
            time.sleep(delay_s)
            rm = s.transceive_unchecked(bytes([s.CMD_RM | s.AXIS_ALL]), 6)
            out["rm"] = rm.hex(" ") if rm else None
            if len(rm) == 7 and valid_status(rm[0]):
                st = rm[0]
                data = rm[1:]
                if (st & 0x03) == 0x02:
                    out["score"] += 4
                    out["reason"].append(f"RM=0x{st:02X},D=2,XYZ={data.hex(' ')}")
                else:
                    out["score"] += 1
                    out["reason"].append(f"RM=0x{st:02X},D={st & 3}")
                if not (st & 0x10):
                    out["score"] += 1
                if data not in (b"\xFF" * 6, b"\x00" * 6):
                    out["score"] += 1
        except Exception as e:
            out["reason"].append(f"SM/RM异常:{e}")

        score = out["score"]
        # RM(D=2) + RR0 合理配置是“确认”的核心；避免单纯某个随机状态字误判。
        rm_ok = isinstance(out.get("rm"), str) and out["rm"] and out["rm"].split()[0] != "FF"
        hall_ok = out.get("hallconf") in (0x0C, 0x00)
        if score >= 10 and rm_ok and hall_ok:
            out["confidence"] = "CONFIRMED"
        elif score >= 7 and rm_ok:
            out["confidence"] = "PROBABLE"
        elif score >= 4:
            out["confidence"] = "WEAK"
        else:
            out["confidence"] = "NO"
        return out

    def reset(self) -> None:
        self.transceive(bytes([self.CMD_EX]), 0, check_status=False)
        time.sleep(0.001)
        self.transceive(bytes([self.CMD_RT]), 0, check_status=False)
        time.sleep(0.002)
        self.sync_config()
        # Legacy 2015 文档明确给出 T25=46244 LSB、45.2 LSB/°C，但未把 0x24 TREF 作为应用层接口说明。
        # 为避免把新版/未公开的 Melexis 区假设强加到旧硅片，Legacy 模式不主动读取 0x24。
        if self.profile == self.PROFILE_LEGACY_2015:
            self.tref = None
        else:
            try:
                self.tref = self.read_reg(0x24)
            except Exception:
                self.tref = None

    def read_reg(self, reg: int) -> int:
        reg &= 0x3F
        rx = self.transceive(bytes([self.CMD_RR, (reg << 2) & 0xFC]), 2)
        return (rx[1] << 8) | rx[2]

    def write_reg(self, reg: int, value: int) -> None:
        reg &= 0x3F
        value &= 0xFFFF
        self.transceive(
            bytes([self.CMD_WR, (value >> 8) & 0xFF, value & 0xFF, (reg << 2) & 0xFC]),
            0,
        )

    def sync_config(self) -> dict[str, int | bool]:
        r0 = self.read_reg(0x00)
        r1 = self.read_reg(0x01)
        r2 = self.read_reg(0x02)
        self.gain = (r0 >> 4) & 0x7
        self.hallconf = r0 & 0x0F
        self.tcmp = bool((r1 >> 10) & 0x1)
        self.res_x = (r2 >> 5) & 0x3
        self.res_y = (r2 >> 7) & 0x3
        self.res_z = (r2 >> 9) & 0x3
        self.filter = (r2 >> 2) & 0x7
        self.osr = r2 & 0x3
        self.osr2 = (r2 >> 12) & 0x3
        return {
            "reg0": r0,
            "reg1": r1,
            "reg2": r2,
            "gain": self.gain,
            "hallconf": self.hallconf,
            "tcmp": self.tcmp,
            "res_x": self.res_x,
            "res_y": self.res_y,
            "res_z": self.res_z,
            "filter": self.filter,
            "osr": self.osr,
            "osr2": self.osr2,
        }

    def apply_config(
        self,
        gain: int,
        res_x: int,
        res_y: int,
        res_z: int,
        filt: int,
        osr: int,
        osr2: int,
        tcmp: bool,
    ) -> None:
        gain = int(gain)
        res_x, res_y, res_z = int(res_x), int(res_y), int(res_z)
        filt, osr, osr2 = int(filt), int(osr), int(osr2)
        if gain not in range(8):
            raise ValueError("GAIN 必须为 0~7")
        if any(v not in range(4) for v in (res_x, res_y, res_z)):
            raise ValueError("RES X/Y/Z 必须为 0~3")
        if filt not in range(8) or osr not in range(4) or osr2 not in range(4):
            raise ValueError("Filter=0~7，OSR/OSR2=0~3")
        if tcmp and any(v > 1 for v in (res_x, res_y, res_z)):
            raise ValueError("启用温度补偿时，本程序限制 RES X/Y/Z 只能为 0 或 1")

        r0 = self.read_reg(0x00)
        r1 = self.read_reg(0x01)
        r2 = self.read_reg(0x02)

        r0 = (r0 & ~(0x7 << 4)) | ((gain & 0x7) << 4)
        r1 = (r1 & ~(1 << 10)) | ((1 if tcmp else 0) << 10)

        # CONF3：OSR[1:0]、DIG_FILT[4:2]、RES_X[6:5]、RES_Y[8:7]、RES_Z[10:9]、OSR2[13:12]
        clear_mask = (
            0x0003
            | 0x001C
            | (0x3 << 5)
            | (0x3 << 7)
            | (0x3 << 9)
            | (0x3 << 12)
        )
        r2 &= ~clear_mask & 0xFFFF
        r2 |= (osr & 0x3)
        r2 |= (filt & 0x7) << 2
        r2 |= (res_x & 0x3) << 5
        r2 |= (res_y & 0x3) << 7
        r2 |= (res_z & 0x3) << 9
        r2 |= (osr2 & 0x3) << 12

        self.write_reg(0x00, r0)
        self.write_reg(0x01, r1)
        self.write_reg(0x02, r2)
        self.sync_config()

    def _unpack_axis(self, data: bytes, resolution: int) -> int:
        u = struct.unpack(">H", data)[0]
        if resolution == 3:  # 19-bit output window，0 field = 0x4000
            return u - 0x4000
        if resolution == 2:  # 18-bit output window，0 field = 0x8000
            return u - 0x8000
        if self.tcmp:  # TCMP on + RES 0/1 -> unsigned centered at 0x8000
            return u - 0x8000
        return struct.unpack(">h", data)[0]

    def _sens(self, resolution: int, is_z: bool) -> float:
        table = self.SENS_2015_HALLCONF_C if self.profile == self.PROFILE_LEGACY_2015 else self.SENS_25C_HALLCONF_C
        base = table[self.gain][resolution][1 if is_z else 0]
        if self.hallconf == 0x0C:
            return base
        if self.hallconf == 0x00:
            # 2023 文档明确给出 HALLCONF=0 相对 0xC 的 98/75 缩放；Legacy 下仅作为兼容估计。
            return base * (98.0 / 75.0)
        raise MLX90393Error(
            f"当前 HALLCONF=0x{self.hallconf:X}，本程序没有对应灵敏度换算表；可先查看 Raw 数据"
        )

    def estimated_conversion_delay_s(self, include_temp: bool = True) -> float:
        base_ms = self.TCONV_MS[self.filter][self.osr]
        if include_temp and self.osr2 > 0:
            # TCONVT = 67 + 192*2^OSR2 us；表 19 基于 OSR2=0，补上差值
            base_ms += (192.0 * ((2**self.osr2) - 1.0)) / 1000.0
        return (base_ms * 1.18 + 0.25) / 1000.0

    def sample(self, include_temperature: bool = True) -> dict[str, float | int | str | None]:
        mask = self.ALL if include_temperature else self.AXIS_ALL
        sm = self.transceive(bytes([self.CMD_SM | mask]), 0)
        sm_status = sm[0]
        time.sleep(self.estimated_conversion_delay_s(include_temperature))

        data_bytes = 8 if include_temperature else 6
        rm = self.transceive(bytes([self.CMD_RM | mask]), data_bytes)
        status = rm[0]
        p = 1
        temp_raw = None
        temp_c = None
        if include_temperature:
            temp_raw = (rm[p] << 8) | rm[p + 1]
            p += 2
            # 优先使用每颗芯片 Melexis 区 TREF；若不可用则用数据手册 25°C 典型点
            if self.tref is not None and 0x8000 < self.tref < 0xFFFF:
                temp_c = 35.0 + (temp_raw - self.tref) / 45.2
            else:
                temp_c = 25.0 + (temp_raw - 46244.0) / 45.2

        x_raw = self._unpack_axis(rm[p : p + 2], self.res_x)
        p += 2
        y_raw = self._unpack_axis(rm[p : p + 2], self.res_y)
        p += 2
        z_raw = self._unpack_axis(rm[p : p + 2], self.res_z)

        x_ut = x_raw * self._sens(self.res_x, False)
        y_ut = y_raw * self._sens(self.res_y, False)
        z_ut = z_raw * self._sens(self.res_z, True)
        mag_ut = math.sqrt(x_ut * x_ut + y_ut * y_ut + z_ut * z_ut)

        return {
            "timestamp": time.time(),
            "sm_status": sm_status,
            "status": status,
            "status_text": self.decode_status(status),
            "x_raw": x_raw,
            "y_raw": y_raw,
            "z_raw": z_raw,
            "x_ut": x_ut,
            "y_ut": y_ut,
            "z_ut": z_ut,
            "mag_ut": mag_ut,
            "temp_raw": temp_raw,
            "temp_c": temp_c,
        }


# ============================================================
# Tkinter UI 基础组件
# ============================================================


def parse_hex_bytes(text: str) -> bytes:
    s = text.strip().replace(",", " ").replace(";", " ")
    if not s:
        return b""
    out = []
    for token in s.split():
        token = token.strip()
        if token.lower().startswith("0x"):
            token = token[2:]
        if len(token) > 2:
            # 允许连续十六进制，例如 3E4E -> 3E 4E
            if len(token) % 2:
                raise ValueError(f"十六进制串长度应为偶数：{token}")
            out.extend(int(token[i : i + 2], 16) for i in range(0, len(token), 2))
        else:
            out.append(int(token, 16))
    return bytes(out)




# ============================================================
# MagTouch formal acquisition wrappers
# ============================================================

class MultiMLX90393Sampler:
    """Small non-UI wrapper for grouped 5-sensor acquisition.

    Keeps the proven V1.7 sequence:
      SM(S1)..SM(S5) -> one shared conversion wait -> RM(S1)..RM(S5)
    """
    def __init__(self, bridge: CH341Bridge, addresses, profile=MLX90393.PROFILE_LEGACY_2015):
        self.bridge = bridge
        self.addresses = [int(a) for a in addresses]
        self.profile = profile
        self.sensors = [MLX90393(bridge, a, profile=profile) for a in self.addresses]
        self.cycle_index = 0

    def initialize(self, reset=True, config=None):
        """Initialize all sensors and optionally apply one versioned common config.

        config keys: gain,res_x,res_y,res_z,filter,osr,osr2,tcmp.
        The values are written to volatile registers after reset and then read back.
        Existing callers that omit config keep the historical behavior.
        """
        configs=[]
        cfg_req=dict(config or {})
        for i,s in enumerate(self.sensors):
            if reset:
                s.reset()
            else:
                s.sync_config()
            if cfg_req:
                s.apply_config(
                    gain=cfg_req.get("gain",s.gain),
                    res_x=cfg_req.get("res_x",s.res_x),
                    res_y=cfg_req.get("res_y",s.res_y),
                    res_z=cfg_req.get("res_z",s.res_z),
                    filt=cfg_req.get("filter",cfg_req.get("filt",s.filter)),
                    osr=cfg_req.get("osr",s.osr),
                    osr2=cfg_req.get("osr2",s.osr2),
                    tcmp=bool(cfg_req.get("tcmp",s.tcmp)),
                )
            cfg=s.sync_config()
            configs.append({"index":i,"address":s.address,"profile":s.profile,**cfg})
        return configs

    @staticmethod
    def _decode_rm(sensor: MLX90393, rm: bytes, include_temperature: bool, sm_status: int,
                   trigger_ns: int, read_ns: int):
        if not rm:
            raise MLX90393Error("RM returned no status byte")
        status=rm[0]
        sensor._check_status(status,"group RM")
        p=1; temp_raw=None; temp_c=None
        if include_temperature:
            if len(rm)<9: raise MLX90393Error(f"RM(TXYZ) length invalid: {len(rm)}")
            temp_raw=(rm[p]<<8)|rm[p+1]; p+=2
            if sensor.tref is not None and 0x8000 < sensor.tref < 0xFFFF:
                temp_c=35.0+(temp_raw-sensor.tref)/45.2
            else:
                temp_c=25.0+(temp_raw-46244.0)/45.2
        else:
            if len(rm)<7: raise MLX90393Error(f"RM(XYZ) length invalid: {len(rm)}")
        x_raw=sensor._unpack_axis(rm[p:p+2],sensor.res_x); p+=2
        y_raw=sensor._unpack_axis(rm[p:p+2],sensor.res_y); p+=2
        z_raw=sensor._unpack_axis(rm[p:p+2],sensor.res_z)
        x_ut=x_raw*sensor._sens(sensor.res_x,False)
        y_ut=y_raw*sensor._sens(sensor.res_y,False)
        z_ut=z_raw*sensor._sens(sensor.res_z,True)
        return {
            "timestamp":time.time(), "sm_status":sm_status, "status":status,
            "status_text":sensor.decode_status(status),
            "x_raw":x_raw,"y_raw":y_raw,"z_raw":z_raw,
            "x_ut":x_ut,"y_ut":y_ut,"z_ut":z_ut,
            "mag_ut":math.sqrt(x_ut*x_ut+y_ut*y_ut+z_ut*z_ut),
            "temp_raw":temp_raw,"temp_c":temp_c,
            "trigger_ns":trigger_ns,"read_ns":read_ns,
        }

    def group_sample(self, include_temperature=False):
        mask=MLX90393.ALL if include_temperature else MLX90393.AXIS_ALL
        sm_status={}; trig_ns={}; errs={}
        for i,s in enumerate(self.sensors):
            try:
                trig_ns[i]=time.perf_counter_ns()
                rx=s.transceive(bytes([s.CMD_SM|mask]),0)
                sm_status[i]=rx[0]
            except Exception as e:
                errs[i]=f"SM: {e}"
        active=[(i,s) for i,s in enumerate(self.sensors) if i not in errs]
        if not active:
            raise MLX90393Error("All sensor SM commands failed")
        delay=max(s.estimated_conversion_delay_s(include_temperature) for _i,s in active)
        time.sleep(delay)
        out={}; read_ns={}
        data_bytes=8 if include_temperature else 6
        for i,s in active:
            try:
                read_ns[i]=time.perf_counter_ns()
                rm=s.transceive(bytes([s.CMD_RM|mask]),data_bytes)
                out[i]=self._decode_rm(s,rm,include_temperature,sm_status[i],trig_ns[i],read_ns[i])
            except Exception as e:
                errs[i]=f"RM: {e}"
        first=min(trig_ns.values()) if trig_ns else 0
        last=max(trig_ns.values()) if trig_ns else 0
        meta={
            "cycle":self.cycle_index,
            "trigger_span_ms":(last-first)/1e6 if first else 0.0,
            "conversion_wait_ms":delay*1000.0,
            "errors":errs,
            "trigger_base_ns":first,
        }
        self.cycle_index+=1
        return out,meta

    def current_config(self):
        out=[]
        for i,s in enumerate(self.sensors):
            try:
                cfg=s.sync_config()
            except Exception as e:
                cfg={"error":str(e)}
            out.append({"index":i,"address":s.address,"profile":s.profile,**cfg})
        return out


class SimulatedFiveSensorSampler:
    def __init__(self, addresses):
        self.addresses=[int(a) for a in addresses]
        self.cycle_index=0
        self.t0=time.perf_counter()

    def initialize(self, reset=True, config=None):
        c=dict(config or {})
        return [{"index":i,"address":a,"profile":"SIM",
                 "gain":int(c.get("gain",0)),"res_x":int(c.get("res_x",0)),
                 "res_y":int(c.get("res_y",0)),"res_z":int(c.get("res_z",0)),
                 "filter":int(c.get("filter",0)),"osr":int(c.get("osr",0)),
                 "osr2":int(c.get("osr2",0)),"tcmp":bool(c.get("tcmp",False))}
                for i,a in enumerate(self.addresses)]

    def group_sample(self, include_temperature=False):
        base=time.perf_counter_ns(); t=time.perf_counter()-self.t0
        out={}
        for i,a in enumerate(self.addresses):
            phase=i*0.55
            x=110.0*math.sin(1.3*t+phase)+i*25.0
            y=90.0*math.cos(0.9*t+phase)-i*18.0
            z=350.0+60.0*math.sin(0.7*t+phase)
            trig=base+i*180_000
            out[i]={
                "timestamp":time.time(),"sm_status":0,"status":0,"status_text":"SIM",
                "x_raw":int(x),"y_raw":int(y),"z_raw":int(z),
                "x_ut":x,"y_ut":y,"z_ut":z,"mag_ut":math.sqrt(x*x+y*y+z*z),
                "temp_raw":None,"temp_c":25.0 if include_temperature else None,
                "trigger_ns":trig,"read_ns":trig+2_000_000,
            }
        meta={"cycle":self.cycle_index,"trigger_span_ms":0.18*(len(self.addresses)-1),
              "conversion_wait_ms":2.0,"errors":{},"trigger_base_ns":base}
        self.cycle_index+=1
        return out,meta

    def current_config(self):
        return self.initialize(reset=False)
