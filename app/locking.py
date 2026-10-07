"""跨进程共享数据目录用的文件锁与磁盘签名工具。

当两个 API 进程（见 ``docker-compose.dual.yml``）把 ``UNIT_API_DATA_DIR``
指向同一个共享卷时，仅靠进程内的 :class:`threading.RLock` 无法串行化跨进程
写入：两个进程可能各自基于启动时加载的 JSON 快照做“读-改-全量写回”，后写
者就会用自己的旧缓存覆盖先写者刚刚创建/修订的工作单或刚发布的定义集。

本模块在数据文件之外提供一把“每个存储一把”的 :func:`fcntl.flock` 互斥锁：

- **写事务**持共享锁文件上的排他锁（``LOCK_EX``）：进入临界区后先按磁盘
  文件签名重新加载，再在最新状态上修改、原子落盘，因此提交的永远是
  “最新磁盘版本 + 本次修改”，不会用旧缓存覆盖别的进程（或别的进程之后
  本进程未见过）的内容；
- **读事务**持共享锁（``LOCK_SH``）：同样先按磁盘签名同步再返回快照；
  写事务进行期间读者在锁上等待，既不会读到半写文件，也不会返回本进程
  的陈旧缓存；一方提交后，另一方下一次读取即可立即见到；
- 锁绑定在一个长期打开的文件描述符（open file description）上，进程内再用
  :class:`threading.RLock` 串行化同进程的线程：避免一个线程正持锁时，另一
  线程关闭同一文件描述符而把锁提前释放（``flock`` 锁在关联的打开文件描述
  全部关闭时释放）。同线程重入按深度计数，同模式重入不重复加锁。

锁文件（``.<数据文件名>.lock``）本身不承载数据，仅用于互斥；可以长期存在
于共享卷上。数据文件被原子替换后 inode 必然改变，:func:`file_signature`
据此让存储层识别出“磁盘已有新版本”。
"""

from __future__ import annotations

import fcntl
import os
import threading
from contextlib import contextmanager
from typing import Iterator


class FileLock:
    """单个锁文件上的跨进程 ``flock``（进程内由 ``RLock`` 串行）。

    使用方式::

        with lock.exclusive():   # 或 lock.shared()
            ... 临界区 ...

    锁文件描述符在首次使用时打开并长期复用（每个进程、每个存储仅一个
    fd）；存储被丢弃前可调用 :meth:`close` 释放（测试中每个用例使用独立
    数据目录时需要）。
    """

    def __init__(self, lock_path: str) -> None:
        self._lock_path = lock_path
        self._rlock = threading.RLock()
        self._fd: int | None = None
        # 仅在持有 self._rlock 时访问以下两项：同线程重入深度与各层锁模式。
        self._depth = 0
        self._mode: int | None = None

    def _ensure_fd(self) -> int:
        if self._fd is None:
            directory = os.path.dirname(self._lock_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            self._fd = os.open(self._lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        return self._fd

    def _acquire(self, mode: int) -> None:
        """调用方须持有 self._rlock。"""

        fd = self._ensure_fd()
        if self._depth == 0:
            fcntl.flock(fd, mode)
            self._mode = mode
        elif self._mode == fcntl.LOCK_SH and mode == fcntl.LOCK_EX:
            # 同线程在同一 fd 上由共享升级为排他（实际调用路径不出现，
            # 这里仅保证升级语义不把锁提前释放）。
            fcntl.flock(fd, fcntl.LOCK_EX)
            self._mode = fcntl.LOCK_EX
        # 其余重入（EX 内再要 EX/SH、SH 内再要 SH）：外层锁已足够强，
        # 不重复加锁，避免在新打开文件描述上自阻塞。
        self._depth += 1

    def _release(self) -> None:
        """调用方须持有 self._rlock。"""

        self._depth -= 1
        if self._depth > 0:
            return
        fd = self._fd
        assert fd is not None
        fcntl.flock(fd, fcntl.LOCK_UN)
        self._mode = None

    @contextmanager
    def shared(self) -> Iterator[None]:
        """获取跨进程共享锁（读者之间不互斥，与任一写者互斥）。"""

        with self._rlock:
            self._acquire(fcntl.LOCK_SH)
            try:
                yield
            finally:
                self._release()

    @contextmanager
    def exclusive(self) -> Iterator[None]:
        """获取跨进程排他锁（与任何其他读者/写者互斥）。"""

        with self._rlock:
            self._acquire(fcntl.LOCK_EX)
            try:
                yield
            finally:
                self._release()

    def close(self) -> None:
        """释放长期打开的锁文件描述符（仅存储被丢弃前使用）。"""

        with self._rlock:
            if self._fd is None:
                return
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(self._fd)
            self._fd = None
            self._depth = 0
            self._mode = None


def file_signature(path: str) -> tuple[int, int, int, int] | None:
    """返回数据文件当前的磁盘签名；文件不存在时返回 ``None``。

    签名为 ``(设备号, inode, 纳秒级修改时间, 字节数)``。两个存储都以
    “临时文件 + :func:`os.replace` 原子替换”落盘，替换后 inode 必然改变，
    因此即使两次写入的大小与修改时间戳碰巧相同，也不会把新版本误判为
    旧缓存；文件被删除后重建（新 inode）同样能被识别。
    """

    try:
        st = os.stat(path)
    except FileNotFoundError:
        return None
    return (st.st_dev, st.st_ino, st.st_mtime_ns, st.st_size)
