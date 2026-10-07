"""跨进程文件锁与磁盘状态签名。

两个（或更多）API 进程可以把数据目录（``UNIT_API_DATA_DIR``）指向同一个
共享目录：已发布定义集与可修订工作单各自仍是目录下的一个 JSON 文件，
文件格式与单进程时代完全一致；本模块在 JSON 文件之外额外提供：

- :func:`signature`：数据文件的轻量签名（inode、mtime 纳秒、大小）。
  进程把内存缓存与“上次读取/写入时的签名”绑定，签名未变即跳过读盘，
  签名变化（由其他进程写入并原子替换造成）才重新加载，因此一方的提交
  对另一方**下一次操作**立即可见，空闲读几乎零开销；
- :func:`locked`：基于 ``fcntl.flock`` 的跨进程互斥/共享锁，锁对象是数据
  文件旁边的一个零字节锁文件（如 ``revisable_worksheets.json.lock``）。
  写操作取排他锁（``LOCK_EX``）、读操作取共享锁（``LOCK_SH``）：
  跨进程的“重读最新快照 → CAS 判定 → 整单重算 → 全量原子落盘”因此被
  串行化为同一份磁盘状态上的事务，旧内存缓存不可能覆盖其他进程的新提交；
  同进程的不同线程各开独立文件描述符，锁同样互斥，与存储内的可重入锁
  共同保证内存缓存一致。

锁仅用于进程间互斥，不进入 JSON 存档：删除锁文件不影响任何已保存数据
（它会在下次操作时自动重建）。``flock`` 随文件描述符关闭而自动释放，
进程异常退出也不会留下被持有的锁。
"""

from __future__ import annotations

import fcntl
import os
from contextlib import contextmanager
from typing import Iterator


def signature(path: str) -> tuple[int, int, int] | None:
    """返回数据文件当前的签名 ``(inode, mtime_ns, size)``；文件不存在返回 None。

    原子替换（``os.replace``）会更换 inode 与 mtime，原地写入至少改变
    mtime/size，因此任何一次成功提交都必然改变签名。
    """

    try:
        st = os.stat(path)
    except FileNotFoundError:
        return None
    return (st.st_ino, st.st_mtime_ns, st.st_size)


def lock_path_for(data_path: str) -> str:
    """数据文件对应的锁文件路径（与其同目录、仅多 ``.lock`` 后缀）。"""

    return f"{data_path}.lock"


@contextmanager
def locked(data_path: str, *, exclusive: bool) -> Iterator[None]:
    """持有 ``data_path`` 对应锁文件上的跨进程锁进入临界区。

    ``exclusive=True`` 为写锁（与任何共享/排他请求互斥）；``False`` 为读锁
    （与其他读锁共享、与写锁互斥）。每次进入都重新打开锁文件：不同进程、
    同进程不同线程因此分别持有各自的 open file description，``flock`` 对
    它们同样生效。锁文件的父目录（即数据目录）会在加锁前确保存在。
    """

    lock_path = lock_path_for(data_path)
    os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
    fd = open(lock_path, "a+")
    try:
        fcntl.flock(fd.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        yield
    finally:
        try:
            fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
        finally:
            fd.close()
