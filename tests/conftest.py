"""pytest 公共夹具。

已发布定义集与可修订工作单都带磁盘持久化（app/unit_sets.py、
app/revisions.py），测试中把两个进程级存储单例重定向到每个用例独立的
临时目录，既避免污染开发机上的默认数据目录，也保证用例之间互不干扰。
"""

from __future__ import annotations

import pytest

from app import revisions, unit_sets


@pytest.fixture(autouse=True)
def _isolated_stores(tmp_path):
    data_dir = str(tmp_path / "data")
    store = unit_sets.UnitSetStore(data_dir)
    store.load()
    unit_sets._store = store
    revision_store = revisions.RevisionStore(data_dir)
    revision_store.load()
    revisions._store = revision_store
    yield store
    # 显式关闭每用例独立数据目录上的 flock 文件描述符，避免跨用例泄漏
    store.close()
    revision_store.close()
    unit_sets._store = None
    revisions._store = None
