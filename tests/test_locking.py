"""跨进程共享原语的快速定向测试。

两层验证互补：

- :mod:`tests.test_shared_dir_two_processes` 启动两个真实 uvicorn 进程做
  端到端 HTTP 验证（慢、覆盖面广）；
- 本文件在单进程内为同一数据目录构造**两个互相独立**的存储对象（各自独立
  的锁文件描述符与内存缓存）。POSIX ``flock`` 锁绑定在“打开文件描述”
  （open file description）上：同一进程对同一锁文件的两次独立 ``open``
  得到的描述符**仍然互相争用**，因此两个存储对象在锁与缓存同步上的行为
  与两个独立进程一致，可快速验证：
  - 排他锁确实阻塞另一把描述符（``LOCK_NB`` 立即失败），共享锁互不阻塞；
  - 一个存储创建/发布后，另一个存储（缓存陈旧）立即快照得到新内容；
  - 两个存储基于同一 expected_revision 并发修订时恰有一方 409；
  - 交错创建不会让任何一方把对方的工作单从落盘文件中覆盖掉。
"""

from __future__ import annotations

import fcntl
import os
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal

import pytest

from app import revisions, unit_sets
from app.revisions import RevisionConflictError, VariableChange
from app.schemas import CustomUnitItem
from app.worksheet import run_worksheet


# ---------------------------------------------------------------------------
# FileLock 原语
# ---------------------------------------------------------------------------


def test_exclusive_flock_blocks_independent_handle(tmp_path):
    lock_path = str(tmp_path / "x.lock")
    from app.locking import FileLock

    primary = FileLock(lock_path)
    other_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        with primary.exclusive():
            # 另一把独立打开的描述符以非阻塞方式申请排他锁：立即失败
            with pytest.raises(BlockingIOError):
                fcntl.flock(other_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # 同进程内另一 FileLock 对象申请共享锁同样被拒
            from app.locking import FileLock as _FL

            second = _FL(lock_path)
            second_fd = second._ensure_fd()
            with pytest.raises(BlockingIOError):
                fcntl.flock(second_fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            second.close()
        # 释放后另一描述符可立即取得锁
        fcntl.flock(other_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(other_fd, fcntl.LOCK_UN)
    finally:
        os.close(other_fd)
        primary.close()


def test_shared_flock_admits_concurrent_readers(tmp_path):
    from app.locking import FileLock

    first = FileLock(str(tmp_path / "r.lock"))
    second = FileLock(str(tmp_path / "r.lock"))
    other_fd = os.open(str(tmp_path / "r.lock"), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        with first.shared(), second.shared():
            # 第三个读者（独立描述符、非阻塞）同样立即获得共享锁
            fcntl.flock(other_fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            # 但写者必须等待
            writer_fd = os.open(str(tmp_path / "r.lock"),
                                os.O_RDWR | os.O_CREAT, 0o600)
            try:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(writer_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                os.close(writer_fd)
            fcntl.flock(other_fd, fcntl.LOCK_UN)
    finally:
        os.close(other_fd)
        first.close()
        second.close()


# ---------------------------------------------------------------------------
# 两个独立存储对象共享同一数据目录（等价于两个进程的缓存视图）
# ---------------------------------------------------------------------------


@pytest.fixture
def shared_dir():
    # autouse 的 _isolated_stores 已在每用例临时目录初始化全局单例；
    # 这里取其数据目录并构造第二个独立存储对象。
    data_dir = os.path.dirname(revisions.get_store().path)
    yield data_dir


def _publish_set() -> str:
    item = CustomUnitItem(symbol="ft", factor=Decimal("0.3048"), base_unit="m")
    return unit_sets.get_store().publish([item])


def _recalculate(req):
    scope = unit_sets.get_store().build_scope(req.custom_unit_set_id)
    return run_worksheet(req, scope)


def _standard_body(set_id):
    from app.schemas import WorksheetRequest

    return WorksheetRequest.model_validate({
        "variables": {
            "v": {"value": "60", "unit": "km/h"},
            "t": {"value": "2", "unit": "h"},
        },
        "steps": [
            {"name": "s", "formula": "v*t", "output_unit": "km"},
            {"name": "d", "equation": "v*th = s", "solve_for": "th",
             "output_unit": "h"},
        ],
        "custom_unit_set_id": set_id,
    })


def test_second_revision_store_sees_worksheet_created_by_first(shared_dir):
    sid = _publish_set()
    first = revisions.RevisionStore(shared_dir)
    first.load()
    second = revisions.RevisionStore(shared_dir)
    second.load()
    try:
        body = _standard_body(sid)
        wid, record = first.create(body, sid, _recalculate(body))
        assert record["revision"] == 1

        # second 的缓存只含启动时的空状态；snapshot 经共享锁 + 磁盘签名同步
        seen = second.snapshot(wid)
        assert seen["revision"] == 1
        assert [r["name"] for r in seen["results"]] == ["s", "d"]
        assert seen["results"][0]["value"] == "120"

        # 反向：second 修订后 first 也立即读到新修订
        second.revise(
            wid, 1, VariableChange("t", Decimal("3"), "h"), _recalculate
        )
        assert first.snapshot(wid)["revision"] == 2
        assert second.snapshot(wid)["results"][0]["value"] == "180"
    finally:
        first.close()
        second.close()


def test_two_stores_same_expected_revision_only_one_revise_wins(shared_dir):
    sid = _publish_set()
    first = revisions.RevisionStore(shared_dir)
    first.load()
    second = revisions.RevisionStore(shared_dir)
    second.load()
    try:
        body = _standard_body(sid)
        wid, _ = first.create(body, sid, _recalculate(body))

        outcomes: list[object] = []

        def attempt(store, value):
            try:
                store.revise(
                    wid, 1, VariableChange("t", Decimal(str(value)), "h"),
                    _recalculate,
                )
                return "ok"
            except RevisionConflictError as exc:
                return exc.current_revision

        with ThreadPoolExecutor(2) as pool:
            futures = [
                pool.submit(attempt, first, 5),
                pool.submit(attempt, second, 9),
            ]
            outcomes = [f.result() for f in futures]

        assert outcomes.count("ok") == 1, outcomes
        assert outcomes.count(2) == 1, outcomes  # 落选方拿到最新修订号 2
        assert first.snapshot(wid)["revision"] == 2
        assert second.snapshot(wid)["revision"] == 2
    finally:
        first.close()
        second.close()


def test_interleaved_creates_between_two_stores_never_clobber(shared_dir):
    sid = _publish_set()
    first = revisions.RevisionStore(shared_dir)
    first.load()
    second = revisions.RevisionStore(shared_dir)
    second.load()
    try:
        body = _standard_body(sid)
        seed_id, _ = first.create(body, sid, _recalculate(body))
        # second 先“看看”种子单，此后它的缓存里有种子单
        assert second.snapshot(seed_id)["revision"] == 1

        created = []
        for store in (first, second, first, second):
            wid, _ = store.create(_standard_body(sid), sid, _recalculate(body))
            created.append(wid)

        # 每次落盘都必须包含此前全部工作单
        for store in (first, second):
            for wid in [seed_id, *created]:
                assert store.snapshot(wid)["revision"] == 1

        # 磁盘文件同样完整
        import json

        with open(os.path.join(shared_dir, revisions.STORE_FILENAME),
                  encoding="utf-8") as f:
            persisted = json.load(f)
        assert {seed_id, *created} <= set(persisted["sheets"])
    finally:
        first.close()
        second.close()


def test_second_unit_set_store_sees_publish_and_alias_switch(shared_dir):
    first = unit_sets.UnitSetStore(shared_dir)
    first.load()
    second = unit_sets.UnitSetStore(shared_dir)
    second.load()
    try:
        new_id = first.publish([
            CustomUnitItem(symbol="span", factor=Decimal("2.5"), base_unit="m")
        ])
        # second 的缓存为空，却能立即取到 first 刚发布的定义
        items = second.get(new_id)
        assert items[0].symbol == "span"

        # second 切换别名后 first 立即按新指向解析（两进程视图一致）
        assert second.bind_alias("lengths", new_id, None) == new_id
        assert first.get_alias("lengths") == new_id

        # first 以过期 expected_id 切换 -> 409（CAS 针对的是最新磁盘指向）
        from app.errors import AliasConflictError

        another = first.publish([
            CustomUnitItem(symbol="cube", factor=Decimal("3"), base_unit="m")
        ])
        with pytest.raises(AliasConflictError):
            first.bind_alias("lengths", another, None)
    finally:
        first.close()
        second.close()
