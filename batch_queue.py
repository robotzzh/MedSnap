# -*- coding: utf-8 -*-
"""
MedSnap 批量上传消费队列模块

设计说明:
  - 每个批次最多 100 个文件，超出则拒绝
  - 使用 threading.Thread 后台线程消费队列
  - 每个批次有唯一 batch_id，可通过接口查询进度
  - 队列中的任务按顺序串行处理，避免并发打爆 AI API
"""

import os
import uuid
import threading
from datetime import datetime
from queue import Queue

# ========== 常量 ==========

MAX_FILES_PER_BATCH = 100

# ========== 全局状态 ==========

# 任务队列：存放待处理的 batch_task dict
_task_queue = Queue()

# 批次状态表：batch_id -> BatchStatus
# 结构：{
#   "batch_id": str,
#   "total": int,
#   "pending": int,
#   "success": int,
#   "failed": int,
#   "status": "queued" | "processing" | "done",
#   "results": [...],
#   "errors": [...],
#   "create_time": str,
#   "finish_time": str | None,
# }
_batch_status_store: dict[str, dict] = {}
_store_lock = threading.Lock()

# ========== 状态管理工具 ==========

def _get_batch(batch_id: str) -> dict | None:
    with _store_lock:
        return _batch_status_store.get(batch_id)

def _update_batch(batch_id: str, **kwargs):
    with _store_lock:
        if batch_id in _batch_status_store:
            _batch_status_store[batch_id].update(kwargs)

def _append_result(batch_id: str, result: dict):
    with _store_lock:
        if batch_id in _batch_status_store:
            _batch_status_store[batch_id]['results'].append(result)

def _append_error(batch_id: str, error: dict):
    with _store_lock:
        if batch_id in _batch_status_store:
            _batch_status_store[batch_id]['errors'].append(error)

# ========== 后台消费线程 ==========

def _worker():
    """后台消费线程，持续从队列取任务并处理。"""
    while True:
        batch_task = _task_queue.get()
        batch_id = batch_task['batch_id']
        file_tasks = batch_task['file_tasks']
        processor = batch_task['processor']

        _update_batch(batch_id, status='processing')

        for file_task in file_tasks:
            try:
                result = processor(file_task)
                _append_result(batch_id, result)
                with _store_lock:
                    _batch_status_store[batch_id]['success'] += 1
                    _batch_status_store[batch_id]['pending'] -= 1
            except Exception as error:
                _append_error(batch_id, {
                    'filename': file_task.get('filename', 'unknown'),
                    'error': str(error),
                })
                with _store_lock:
                    _batch_status_store[batch_id]['failed'] += 1
                    _batch_status_store[batch_id]['pending'] -= 1

        _update_batch(
            batch_id,
            status='done',
            finish_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        )
        _task_queue.task_done()


# 启动后台消费线程（daemon=True 保证主进程退出时自动结束）
_consumer_thread = threading.Thread(target=_worker, daemon=True)
_consumer_thread.start()

# ========== 公开 API ==========

def submit_batch(file_tasks: list[dict], processor) -> tuple[str, str]:
    """
    提交一个批次任务到队列。

    Args:
        file_tasks: 文件任务列表，每项是传给 processor 的参数 dict，
                    必须包含 'filename' 字段用于错误追踪。
        processor:  处理单个文件任务的函数，接收 file_task dict，返回结果 dict。

    Returns:
        (batch_id, error_message)
        成功时 error_message 为空字符串，失败时 batch_id 为空字符串。
    """
    file_count = len(file_tasks)
    if file_count == 0:
        return '', '文件列表不能为空'
    if file_count > MAX_FILES_PER_BATCH:
        return '', f'每批次最多 {MAX_FILES_PER_BATCH} 个文件，当前提交了 {file_count} 个'

    batch_id = uuid.uuid4().hex
    with _store_lock:
        _batch_status_store[batch_id] = {
            'batch_id': batch_id,
            'total': file_count,
            'pending': file_count,
            'success': 0,
            'failed': 0,
            'status': 'queued',
            'results': [],
            'errors': [],
            'create_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'finish_time': None,
        }

    _task_queue.put({
        'batch_id': batch_id,
        'file_tasks': file_tasks,
        'processor': processor,
    })

    return batch_id, ''


def get_batch_status(batch_id: str) -> dict | None:
    """
    查询批次状态。

    Returns:
        批次状态 dict，不存在时返回 None。
    """
    return _get_batch(batch_id)


def list_all_batches() -> list[dict]:
    """返回所有批次的摘要列表（不含 results 详情，避免数据过大）。"""
    with _store_lock:
        summaries = []
        for batch in _batch_status_store.values():
            summaries.append({
                'batch_id': batch['batch_id'],
                'total': batch['total'],
                'pending': batch['pending'],
                'success': batch['success'],
                'failed': batch['failed'],
                'status': batch['status'],
                'create_time': batch['create_time'],
                'finish_time': batch['finish_time'],
            })
        return sorted(summaries, key=lambda x: x['create_time'], reverse=True)
