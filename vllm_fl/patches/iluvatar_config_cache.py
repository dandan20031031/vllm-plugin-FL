# Copyright 2026 FlagOS Contributors
# Copyright 2026 eggking
# SPDX-License-Identifier: Apache-2.0
"""Bounded SQLite mapper optimization for the frozen Iluvatar FlagGems build.

The SQL lookup/write and Triton configuration values are unchanged. Only the
set of reflected tables and the key of an already-created ORM model change.
The get_sql_model body is adapted from flag_gems/utils/models/sql.py at the
official a7620cc revision; its Apache-2.0 attribution is retained above.
"""
from __future__ import annotations

from hashlib import md5, sha256
import inspect
import logging
from pathlib import Path
import threading
import time


logger = logging.getLogger(__name__)
_SQL_SOURCE_SHA256 = '6dc19cb581d70154f14b95ed9112d682c6b5919b6cf3f8dea01a78a75ec0efd7'
_registration_lock = threading.RLock()
_sql_class = None
_original_get = None
_original_mapper = None
_stats = dict(pool_hits=0, mapping_calls=0, mapping_ns=0, created_tables=0,
              non_sqlite_fallbacks=0)


def _build_sql_model_by_db(name, engine):
    if engine.dialect.name != 'sqlite':
        return _original_mapper(name, engine)
    import sqlalchemy.ext.automap

    started = time.perf_counter_ns()
    base = sqlalchemy.ext.automap.automap_base()
    # A callable filter keeps a missing table as None. Passing only=[name]
    # instead would raise when tuning needs to create a new cache table.
    base.prepare(autoload_with=engine,
                 reflection_options={'only': lambda candidate, _: candidate == name})
    model = base.classes.get(name)
    _stats['mapping_calls'] += 1
    _stats['mapping_ns'] += time.perf_counter_ns() - started
    if _stats['mapping_calls'] == 1:
        logger.warning('eggking config mapper dispatched: reflected=%d target_found=%s',
                       len(base.metadata.tables), model is not None)
    return model


def _get_sql_model(self, name, keys={}, values={}):
    # Other database dialects preserve the original implementation, including
    # PostgreSQL table-name truncation and error handling.
    if self.engine.dialect.name != 'sqlite':
        _stats['non_sqlite_fallbacks'] += 1
        return _original_get(self, name, keys, values)
    import sqlalchemy

    with self.lock:
        full_name = '{}-{}'.format(name, md5(''.join(keys.keys()).encode()).hexdigest())
        model = self.sql_model_pool.get(full_name)
        if model is not None:
            _stats['pool_hits'] += 1
            return model
        model = _sql_class.build_sql_model_by_db(full_name, self.engine)
        if model is not None:
            self.sql_model_pool[full_name] = model
            return model
        if not keys or not values:
            return None
        model = _sql_class.build_sql_model_by_py(full_name, keys, values)
        try:
            with self.engine.begin() as conn:
                conn.execute(sqlalchemy.schema.CreateTable(model.__table__, if_not_exists=True))
        except Exception as exc:
            message = str(exc)
            if 'duplicate key value violates unique constraint' not in message:
                if 'type "' not in message or '" already exists' not in message:
                    raise
        # The original stores pool[name] but subsequently reads pool[full_name],
        # causing another complete reflection after a successful table creation.
        self.sql_model_pool[full_name] = model
        _stats['created_tables'] += 1
        return model


_get_sql_model._eggking_targeted_sqlite_mapper = True
_build_sql_model_by_db._eggking_targeted_sqlite_mapper = True


def _install_for_class(sql_class):
    """Install once; the caller checks backend, dependency and source identity."""
    global _sql_class, _original_get, _original_mapper
    with _registration_lock:
        current_get = sql_class.__dict__['get_sql_model']
        current_mapper = sql_class.__dict__['build_sql_model_by_db'].__func__
        if current_get is _get_sql_model and current_mapper is _build_sql_model_by_db:
            return False
        if _sql_class is not None:
            raise RuntimeError('SQL mapper registration conflicts with another patch')
        _sql_class = sql_class
        _original_get = current_get
        _original_mapper = current_mapper
        sql_class.build_sql_model_by_db = staticmethod(_build_sql_model_by_db)
        sql_class.get_sql_model = _get_sql_model
        return True


def mapper_stats():
    """CPU diagnostics only; this is not a timed-request kernel trace."""
    return dict(_stats)


def patch_iluvatar_config_cache():
    """Apply only to the verified SQLite implementation; fail closed otherwise."""
    import sqlalchemy
    from flag_gems.runtime.backend import _state
    from flag_gems.utils.models.sql import SQLPersistantModel

    if _state.vendor_module.vendor_info.vendor_name != 'iluvatar':
        return False
    if sqlalchemy.__version__.split('.')[:2] != ['2', '0']:
        logger.warning('eggking config mapper skipped: unsupported SQLAlchemy version')
        return False
    source = Path(inspect.getfile(SQLPersistantModel))
    if sha256(source.read_bytes()).hexdigest() != _SQL_SOURCE_SHA256:
        logger.warning('eggking config mapper skipped: FlagGems source fingerprint differs')
        return False
    with _registration_lock:
        if _sql_class is SQLPersistantModel:
            return _install_for_class(SQLPersistantModel)
        original_get = SQLPersistantModel.__dict__['get_sql_model']
        original_map = SQLPersistantModel.__dict__['build_sql_model_by_db'].__func__
        if any(Path(fn.__code__.co_filename).resolve() != source.resolve()
               for fn in [original_get, original_map]):
            logger.warning('eggking config mapper skipped: an unknown patch owns the methods')
            return False
        changed = _install_for_class(SQLPersistantModel)
    if changed:
        logger.warning('eggking config mapper registered: SQLite target-table reflection; '
                       'configuration values are read from the original database')
    return changed
