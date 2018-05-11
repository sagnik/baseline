import baseline
import json
import numpy as np
import logging
import logging.config
import mead.utils
import os
from mead.downloader import EmbeddingDownloader, DataDownloader, read_json
from mead.mime_type import mime_type
from baseline.utils import export, import_user_module
import datetime
import hashlib
import socket
import getpass
import portalocker
import uuid
from bson.objectid import ObjectId


__all__ = []
exporter = export(__all__)


def _time_now():
    return datetime.datetime.utcnow().isoformat()


EVENT_TYPES = {
    "train": "train_events", "Train": "train_events",
    "test": "test_events", "Test": "test_events",
    "valid": "valid_events", "Valid": "valid_events",
    "dev": "valid_events", "Dev": "valid_events"
}

@exporter
class EventReporting(object):

    def __init__(self):
        pass

    def create_experiment(self, task, config_obj, **kwargs):
        """Create the initial experiment, with no events attached

        :param task:
        :param config_obj:
        :param kwargs: See below

        :Keyword Arguments:
        * *print_fn* -- A print callback which takes a ``str`` argument
        * *hostname* -- (``str``) A hostname, defaults to name of the local machine
        * *username* -- (``str``) A username, defaults to the name of the user on this machine
        * *label* -- (``str``) An optional, human-readable label name.  Defaults to sha1 of this configuration

        :return: The identifier assigned by the database
        """
        pass

    def update_experiment(self, id, task, metrics, tick, phase, tick_type=None):
        pass

    def close_experiment(self, id, task, status_event):
        pass


@exporter
class StreamingJSONReporting(EventReporting):

    def __init__(self, filename):
        super(StreamingJSONReporting, self).__init__()
        self.handle = open(filename, "a", 1)

    def _write_json(self, msg):
        portalocker.lock(self.handle, portalocker.LOCK_EX)
        self.handle.write(json.dumps(msg) + '\n')
        portalocker.unlock(self.handle)

    def close_experiment(self, id, task, status_event):
        msg = {'event_type': 'CLOSED', 'id': id, 'date': _time_now(), 'status': status_event}
        self._write_json(msg)
        self.handle.close()

    def create_experiment(self, task_type, config, **kwargs):

        label = kwargs.get('label', os.getpid())

        hostname = kwargs.get('hostname', socket.gethostname())
        username = kwargs.get('username', getpass.getuser())

        config_sha1 = hashlib.sha1(json.dumps(config).encode('utf-8')).hexdigest()
        id = str(uuid.uuid4())
        msg = {
            'event_type': 'CREATED',
            'task_type': task_type,
            'config': config,
            'label': label,
            'username': username,
            'hostname': hostname,
            'date': _time_now(),
            'id': id,
            'config_sha1': config_sha1
        }
        self._write_json(msg)
        return id

    def update_experiment(self, id, task, metrics, tick, phase, tick_type=None):
        """Write a streaming JSON line to the handle file

        :param metrics: A map of metrics to scores
        :param tick: The time (resolution defined by `tick_type`)
        :param phase: The phase of training (`Train`, `Valid`, `Test`)
        :param tick_type: The resolution of tick (`STEP`, `EPOCH`)
        :return:
        """

        if tick_type is None:
            tick_type = 'STEP'
            if phase in ['Valid', 'Test']:
                tick_type = 'EPOCH'

        msg = {'event_type': 'TICK', 'id': id, 'date': _time_now(), 'tick_type': tick_type, 'tick': tick, 'phase': phase }
        for k, v in metrics.items():
            msg[k] = v

        self._write_json(msg)


@exporter
class MongoEventReporting(EventReporting):

    def __init__(self, host, port, user, passw):
        import pymongo
        super(MongoEventReporting, self).__init__()
        self.dbhost = host
        if user and passw:
            uri = "mongodb://{}:{}@{}:{}/test".format(user, passw, host, port)
            client = pymongo.MongoClient(uri)
        else:
            client = pymongo.MongoClient(host, port)
        if client is None:
            s = "can not connect to mongo at host: [{}], port [{}], username: [{}], password: [{}]".format(host,
                                                                                                           port,
                                                                                                           user,
                                                                                                           passw)
            raise Exception(s)
        try:
            dbnames = client.database_names()
        except pymongo.errors.ServerSelectionTimeoutError:
            raise Exception("can not get database from mongo at host: {}, port {}, connection timed out".format(host,
                                                                                                                port))

        if "reporting_db" not in dbnames:
            raise Exception("no database for results found")
        self.db = client.reporting_db

    def create_experiment(self, task, config_obj, **kwargs):
        now = datetime.datetime.utcnow().isoformat()
        hostname = kwargs.get('hostname', socket.gethostname())
        username = kwargs.get('username', getpass.getuser())
        config_sha1 = hashlib.sha1(json.dumps(config_obj).encode('utf-8')).hexdigest()
        label = kwargs.get("label", config_sha1)
        post = {
            "config": config_obj,
            "train_events": [],
            "valid_events": [],
            "test_events": [],
            "username": username,
            "hostname": hostname,
            "date": now,
            "label": label,
            "sha1": config_sha1,
            "version": baseline.__version__,
            "status": "CREATED",
            "last_updated": now
        }

        coll = self.db[task]
        result = coll.insert_one(post)
        return result.inserted_id

    def update_experiment(self, id, task, metrics, tick, phase, tick_type=None):
        coll = self.db[task]
        now = datetime.datetime.utcnow().isoformat()
        if tick_type is None:
            tick_type = 'STEP'
        if phase in ['Valid', 'Test']:
            tick_type = 'EPOCH'
        entry = {
            "tick": tick,
            "phase": phase,
            "tick_type": tick_type
        }
        for k, v in metrics.items():
            entry[k] = v

        array_to_append = EVENT_TYPES[phase]
        update = {"$push": {array_to_append: entry}, "$set": {"last_updated": now, "status": "RUNNING"}}

        coll.update_one({'_id': ObjectId(id)}, update)

    def close_experiment(self, id, task, status_event):
        coll = self.db[task]
        now = datetime.datetime.utcnow().isoformat()
        update = {"$set": {"status": status_event, "last_updated": now}}
        print('CLOSING', task, update)
        coll.update_one({'_id': ObjectId(id)}, update)
