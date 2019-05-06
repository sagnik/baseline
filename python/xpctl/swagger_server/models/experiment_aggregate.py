# coding: utf-8

from __future__ import absolute_import
from datetime import date, datetime  # noqa: F401

from typing import List, Dict  # noqa: F401

from swagger_server.models.base_model_ import Model
from swagger_server.models.aggregate_result import AggregateResult  # noqa: F401,E501
from swagger_server import util


class ExperimentAggregate(Model):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """

    def __init__(self, task: str=None, eid: str=None, sha1: str=None, config: str=None, dataset: str=None, username: str=None, hostname: str=None, exp_date: date=None, label: str=None, version: str=None, num_exps: int=None, train_events: List[AggregateResult]=None, valid_events: List[AggregateResult]=None, test_events: List[AggregateResult]=None):  # noqa: E501
        """ExperimentAggregate - a model defined in Swagger

        :param task: The task of this ExperimentAggregate.  # noqa: E501
        :type task: str
        :param eid: The eid of this ExperimentAggregate.  # noqa: E501
        :type eid: str
        :param sha1: The sha1 of this ExperimentAggregate.  # noqa: E501
        :type sha1: str
        :param config: The config of this ExperimentAggregate.  # noqa: E501
        :type config: str
        :param dataset: The dataset of this ExperimentAggregate.  # noqa: E501
        :type dataset: str
        :param username: The username of this ExperimentAggregate.  # noqa: E501
        :type username: str
        :param hostname: The hostname of this ExperimentAggregate.  # noqa: E501
        :type hostname: str
        :param exp_date: The exp_date of this ExperimentAggregate.  # noqa: E501
        :type exp_date: date
        :param label: The label of this ExperimentAggregate.  # noqa: E501
        :type label: str
        :param version: The version of this ExperimentAggregate.  # noqa: E501
        :type version: str
        :param num_exps: The num_exps of this ExperimentAggregate.  # noqa: E501
        :type num_exps: int
        :param train_events: The train_events of this ExperimentAggregate.  # noqa: E501
        :type train_events: List[AggregateResult]
        :param valid_events: The valid_events of this ExperimentAggregate.  # noqa: E501
        :type valid_events: List[AggregateResult]
        :param test_events: The test_events of this ExperimentAggregate.  # noqa: E501
        :type test_events: List[AggregateResult]
        """
        self.swagger_types = {
            'task': str,
            'eid': str,
            'sha1': str,
            'config': str,
            'dataset': str,
            'username': str,
            'hostname': str,
            'exp_date': date,
            'label': str,
            'version': str,
            'num_exps': int,
            'train_events': List[AggregateResult],
            'valid_events': List[AggregateResult],
            'test_events': List[AggregateResult]
        }

        self.attribute_map = {
            'task': 'task',
            'eid': 'eid',
            'sha1': 'sha1',
            'config': 'config',
            'dataset': 'dataset',
            'username': 'username',
            'hostname': 'hostname',
            'exp_date': 'exp_date',
            'label': 'label',
            'version': 'version',
            'num_exps': 'num_exps',
            'train_events': 'train_events',
            'valid_events': 'valid_events',
            'test_events': 'test_events'
        }

        self._task = task
        self._eid = eid
        self._sha1 = sha1
        self._config = config
        self._dataset = dataset
        self._username = username
        self._hostname = hostname
        self._exp_date = exp_date
        self._label = label
        self._version = version
        self._num_exps = num_exps
        self._train_events = train_events
        self._valid_events = valid_events
        self._test_events = test_events

    @classmethod
    def from_dict(cls, dikt) -> 'ExperimentAggregate':
        """Returns the dict as a model

        :param dikt: A dict.
        :type: dict
        :return: The ExperimentAggregate of this ExperimentAggregate.  # noqa: E501
        :rtype: ExperimentAggregate
        """
        return util.deserialize_model(dikt, cls)

    @property
    def task(self) -> str:
        """Gets the task of this ExperimentAggregate.


        :return: The task of this ExperimentAggregate.
        :rtype: str
        """
        return self._task

    @task.setter
    def task(self, task: str):
        """Sets the task of this ExperimentAggregate.


        :param task: The task of this ExperimentAggregate.
        :type task: str
        """

        self._task = task

    @property
    def eid(self) -> str:
        """Gets the eid of this ExperimentAggregate.


        :return: The eid of this ExperimentAggregate.
        :rtype: str
        """
        return self._eid

    @eid.setter
    def eid(self, eid: str):
        """Sets the eid of this ExperimentAggregate.


        :param eid: The eid of this ExperimentAggregate.
        :type eid: str
        """

        self._eid = eid

    @property
    def sha1(self) -> str:
        """Gets the sha1 of this ExperimentAggregate.


        :return: The sha1 of this ExperimentAggregate.
        :rtype: str
        """
        return self._sha1

    @sha1.setter
    def sha1(self, sha1: str):
        """Sets the sha1 of this ExperimentAggregate.


        :param sha1: The sha1 of this ExperimentAggregate.
        :type sha1: str
        """

        self._sha1 = sha1

    @property
    def config(self) -> str:
        """Gets the config of this ExperimentAggregate.


        :return: The config of this ExperimentAggregate.
        :rtype: str
        """
        return self._config

    @config.setter
    def config(self, config: str):
        """Sets the config of this ExperimentAggregate.


        :param config: The config of this ExperimentAggregate.
        :type config: str
        """

        self._config = config

    @property
    def dataset(self) -> str:
        """Gets the dataset of this ExperimentAggregate.


        :return: The dataset of this ExperimentAggregate.
        :rtype: str
        """
        return self._dataset

    @dataset.setter
    def dataset(self, dataset: str):
        """Sets the dataset of this ExperimentAggregate.


        :param dataset: The dataset of this ExperimentAggregate.
        :type dataset: str
        """

        self._dataset = dataset

    @property
    def username(self) -> str:
        """Gets the username of this ExperimentAggregate.


        :return: The username of this ExperimentAggregate.
        :rtype: str
        """
        return self._username

    @username.setter
    def username(self, username: str):
        """Sets the username of this ExperimentAggregate.


        :param username: The username of this ExperimentAggregate.
        :type username: str
        """

        self._username = username

    @property
    def hostname(self) -> str:
        """Gets the hostname of this ExperimentAggregate.


        :return: The hostname of this ExperimentAggregate.
        :rtype: str
        """
        return self._hostname

    @hostname.setter
    def hostname(self, hostname: str):
        """Sets the hostname of this ExperimentAggregate.


        :param hostname: The hostname of this ExperimentAggregate.
        :type hostname: str
        """

        self._hostname = hostname

    @property
    def exp_date(self) -> date:
        """Gets the exp_date of this ExperimentAggregate.


        :return: The exp_date of this ExperimentAggregate.
        :rtype: date
        """
        return self._exp_date

    @exp_date.setter
    def exp_date(self, exp_date: date):
        """Sets the exp_date of this ExperimentAggregate.


        :param exp_date: The exp_date of this ExperimentAggregate.
        :type exp_date: date
        """

        self._exp_date = exp_date

    @property
    def label(self) -> str:
        """Gets the label of this ExperimentAggregate.


        :return: The label of this ExperimentAggregate.
        :rtype: str
        """
        return self._label

    @label.setter
    def label(self, label: str):
        """Sets the label of this ExperimentAggregate.


        :param label: The label of this ExperimentAggregate.
        :type label: str
        """

        self._label = label

    @property
    def version(self) -> str:
        """Gets the version of this ExperimentAggregate.


        :return: The version of this ExperimentAggregate.
        :rtype: str
        """
        return self._version

    @version.setter
    def version(self, version: str):
        """Sets the version of this ExperimentAggregate.


        :param version: The version of this ExperimentAggregate.
        :type version: str
        """

        self._version = version

    @property
    def num_exps(self) -> int:
        """Gets the num_exps of this ExperimentAggregate.


        :return: The num_exps of this ExperimentAggregate.
        :rtype: int
        """
        return self._num_exps

    @num_exps.setter
    def num_exps(self, num_exps: int):
        """Sets the num_exps of this ExperimentAggregate.


        :param num_exps: The num_exps of this ExperimentAggregate.
        :type num_exps: int
        """

        self._num_exps = num_exps

    @property
    def train_events(self) -> List[AggregateResult]:
        """Gets the train_events of this ExperimentAggregate.


        :return: The train_events of this ExperimentAggregate.
        :rtype: List[AggregateResult]
        """
        return self._train_events

    @train_events.setter
    def train_events(self, train_events: List[AggregateResult]):
        """Sets the train_events of this ExperimentAggregate.


        :param train_events: The train_events of this ExperimentAggregate.
        :type train_events: List[AggregateResult]
        """

        self._train_events = train_events

    @property
    def valid_events(self) -> List[AggregateResult]:
        """Gets the valid_events of this ExperimentAggregate.


        :return: The valid_events of this ExperimentAggregate.
        :rtype: List[AggregateResult]
        """
        return self._valid_events

    @valid_events.setter
    def valid_events(self, valid_events: List[AggregateResult]):
        """Sets the valid_events of this ExperimentAggregate.


        :param valid_events: The valid_events of this ExperimentAggregate.
        :type valid_events: List[AggregateResult]
        """

        self._valid_events = valid_events

    @property
    def test_events(self) -> List[AggregateResult]:
        """Gets the test_events of this ExperimentAggregate.


        :return: The test_events of this ExperimentAggregate.
        :rtype: List[AggregateResult]
        """
        return self._test_events

    @test_events.setter
    def test_events(self, test_events: List[AggregateResult]):
        """Sets the test_events of this ExperimentAggregate.


        :param test_events: The test_events of this ExperimentAggregate.
        :type test_events: List[AggregateResult]
        """

        self._test_events = test_events