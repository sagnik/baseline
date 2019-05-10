# coding: utf-8

from __future__ import absolute_import
from datetime import date, datetime  # noqa: F401

from typing import List, Dict  # noqa: F401

from swagger_server.models.base_model_ import Model
from swagger_server.models.aggregate_result_values import AggregateResultValues  # noqa: F401,E501
from swagger_server import util


class AggregateResult(Model):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """

    def __init__(self, metric=None, values=None):  # noqa: E501
        """AggregateResult - a model defined in Swagger

        :param metric: The metric of this AggregateResult.  # noqa: E501
        :type metric: str
        :param values: The values of this AggregateResult.  # noqa: E501
        :type values: List[AggregateResultValues]
        """
        self.swagger_types = {
            'metric': str,
            'values': List[AggregateResultValues]
        }

        self.attribute_map = {
            'metric': 'metric',
            'values': 'values'
        }

        self._metric = metric
        self._values = values

    @classmethod
    def from_dict(cls, dikt):
        """Returns the dict as a model

        :param dikt: A dict.
        :type: dict
        :return: The AggregateResult of this AggregateResult.  # noqa: E501
        :rtype: AggregateResult
        """
        return util.deserialize_model(dikt, cls)

    @property
    def metric(self):
        """Gets the metric of this AggregateResult.


        :return: The metric of this AggregateResult.
        :rtype: str
        """
        return self._metric

    @metric.setter
    def metric(self, metric):
        """Sets the metric of this AggregateResult.


        :param metric: The metric of this AggregateResult.
        :type metric: str
        """
        if metric is None:
            raise ValueError("Invalid value for `metric`, must not be `None`")  # noqa: E501

        self._metric = metric

    @property
    def values(self):
        """Gets the values of this AggregateResult.


        :return: The values of this AggregateResult.
        :rtype: List[AggregateResultValues]
        """
        return self._values

    @values.setter
    def values(self, values):
        """Sets the values of this AggregateResult.


        :param values: The values of this AggregateResult.
        :type values: List[AggregateResultValues]
        """
        if values is None:
            raise ValueError("Invalid value for `values`, must not be `None`")  # noqa: E501

        self._values = values
