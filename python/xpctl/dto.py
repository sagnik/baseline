from swagger_server.models import Experiment, ExperimentAggregate, Result, Error, AggregateResult, TaskSummary
from xpctl.data import TaskSummary as TTaskSummary
from xpctl.backend.mongo.dto import MongoError


def dto_experiment_details(exp):
    if type(exp) == MongoError:
        return Error(code=exp.code, message=exp.message)
    train_events = [Result(**r.__dict__) for r in exp.train_events]
    valid_events = [Result(**r.__dict__) for r in exp.valid_events]
    test_events = [Result(**r.__dict__) for r in exp.test_events]
    d = exp.__dict__
    d.update({'train_events': train_events})
    d.update({'valid_events': valid_events})
    d.update({'test_events': test_events})
    return Experiment(**d)


def dto_get_results(agg_exps):
    if type(agg_exps) == MongoError:
        return Error(code=agg_exps.code, message=agg_exps.message)
    results = []
    for agg_exp in agg_exps:
        if type(agg_exp) == MongoError:
            return Error(code=agg_exp.code, message=agg_exp.message)
        train_events = [AggregateResult(**r.__dict__) for r in agg_exp.train_events]
        valid_events = [AggregateResult(**r.__dict__) for r in agg_exp.valid_events]
        test_events = [AggregateResult(**r.__dict__) for r in agg_exp.test_events]
        d = agg_exp.__dict__
        d.update({'train_events': train_events})
        d.update({'valid_events': valid_events})
        d.update({'test_events': test_events})
        results.append(ExperimentAggregate(**d))
    return results


def dto_list_results(exps):
    if type(exps) == MongoError:
        return Error(code=exps.code, message=exps.message)
    results = []
    for exp in exps:
        if type(exp) == MongoError:
            return Error(code=exp.code, message=exp.message)
        train_events = [Result(**r.__dict__) for r in exp.train_events]
        valid_events = [Result(**r.__dict__) for r in exp.valid_events]
        test_events = [Result(**r.__dict__) for r in exp.test_events]
        d = exp.__dict__
        d.update({'train_events': train_events})
        d.update({'valid_events': valid_events})
        d.update({'test_events': test_events})
        results.append(Experiment(**d))
    return results


def dto_task_summary(task_summary):
    if type(task_summary) == MongoError:
        return Error(code=task_summary.code, message=task_summary.message)
    return TaskSummary(**task_summary.__dict__)


def dto_summary(task_summaries):
    if type(task_summaries) == MongoError:
        return Error(code=task_summaries.code, message=task_summaries.message)
    return [TaskSummary(**task_summary.__dict__) for task_summary in task_summaries]


def dto_config2json(config):
    if type(config) == MongoError:
        return Error(code=config.code, message=config.message)
    return config


def dto_get_model_location(location):
    if type(location) == MongoError:
        return Error(code=location.code, message=location.message)
    return location