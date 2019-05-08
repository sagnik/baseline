import pandas as pd
from swagger_client.models import Experiment, ExperimentAggregate, Result, AggregateResult
from typing import List


def pack_result(results: List[Result]):
    """ List of results to event data"""
    metrics = set([r.metric for r in results])
    d = {metric: [] for metric in metrics}
    for result in results:
        d[result.metric].append(result.value)
    return pd.DataFrame(d)


def pack_aggregate_result(results: List[AggregateResult]):
    metrics = [r.metric for r in results]
    metrics = set(metrics)
    d = {metric: {} for metric in metrics}
    for result in results:
        for value in result.values:
            aggregate_fn = value.aggregate_fn
            score = value.score
            if aggregate_fn in d[result.metric]:
                d[result.metric][aggregate_fn].append(score)
            else:
                d[result.metric][aggregate_fn] = [score]

    dfs = {metric: pd.DataFrame.from_dict(d[metric]) for metric in metrics}
    return pd.concat(dfs.values(), axis=1, keys=dfs.keys())


def get_prop_value(exp, prop_name):
    return exp.__dict__.get('_'+prop_name)


def insert_in_df(prop_name_loc, df, exp):
    for prop_name, location in prop_name_loc.items():
        df.insert(location, column=prop_name, value=[get_prop_value(exp, prop_name)]*len(df))
        
        
def experiment_to_df(exp: Experiment, prop_name_loc={}, sort=None, event_type='test_events'):
    prop_name_loc = {'sha1': 0, 'id': 1, 'user':  2} if not prop_name_loc else prop_name_loc
    if event_type == 'train_events' and exp.train_events:
        result_df = pack_result(exp.train_events)
        insert_in_df(prop_name_loc, result_df, exp)
    if event_type == 'valid_events' and exp.valid_events:
        result_df = pack_result(exp.valid_events)
        insert_in_df(prop_name_loc, result_df, exp)
    if event_type == 'test_events' and exp.test_events:
        result_df = pack_result(exp.test_events)
        insert_in_df(prop_name_loc, result_df, exp)
    if sort is not None:
        result_df.sort_values(by=sort, inplace=True)
    return result_df


def experiment_aggregate_to_df(exp_agg: ExperimentAggregate, prop_name_loc, event_type='test_events'):
    event_dfs = []
    if event_type == 'train_events':
        train_df = pack_aggregate_result(exp_agg.train_events)
        insert_in_df(prop_name_loc, train_df, exp_agg)
        event_dfs.append(train_df)
    if event_type == 'valid_events':
        valid_df = pack_aggregate_result(exp_agg.valid_events)
        insert_in_df(prop_name_loc, valid_df, exp_agg)
        event_dfs.append(valid_df)
    if event_type == 'test_events':
        test_df = pack_aggregate_result(exp_agg.test_events)
        insert_in_df(prop_name_loc, test_df, exp_agg)
        event_dfs.append(test_df)
    result_df = pd.DataFrame()
    for event_df in event_dfs:
        result_df = result_df.append(event_df)
    return result_df


def experiment_aggregate_list_to_df(exp_aggs: List[ExperimentAggregate], event_type='test_events'):
    result_df = pd.DataFrame()
    prop_name_loc = {'sha1': 0, 'num_exps': 1}
    for exp_agg in exp_aggs:
        result_df = result_df.append(experiment_aggregate_to_df(exp_agg, prop_name_loc, event_type))
    return result_df
