import base64
import datetime
import hashlib
import os

import dateutil.relativedelta
import numpy as np
import pandas as pd
from pymongo import MongoClient
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from tqdm import tqdm

from syn.helpers.mongodb import get_default_mongo_client, get_default_local_mongo_client


def get_dataset(dataset, samples, date_range=None):
    client: MongoClient = get_default_mongo_client() if os.environ['WORK_ENVIRONMENT'] == 'aws' \
        else get_default_local_mongo_client()
    if samples == -1:
        df = pd.DataFrame(client[dataset]["clear_assignation"].find({})).drop(["_id"], axis=1)
    else:
        df = pd.DataFrame(client[dataset]["clear_assignation"].find({})[:samples]).drop(["_id"], axis=1)
    df["bug_id"] = df["bug_id"].astype(int)
    df = df.set_index("bug_id")
    df["creation_ts"] = pd.to_datetime(df["creation_ts"], utc=True)
    df["real_closing_date"] = pd.to_datetime(df['real_closing_date'], utc=True)
    return df


def write_database(df, colection_name):
    client: MongoClient = get_default_mongo_client() if os.environ['WORK_ENVIRONMENT'] == 'aws' \
        else get_default_local_mongo_client()
    db = client["assignation"]
    table_dict = df.to_dict("records")
    db[colection_name].insert_many(table_dict)
    return 'Done'


def separate_by_cut_off_date(df, date_limit):
    end = date_limit
    if isinstance(end, tuple):
        end = datetime.datetime(*end, tzinfo=datetime.timezone.utc)
    return df[(df["real_closing_date"] <= end)], df[(df["real_closing_date"] > end)]


def find_open_bugs_at_cut_off_date(df, date_limit):
    end = date_limit
    if isinstance(end, tuple):
        end = datetime.datetime(*end, tzinfo=datetime.timezone.utc)
    return df[df['creation_ts'] <= end]


def find_developers_with_recent_closed_bugs(df, date_limit, window):
    end = date_limit
    if isinstance(end, tuple):
        end = datetime.datetime(*end, tzinfo=datetime.timezone.utc)
    d2 = end - dateutil.relativedelta.relativedelta(months=window)
    df = df[(df['real_closing_date'] >= d2)]
    return set(list(df['assigned_to']))


def new_incomming_bugs(df, date_limit, step):
    end = date_limit
    if isinstance(end, tuple):
        end = datetime.datetime(*end, tzinfo=datetime.timezone.utc)
    d2 = end + dateutil.relativedelta.relativedelta(months=step)
    df = df[(df['creation_ts'] > end) & (df['creation_ts'] <= d2)]
    return df


def evaluate_idea(ground_truth, global_metric, top_k):
    result_df = pd.DataFrame(columns=['bug_id'])
    for k, v in ground_truth.items():
        bug_id = k
        real_dev = v[0]
        results = {'bug_id': bug_id}
        for i in range(top_k):
            specific_bug = global_metric[bug_id]
            firstkpairs = {j: specific_bug for j in list(specific_bug.keys())[:i + 1]}
            dev_retrieval = list(firstkpairs.keys())
            if real_dev in dev_retrieval:
                acc = 1.0
            else:
                acc = 0.0
            results[i + 1] = acc
        result_df = result_df.append(results, ignore_index=True)
    return result_df


def get_ground_truth(df):
    df = df.filter(['assigned_to'])
    real_developer = df.T.to_dict('list')
    return real_developer


def get_occupation_level_all_developers(df_open, all_dev, df_cost):
    occupation_level = {}
    dict_dev_task = get_developers_with_open_bugs(df_open)
    for developer in all_dev:
        try:
            tasks = dict_dev_task[developer]
            cost_task = []
            for task in tasks:
                registry = df_open.loc[task, ['component', 'product', 'priority']].to_dict()
                values = list(registry.values())
                criterion = ''
                for val in values:
                    criterion += val
                try:
                    occ = df_cost.at[criterion, 'normalized_cost']
                except KeyError:
                    occ = 0.0
                cost_task.append(occ)
            occupation_level[developer] = np.mean(cost_task)
        except KeyError:
            occupation_level[developer] = 0.0
    return occupation_level


def get_normalized_cost(df):
    df_cost = df.groupby('composite_id', as_index=False)['resolution_time_hours'].mean()
    resolution_time_hours_list = []
    [resolution_time_hours_list.append([item]) for item in list(df_cost['resolution_time_hours'])]
    p = Pipeline([("a", QuantileTransformer(output_distribution='normal')), ("b", MinMaxScaler())])
    normalized_cost: np.ndarray = p.fit_transform(np.array(resolution_time_hours_list))
    normalized_cost_list = []
    [normalized_cost_list.append(item[0]) for item in list(normalized_cost)]
    data = {'composite_id': list(df_cost['composite_id']), 'normalized_cost': normalized_cost_list}
    df_cost_normalized = pd.DataFrame(data)
    return df_cost_normalized


def get_normalized_workload(
        df_normalized_cost: pd.DataFrame = None,
        df_active_developers: pd.DataFrame = None
) -> pd.DataFrame:
    # workload of each active developer
    df_sum_cost = df_normalized_cost.groupby('label', as_index=False)['normalized_cost'].sum()
    df_workload = df_active_developers.merge(df_sum_cost, how='left', on='label')
    df_workload.rename(columns={'normalized_cost': 'sum_normalized_cost'}, errors='raise',
                       inplace=True)
    df_workload['sum_normalized_cost'].fillna(0.0, inplace=True)

    # workload of all active developer
    all_active_developers_workload = df_workload['sum_normalized_cost'].sum() / df_active_developers.shape[0]

    # workload of each active developer in units of their average load
    df_workload['workload'] = df_workload['sum_normalized_cost'] / all_active_developers_workload

    # quantile function of the workload
    workload_list = []
    [workload_list.append([item]) for item in list(df_workload['workload'])]
    p = Pipeline([("a", QuantileTransformer(output_distribution='normal')), ("b", MinMaxScaler())])
    normalized_workload: np.ndarray = p.fit_transform(np.array(workload_list))
    normalized_workload_list = []
    [normalized_workload_list.append(item[0]) for item in list(normalized_workload)]
    data = {'label': list(df_workload['label']), 'normalized_workload': normalized_workload_list}
    df_normalized_workload = pd.DataFrame(data)
    return df_normalized_workload


def get_issue_developers(train_issues: pd.DataFrame = None):
    data = {}
    composite_id_unique = list(train_issues['composite_id'].unique())
    for composite_id in composite_id_unique:
        composite_id_issues = train_issues[train_issues['composite_id'] == composite_id].copy()
        composite_id_issues_value_counts = composite_id_issues['label'].value_counts().sort_values(
            ascending=False).to_dict()
        data[composite_id] = []
        for k in composite_id_issues_value_counts.keys():
            data[composite_id].append({
                'assigned_to': k,
                'count': composite_id_issues_value_counts[k]
            })

    return data


def get_developer_issues_extra(train_issues: pd.DataFrame = None) -> dict:
    data = {}
    active_developers = list(train_issues['label'].unique())
    for developer in tqdm(active_developers, total=len(active_developers), desc='developers'):
        developer_issues = train_issues[train_issues['label'] == developer].copy()
        developer_issues_value_counts = developer_issues['composite_id'].value_counts().sort_values(ascending=False)

        composite_id_unique = train_issues[['composite_id']].copy()
        composite_id_unique.drop_duplicates(inplace=True)
        data[developer] = []
        for k in developer_issues_value_counts.keys():
            idx = composite_id_unique[composite_id_unique['composite_id'] == k].index
            data[developer].append({
                'composite_id': k,
                'composite_data': list(train_issues.iloc[idx]['composite_data']),
                'count': int(developer_issues_value_counts[k])
            })

    return data


def get_developer_issues(train_issues: pd.DataFrame = None) -> dict:
    data = {}
    developers = list(train_issues['label'].unique())
    for developer in tqdm(developers, total=len(developers), desc='developers'):
        developer_issues = train_issues[train_issues['label'] == developer].copy()
        developer_issues_value_counts = developer_issues['composite_id'].value_counts().sort_values(ascending=False)

        composite_id_unique = train_issues[['composite_id']].copy()
        composite_id_unique.drop_duplicates(inplace=True)
        data[developer] = []

        for k in developer_issues_value_counts.keys():
            idx = composite_id_unique[composite_id_unique['composite_id'] == k].index
            data[developer].append(list(train_issues.iloc[idx]['composite_data'])[0])

    return data


def get_keyword_developer(df, developer):
    """
    Obtiene el las palabras clave (component, product, bug_severity) asociadas a cada desarrollador
    Los valores devueltos corresponden a la MODA
    """
    keyword_develop = dict()
    df = df.filter(['assigned_to', 'component', 'product', 'bug_severity'])
    for develop in developer:
        df_develop = df[df['assigned_to'] == develop]
        df_develop = df_develop.filter(['component', 'product', 'bug_severity'])
        # El valor final es la MODA, el valor mas repetido
        df_mode = df_develop.mode(dropna=True)
        try:
            keyword_develop[develop] = df_mode.loc[0, :].values.tolist()
        except KeyError:
            keyword_develop[develop] = ['new', 'new', 'new']
    return keyword_develop


def get_keyword_bug(df):
    """
    Obtiene las palabras clave (component, product, bug_severity) asociadas a cada incidencia
    """
    df = df.filter(['component', 'product', 'bug_severity'])
    keyword_bug = df.T.to_dict('list')
    return keyword_bug


def get_adecuation_and_global_metric_order(keyword_develop, keyword_bug, ocupation, w):
    """Esta funcion para cada incidencia reccore todos los desarrolladores posibles, calculao la adecuacion
    y luego devuleve la metrica global"""
    global_metric = dict()
    i = 0
    for bug in keyword_bug:
        i = i + 1
        global_metric[bug] = dict()
        for develop in keyword_develop:  #
            # Se calcula el nivel de adecucion con la funcion jaccard
            adecuation = jaccard_similarity(keyword_develop[develop], keyword_bug[bug])
            global_metric[bug][develop] = np.float32(w * (1 - ocupation[develop]) + (1 - w) * adecuation)
        # Se ordenan los valores obtenidos de mayor a menor
        global_metric[bug] = {k: v for k, v in
                              sorted(global_metric[bug].items(), key=lambda item: item[1], reverse=True)}

    return global_metric


def jaccard_similarity(list1, list2):
    """
    Devuelve la similitud de jaccard
    """
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union


def get_developers_with_open_bugs(df):
    df_column = pd.DataFrame(df['assigned_to'].value_counts())
    develop = df_column.index.values.tolist()
    task = dict()
    develop_task = dict()
    for dev in develop:
        df_dev = df[df['assigned_to'] == dev]
        task[dev] = df_dev.index.values.tolist()
        develop_task[dev] = task[dev]
    return develop_task


def sum_up_result(df):
    cols = list(df.columns)
    needed_list = cols[1:]
    x_axis, y_axis = [], []
    for item in needed_list:
        x_axis.append(int(item))
        y_axis.append(100 * df[item].mean())
    dict_of_results = {}
    for i in range(len(x_axis)):
        dict_of_results[x_axis[i]] = y_axis[i]
    return dict_of_results


def get_hash(*composite_fields):
    composite_fields_str = ''
    for composite_field in composite_fields:
        composite_fields_str += f"{str(composite_field)}-"
    composite_fields_str_final = composite_fields_str[:-1]
    h = hashlib.md5(composite_fields_str_final.encode('utf-8'))
    result = base64.b64encode(h.digest())[:12].decode("utf-8").replace("/", "_")
    return result


def get_composite_fields(*composite_fields):
    composite_data = []
    for composite_field in composite_fields:
        composite_data.append(composite_field)

    return composite_data


def get_resolution_time(delta_ts: datetime, creation_ts: datetime):
    resolution_time = delta_ts - creation_ts
    hours, remainder = divmod(resolution_time.total_seconds(), 3600)
    return hours


def get_normalized_cost_by_composite_id(composite_id: str = None, normalized_cost: dict = None):
    result = 0.0
    for item in normalized_cost.items():
        result += item[1]
    result /= len(normalized_cost.keys())
    if composite_id in normalized_cost.keys():
        result = normalized_cost[composite_id]

    return result


def predict_assigned_to(developers: list = None, developers_info: dict = None):
    print(len(developers))
    return None
