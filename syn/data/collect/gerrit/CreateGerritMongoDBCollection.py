#!/usr/bin/env python3
import argparse
import datetime
import os
import time

import pandas as pd
from pymongo import MongoClient

from syn.helpers.environment import load_environment_variables
from syn.helpers.gerrit import get_data_from_gerrit_rest_api
from syn.helpers.logging import set_logger
from syn.helpers.mongodb import get_default_mongo_client, save_issues_to_mongodb
from syn.helpers.system import check_same_python_module_already_running

load_environment_variables()
log = set_logger()


def get_input_params():
    parser = argparse.ArgumentParser(description='Retrieve clear and normalized clear collections.')

    parser.add_argument('--db_name', default='gerrit', type=str, help='Gerrit database name.')
    parser.add_argument('--collection_name', default='eclipse', type=str, help='Gerrit collection name.')
    parser.add_argument('--year', default=2001, type=int, help='Gerrit bug year.')
    parser.add_argument('--start_month', default=1, type=int, help='Gerrit bug creation month filter.')
    parser.add_argument('--end_month', default=12, type=int, help='Gerrit bug creation month filter.')

    args = parser.parse_args()

    return {
        'db_name': args.db_name,
        'collection_name': args.collection_name,
        'year': args.year,
        'start_month': args.start_month,
        'end_month': args.end_month
    }


def main():
    # Stores the execution start time to calculate the time it takes for the module to execute.
    initial_time = time.time()

    # Check if there is a running process that contains the name of this module.
    check_same_python_module_already_running(os.path.split(__file__))

    # Load parameters.
    input_params = get_input_params()

    # Mongo client.
    mongodb_client: MongoClient = get_default_mongo_client()

    db = mongodb_client[os.environ.get('GERRIT_DB_NAME', input_params['db_name'])]
    year = input_params['year']
    col = db[
        f"{input_params['collection_name']}"
        f"_{year}_{year + 1}"
    ]

    log.debug(f"Existent collections in '{db.name}': {str(db.list_collection_names())}")
    if col.name in db.list_collection_names():
        db.drop_collection(col.name)

    # Get Gerrit Issues by month.
    max_year = year
    for month in range(input_params['start_month'], input_params['end_month'] + 1):
        max_month = month + 1
        if max_month > 12:
            max_month = 1
            max_year += 1

        after = str(datetime.datetime(input_params['year'], month, 1))
        before = str(datetime.datetime(max_year, max_month, 1))
        s = 0
        change_id, project, status, date, subject, author, committer, commit_msg = [], [], [], [], [], [], [], []
        file_list = []
        skip = ''
        flag = True
        log.info(f"Retrieving issues in range: {after} - {before}")
        while flag:
            issues_by_month = get_data_from_gerrit_rest_api(os.environ['GERRIT_API_URL'], after, before, skip=skip)
            log.info(f"Read issues: {len(issues_by_month)}")
            if len(issues_by_month) < 1:
                print('No issues for month {} and year {}'.format(month, year))
                flag = False
            else:
                for issue in issues_by_month:
                    change_id.append(issue['change_id'])
                    project.append(issue['project'])
                    status.append(issue['status'])
                    date.append(issue['updated'])
                    subject.append(issue['subject'])
                    current_rev = issue['current_revision']
                    author.append(issue['revisions'][current_rev]['commit']['author']['email'])
                    committer.append(issue['revisions'][current_rev]['commit']['committer']['email'])
                    commit_msg.append(issue['revisions'][current_rev]['commit']['message'])
                    file_list.append(list(issue['revisions'][current_rev]['files'].keys()))
                if len(issues_by_month) == 100:
                    last_dict = issues_by_month[99]
                    try:
                        if last_dict['_more_changes']:
                            log.info(f"There are more issues to read")
                            s = s + 100
                            skip = '&S=' + str(s)
                    except KeyError:
                        flag = False
                        print('Issues for month {} and year {}: {}'.format(month, year, len(change_id)))
                else:
                    print('Issues for month {} and year {}: {}'.format(month, year, len(change_id)))
                    flag = False
        dict_of_reg = {'change_id': change_id, 'project': project, 'status': status, 'date': date, 'subject': subject,
                       'author': author, 'committer': committer, 'commit_msg': commit_msg, 'file_list': file_list}
        df = pd.DataFrame(dict_of_reg)
        issues = df.to_dict("records")
        save_issues_to_mongodb(mongodb_collection=col, issues=issues)

    final_time = time.time()
    log.info(f"Total execution time = {((final_time - initial_time) / 60)} minutes")


if __name__ == '__main__':
    main()
