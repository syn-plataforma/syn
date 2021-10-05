#!/usr/bin/env python3

from pygerrit2 import GerritRestAPI

from syn.helpers.environment import load_environment_variables
from syn.helpers.logging import set_logger

load_environment_variables()
log = set_logger()


def get_data_from_gerrit_rest_api(rest_api_url, after, before, skip):
    rest_api = GerritRestAPI(url=rest_api_url)
    url = "changes/?q=status:closed+after:{" + after + "}+before:{" + before + "}" \
                                                                               "&o=CURRENT_FILES&o=CURRENT_REVISION" \
                                                                               "&o=CURRENT_COMMIT" + skip
    # certificate verify failed: self signed certificate in certificate chain (_ssl.c:1124)'))':
    # /r/changes/?q=status:closed+after:%7B2001-01-01%7D+before:%7B2001-02-01%7D&o=CURRENT_FILES&o=CURRENT_REVISION&o=CURRENT_COMMIT
    changes = rest_api.get(url, headers={'Content-Type': 'application/json'}, verify=False)
    return changes
