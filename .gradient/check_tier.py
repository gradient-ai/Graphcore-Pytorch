# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import os

hostname = os.getenv("HOSTNAME", "unknown")
# TODO: Get list of hostnames for paid tier only
paid_hostnames = [
    "list",
    "of",
    "hostnames",
]

if hostname in paid_hostnames:
    print("PAID")
else:
    print("FREE")
