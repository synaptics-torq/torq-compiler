#!/usr/bin/env python3

import os
import requests


relative_uri = os.environ.get("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", "")

if not relative_uri:
    raise RuntimeError("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI is not set")

response = requests.get(f"http://169.254.170.2{relative_uri}")

if response.status_code != 200:
    raise RuntimeError(f"Failed to retrieve AWS credentials: {response.text}")

creds = response.json()

OUTPUT_TO_FIELD = {"access_key": "AccessKeyId", "secret_key": "SecretAccessKey", "session_token": "Token"}

# tell GitHub to mask the secrets in logs
for creds_field in OUTPUT_TO_FIELD.values():
    print(f"::add-mask::{creds[creds_field]}")

# write the secrets to GITHUB_OUTPUT to set them as action outputs
with open(os.environ["GITHUB_OUTPUT"], "a") as gh_output:
    for output_field, creds_field in OUTPUT_TO_FIELD.items():
        gh_output.write(f"{output_field}={creds[creds_field]}\n")
