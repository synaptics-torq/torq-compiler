#!/usr/bin/env python

import json
import argparse
import http.client
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Get the latest workflow run ID from a GitHub repository branch.")

    parser.add_argument("--repo", type=str, required=True, help="GitHub repository name")
    parser.add_argument("--branch", type=str, required=True, help="GitHub repository branch")
    parser.add_argument("--workflow", type=str, required=True, help="GitHub Actions workflow file name")

    args = parser.parse_args()

    conn = http.client.HTTPSConnection("api.github.com")

    headers = {
        'Accept': 'application/vnd.github+json',
        'User-Agent': 'python-http-client',
        'Authorization': f'Bearer {os.environ["GITHUB_TOKEN"]}',
    }

    branch = args.branch

    if branch.startswith("refs/heads/"):
        branch = branch[len("refs/heads/"):]

    # this relies on the undocumented sorting by created_at descending
    endpoint = f"/repos/{args.repo}/actions/workflows/{args.workflow}/runs?branch={branch}&per_page=1&status=completed"

    conn.request("GET", endpoint, headers=headers)

    res = conn.getresponse()
    data = res.read()
    runs = json.loads(data)

    if 'workflow_runs' not in runs or len(runs['workflow_runs']) == 0:
        sys.stderr.write("No completed workflow runs found.\n")
        exit(1)

    latest_run_id = runs['workflow_runs'][0]['id']

    print(latest_run_id)


if __name__ == "__main__":
    main()