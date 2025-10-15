#!/usr/bin/env python3

import requests
import sys
import subprocess


def get_github_token():
    result = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def list_artifacts(repo, workflow_id, token):
    url = f"https://api.github.com/repos/{repo}/actions/workflows/{workflow_id}/runs?branch=main"
    headers = {"Authorization": f"Bearer {token}"}
    artifacts = []

    while url:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        runs = data["workflow_runs"]

        for run in runs:
            artifacts_url = run["artifacts_url"]
            artifacts_response = requests.get(artifacts_url, headers=headers)
            artifacts_response.raise_for_status()
            artifacts.extend(artifacts_response.json()["artifacts"])

        # Handle pagination using the Link header
        link_header = response.headers.get("Link", "")
        next_url = None
        if 'rel="next"' in link_header:
            parts = link_header.split(",")
            for part in parts:
                if 'rel="next"' in part:
                    next_url = part.split(";")[0].strip().strip("<>")
                    break
        url = next_url

    return artifacts


def delete_artifact(repo, artifact_id, token):
    url = f"https://api.github.com/repos/{repo}/actions/artifacts/{artifact_id}"
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.delete(url, headers=headers)
    response.raise_for_status()


def get_workflow_id(repo, token, workflow_filename="build.yml"):
    url = f"https://api.github.com/repos/{repo}/actions/workflows"
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    workflows = response.json()["workflows"]

    for workflow in workflows:
        if workflow["path"].endswith(workflow_filename):
            return workflow["id"]

    print(f"Error: Could not find a workflow with the {workflow_filename} file.")
    sys.exit(1)


def main():

    repo = "syna-astra-dev/iree-synaptics-synpu"

    token = get_github_token()
    workflow_id = get_workflow_id(repo, token)
    token = get_github_token()

    print("Listing artifacts...")
    artifacts = list_artifacts(repo, workflow_id, token)

    if not artifacts:
        print("No artifacts found.")
        return
    
    total_size = sum(artifact["size_in_bytes"] for artifact in artifacts)
    print(f"Total size of artifacts: {total_size / (1024 * 1024):.2f} MB")

    print(f"Found {len(artifacts)} artifacts. Deleting...")
    for artifact in artifacts:
        print(f"Deleting artifact: {artifact['name']} (ID: {artifact['id']})")
        delete_artifact(repo, artifact["id"], token)

    print("All artifacts deleted.")

if __name__ == "__main__":
    main()