#!/usr/bin/env python3

import argparse
import subprocess
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import json
import os
import zipfile
import stat


def find_repo():
    """
    Find the owner/name of the repo where this script resides.
    """

    if os.environ.get("GITHUB_REPOSITORY", "") != "":
        print("Using repository from environment variable GITHUB_REPOSITORY")
        return os.environ.get("GITHUB_REPOSITORY")

    remote_url = subprocess.check_output(
        ["git", "config", "--get", "remote.origin.url"], text=True
    ).strip()

    if remote_url.startswith("git@"):
        host, sep, path = remote_url.partition(":")
        if host != "git@github.com" or not sep:
            raise RuntimeError(f"Unsupported git remote (expected GitHub): {remote_url}")
    elif "://" in remote_url:

        parsed = urlparse(remote_url)
        if parsed.hostname != "github.com":
            raise RuntimeError(f"Unsupported git remote host: {parsed.hostname}")
        path = parsed.path.lstrip("/")
    else:
        raise RuntimeError(f"Unrecognized git remote format: {remote_url}")

    path = path.rstrip("/")
    if path.endswith(".git"):
        path = path[:-4]

    parts = path.split("/")
    if len(parts) != 2 or not all(parts):
        raise RuntimeError(f"Could not parse GitHub owner/repo from remote: {remote_url}")

    return f"{parts[0]}/{parts[1]}"


def _http_get(url, headers=None):
    """
    Perform an HTTP GET and return status code and response body bytes.
    """
    request = Request(url, headers=headers or {}, method="GET")
    try:
        with urlopen(request) as response:
            return response.status, response.read()
    except HTTPError as e:
        return e.code, e.read()
    except URLError as e:
        raise RuntimeError(f"Failed to reach {url}: {e.reason}")


def _github_get(url, token=None, accept=None):
    """
    Perform a GitHub API GET and return status code and response body bytes.
    """
    headers = {"User-Agent": "download_tflite_binaries.py"}
    if token:
        headers["Authorization"] = f"token {token}"
    if accept:
        headers["Accept"] = accept
    return _http_get(url, headers=headers)


def get_token(repo):
    """
    Get a GitHub token with permissions to read releases from the given repo. This will first check for a token in the environment variable GITHUB_TOKEN, and if not found, will attempt to use the gh CLI to get a token.
    """
              
    # check if this repo is public by trying to access releases without auth
    status_code, _ = _github_get(f"https://api.github.com/repos/{repo}/releases")
    if status_code == 200:
        return None

    # if not public, try to get a token from the environment variable
    if os.environ.get("GITHUB_TOKEN", "") != "":
        print("Using GitHub token from environment variable")
        return os.environ.get("GITHUB_TOKEN")

    # not found, try to get a token from the gh CLI
    try:
        token = subprocess.check_output(
            ["gh", "auth", "token", "--hostname", "github.com"], text=True
        ).strip()

    except Exception as e:
        raise RuntimeError(
            f"Failed to get GitHub token from environment or gh CLI: {e}"
        )
        
    # verify the token
    status_code, body = _github_get(f"https://api.github.com/repos/{repo}/releases", token=token)
    if status_code != 200:
        raise RuntimeError(
            f"GitHub token is not valid for accessing releases of {repo}: "
            f"{status_code} {body.decode('utf-8', errors='replace')}"
        )

    return token

def get_artifact_url(repo, token, release_version, tflite_version):
    """
    Get the download URL for the artifact containing the TensorFlow binaries for the given release version and TensorFlow version.
    """

    status_code, body = _github_get(
        f"https://api.github.com/repos/{repo}/releases/tags/{release_version}",
        token=token,
    )
    if status_code != 200:
        raise RuntimeError(
            f"Failed to get release information for {release_version} from {repo}: "
            f"{status_code} {body.decode('utf-8', errors='replace')}"
        )

    release_info = json.loads(body.decode("utf-8"))

    for asset in release_info.get("assets", []):
        if asset["name"] == f"tflite-tools-{tflite_version}.zip":
            if token:
                # return the API url for the asset, which requires authentication, instead of the browser download URL, 
                # which does not work with token authentication
                return f"https://api.github.com/repos/{repo}/releases/assets/{asset['id']}"
            else:
                # return the browser download URL if no token is needed, since it is simpler and does not require authentication
                return asset["browser_download_url"]

    raise RuntimeError(f"Could not find artifact tflite-tools-{tflite_version}.zip in release {release_version} of {repo}")


def extract_artifact(url, token, output_dir):
    """
    Download the artifact from the given URL and extract it to the output directory.
    """

    status_code, body = _github_get(url, token=token, accept="application/octet-stream")
    if status_code != 200:
        raise RuntimeError(
            f"Failed to download artifact from {url}: "
            f"{status_code} {body.decode('utf-8', errors='replace')}"
        )

    os.makedirs(output_dir, exist_ok=True)

    # save the zip file to the output directory
    artifact_path = os.path.join(output_dir, "artifact.zip")
    with open(artifact_path, "wb") as f:
        f.write(body)

    with zipfile.ZipFile(artifact_path, "r") as zip_ref:
        for member in zip_ref.infolist():
            extracted_path = zip_ref.extract(member, path=output_dir)

            # preserve UNIX mode bits (including executable permissions) when present.
            mode = member.external_attr >> 16
            if mode and not stat.S_ISLNK(mode):
                os.chmod(extracted_path, mode)

    os.remove(artifact_path)


def main():

    argparser = argparse.ArgumentParser(description="Download the latest TensorFlow binaries built by GitHub Actions.")

    argparser.add_argument("--repo", type=str, default=None, help="The name of the repository to download the binaries from. If not specified, it will be inferred from the location of this script.")
    argparser.add_argument("--release-version", type=str, default="snapshot", help="The name of the release to download the binaries from. Defaults to 'snapshot' which is a special release that is updated on each commit to main. Can also specify a specific release tag (e.g. 'v2.20.0').")
    argparser.add_argument("--tflite-version", type=str, default="v2.20.0", help="The version of TensorFlow to download the binaries for.")
    argparser.add_argument("--output-dir", type=str, required=True, help="Directory to download and extract the TensorFlow binaries to.")

    args = argparser.parse_args()
    
    # find the name of the repo that stores this script
    if args.repo is None:
        remote_repo = find_repo()
    else:
        remote_repo = args.repo

    print(f"Downloading artifact from repository {remote_repo}")

    # get a GitHub token with permissions to read releases from the repo if this is not a public repo
    token = get_token(remote_repo)

    if not token:
        print("Repository is public, no token needed")
    else:
        print("Private repository, using GitHub token for authentication")
    
    # get the relase aritfact url
    artifact_url = get_artifact_url(remote_repo, token, args.release_version, args.tflite_version)

    print("Downloading artifact from URL: ", artifact_url)

    # download the artifact and extract it 
    extract_artifact(artifact_url, token, args.output_dir)


if __name__ == "__main__":
    main()