"""Manage Jenkins-backed board allocation for Torq test workflows.

This module provides a small command-line utility for requesting boards from
the `SL261X_NPU_Test` Jenkins job, polling until the boards are ready,
connecting to an allocated board over SSH, releasing the allocation back to the
farm, and listing currently queued or running jobs.

The CLI stores allocation state in a JSON control file so long-running flows can
be split across commands:

- `acquire`: create a new allocation or attach to an existing Jenkins build.
- `wait`: poll Jenkins until the allocated boards publish their IP addresses.
- `connect`: SSH to one of the allocated boards.
- `release`: mark the Jenkins input step as complete and clean up local state.
- `list-queue`: show queued, running, and completed builds for the board job.

Environment:
- `JENKINS_USER`: Jenkins username used for API requests.
- `JENKINS_API_TOKEN`: Jenkins API token paired with `JENKINS_USER`.
"""

import requests
import uuid
import subprocess
import tempfile
import argparse
import os
import time
import json
import shutil
import fcntl
from abc import ABC, abstractmethod

from pathlib import Path

from .remote_runner import remote_command_runner_factory


BOARD_USER = "root"

class BuildFailedException(Exception):
    pass


class BoardsControl(ABC):

    backend_name = "unknown"

    def __init__(self, control_file: str | None):
        self.control_file = control_file

    def _read_control_data_locked(self, file_obj):
        file_obj.seek(0)
        raw = file_obj.read().strip()
        if not raw:
            return {}
        return json.loads(raw)

    def _write_control_data_locked(self, file_obj, data):
        file_obj.seek(0)
        file_obj.truncate()
        json.dump(data, file_obj)
        file_obj.flush()
        os.fsync(file_obj.fileno())

    def _require_control_file(self):
        if not self.control_file:
            raise Exception("A control file is required for this operation")

    def _initialize_control_data(self, data):
        data["backend"] = self.backend_name
        return data

    @property
    def remote_runner_path(self):
        """
        Returns the path where the torq-run-module binary is expected to be on the board, this is where the setup_board method will copy it to.
        """

        return "/tmp/torq-run-module"

    def get_remote_address(self, board_user, board_id):
        if board_user:
            return f"{board_user}@{board_id}"
        return board_id

    def get_any_board(self):
        """
        Get the IP of an available board and mark it locked, wait until a board is available if all are locked.

        This works even if multiple programs are using the same control file
        """

        self.wait_boards_ready()

        while True:
            with open(self.control_file, "a+") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                data = self._read_control_data_locked(f)

                if not data.get("board_ips"):
                    raise Exception("Boards are not ready yet: missing board IPs in control file")

                for i, (ip, locked) in enumerate(zip(data["board_ips"], data["board_locks"])):
                    if not locked:
                        data["board_locks"][i] = True
                        self._write_control_data_locked(f, data)
                        return data.get("board_user", BOARD_USER), ip, data.get("board_private_key_path")

            print("All boards are currently in use, waiting for a board to be available...")
            time.sleep(10)

    def get_board(self, address):
        """
        Get mark the specified board as locked, wait until a board is available if it is already locked.

        This works even if multiple programs are using the same control file
        """

        self.wait_boards_ready()

        while True:
            with open(self.control_file, "a+") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                data = self._read_control_data_locked(f)

                if not data.get("board_ips"):
                    raise Exception("Boards are not ready yet: missing board IPs in control file")

                for i, (ip, locked) in enumerate(zip(data["board_ips"], data["board_locks"])):
                    if address == ip and not locked:
                        data["board_locks"][i] = True
                        self._write_control_data_locked(f, data)
                        return data.get("board_user", BOARD_USER), ip, data.get("board_private_key_path")

            print("Board is currently in use, waiting for a board to be available...")
            time.sleep(10)

    def board(self, address=None):
        """
        Context manager that acquires a board on entry and releases it on exit.

        Usage::

            with control.board() as (user, ip, private_key):
                # use the board at `ip` and `user` with the given `private_key` for SSH access
        """
        from contextlib import contextmanager

        @contextmanager
        def _board_ctx():
            if address is not None:
                user, ip, private_key = self.get_board(address)
            else:
                user, ip, private_key = self.get_any_board()

            try:
                yield user, ip, private_key
            finally:
                self.put_board(ip)

        return _board_ctx()

    def put_board(self, ip):
        """
        Mark a previously acquired board as available again.
        """

        with open(self.control_file, "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            data = self._read_control_data_locked(f)

            try:
                idx = data["board_ips"].index(ip)
            except ValueError:
                raise Exception(f"Board IP {ip} not found in control file")

            if not data["board_locks"][idx]:
                raise Exception(f"Board IP {ip} is not currently locked")

            data["board_locks"][idx] = False
            self._write_control_data_locked(f, data)

    def setup_board(self, ip, local_runner_path, local_ko_path=None):
        """
        Installs torq-runtime and kernel driver to a board
        """

        print(f"Setting up board at {ip}...")

        if not Path(local_runner_path).exists():
            raise Exception(f"Local runner binary not found at {local_runner_path}")

        if local_ko_path is not None and not Path(local_ko_path).exists():
            raise Exception(f"Local kernel module not found at {local_ko_path}")

        with self.board(ip) as (board_user, board_ip, private_key_path):
            with remote_command_runner_factory(
                self.get_remote_address(board_user, board_ip),
                timeout=5 * 60,
                ssh_multiplex=True,
                ssh_port=22,
                ssh_private_key=private_key_path,
            ) as r:

                print("Checking board is alive...")
                r.run_cmd("echo hello")

                print(f"Copying torq-run-module from {local_runner_path}...")
                r.copy_files(str(local_runner_path), self.remote_runner_path, board_dst=True, verbose=True)

                if local_ko_path is not None:
                    print(f"Copying kernel module from {local_ko_path}...")
                    r.copy_files(str(local_ko_path), "/tmp/syna_npu.ko", board_dst=True, verbose=True)
                    r.run_cmd("rmmod syna_npu")
                    r.run_cmd("insmod /tmp/syna_npu.ko")
                else:
                    print("No kernel module path provided, skipping kernel module installation")

        print(f"Board at {board_ip} setup completed")

    def list_boards(self) -> list[str]:
        """
        Return the list of board IPs from the control file, or an empty list if the boards are not ready yet.
        """
        with open(self.control_file, "a+") as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            data = self._read_control_data_locked(f)
            return data.get("board_ips", [])

    def setup_boards(self, local_runner_path, local_ko_path=None) -> str:
        """
        Installs torq runtime and kernel driver to all boards
        """

        self.wait_boards_ready()

        for ip in self.list_boards():
            self.setup_board(ip, local_runner_path, local_ko_path)

    @abstractmethod
    def wait_boards_ready(self):
        pass

    @abstractmethod
    def is_active(self):
        pass

    @abstractmethod
    def release_boards(self, general_result="pass", reason="Boards no longer needed"):
        pass

    @abstractmethod
    def connect_to_board(self, board_user, board_id, private_key_path=None, verbose=False):
        pass


class JenkinsBoardsControl(BoardsControl):

    backend_name = "jenkins"

    def __init__(self, control_file: str):
        super().__init__(control_file)
        self.jenkins_url = os.environ.get("JENKINS_URL", "").rstrip("/")
        self.jenkins_job = os.environ.get("JENKINS_JOB", "")
        self.user = os.environ.get("JENKINS_USER", "")
        self.token = os.environ.get("JENKINS_API_TOKEN", "")

        if not self.jenkins_url or not self.user or not self.token:
            raise Exception("Missing Jenkins credentials in environment variables, please set JENKINS_URL, JENKINS_USER, and JENKINS_API_TOKEN")


    def load_keypair(self, private_key_path):
        """
        Load an existing SSH key pair from the given private key path. The public key is expected to be in the same directory with the same name but with .pub extension.
        """        

        with open(self.control_file, "a+") as f:

            fcntl.flock(f, fcntl.LOCK_EX)

            data = self._initialize_control_data(self._read_control_data_locked(f))

            if data.get("board_private_key_path") is not None:
                raise Exception(f"Control file already contains a private key path {data['board_private_key_path']}, cannot load new key pair")

            public_key_path = f"{private_key_path}.pub"

            if not os.path.exists(private_key_path):
                raise Exception(f"Private key not found at {private_key_path}")

            if not os.path.exists(public_key_path):
                raise Exception(f"Public key not found at {public_key_path}")

            with open(public_key_path, "r") as kf:
                board_public_key = kf.read()

            data["board_public_key"] = board_public_key
            data["board_private_key_path"] = private_key_path

            self._write_control_data_locked(f, data)

    def create_keypair(self):
        """
        Create an SSH key pair to be used for board creation and access. The public key will be sent to the Jenkins job that 
        creates the board, and the private key can be used to SSH into the board once it is created.
        """

        with open(self.control_file, "a+") as f:
            
            fcntl.flock(f, fcntl.LOCK_EX)

            data = self._initialize_control_data(self._read_control_data_locked(f))

            if data.get("board_private_key_path") is not None:
                raise Exception(f"Control file already contains a private key path {data['board_private_key_path']}, cannot create new key pair")

            board_tmp_key_path = tempfile.mkdtemp(prefix="key")

            board_private_key_path = f"{board_tmp_key_path}/id_ed25519"
            board_public_key_path = f"{board_tmp_key_path}/id_ed25519.pub"

            with open(os.devnull, "w") as devnull:
                subprocess.run(["ssh-keygen", "-t", "ed25519", "-f",  board_private_key_path, "-N", ""], stdout=devnull, stderr=devnull, check=True)
                
            with open(board_public_key_path, "r") as kf:
                board_public_key = kf.read()

            data["board_public_key"] = board_public_key
            data["board_private_key_path"] = board_private_key_path
            data["board_tmp_key_path"] = board_tmp_key_path
            
            self._write_control_data_locked(f, data)


    def create_boards(self, build_version, force=False):
        """
        Requests Jenkins to create a board with the given public key. The board creation will be queued.
        """

        with open(self.control_file, "a+") as f:

            fcntl.flock(f, fcntl.LOCK_EX)

            data = self._initialize_control_data(self._read_control_data_locked(f))

            if data.get("board_public_key") is None:
                raise Exception("Missing board public key in control file, cannot create board")
            
            if data.get("queue_item_url") is not None:
                raise Exception(f"Control file already contains a queue item URL {data['queue_item_url']}, cannot create new board")
            
            if data.get("build_item_url") is not None:
                raise Exception(f"Control file already contains a build item URL {data['build_item_url']}, cannot create new board")

            trigger_url = f"{self.jenkins_url}/job/{self.jenkins_job}/buildWithParameters"

            data["board_uuid"] = uuid.uuid4().hex

            payload = {
                "UUID": data["board_uuid"],
                "PublicKey": data["board_public_key"],
                "ForceUpdate": str(force).lower(),
                "BuildVersion": build_version
                }

            res = requests.post(trigger_url, data=payload, auth=(self.user, self.token))

            if res.status_code != 201:
                raise Exception(f"Failed to trigger board creation, status code: {res.status_code}, response: {res.text}")        
        
            data["queue_item_url"] = res.headers["Location"]            

            print(f"Board creation triggered successfully, queue item URL: {data['queue_item_url']}")

            self._write_control_data_locked(f, data)


    def use_existing_build(self, board_build_id: int):

        with open(self.control_file, "a+") as f:

            fcntl.flock(f, fcntl.LOCK_EX)

            data = self._initialize_control_data(self._read_control_data_locked(f))

            if data.get("queue_item_url") is not None:
                raise Exception(f"Control file already contains a queue item URL {data['queue_item_url']}, cannot use existing build")
            
            if data.get("build_item_url") is not None:
                raise Exception(f"Control file already contains a build item URL {data['build_item_url']}, cannot use existing build")
            
            data["build_item_url"] = f"{self.jenkins_url}/job/{self.jenkins_job}/{board_build_id}/"
            
            self._write_control_data_locked(f, data)
    

    def _fetch_data(self, url):
        """
        Fetch some JSON data from Jenkins
        """
        res = requests.get(url, auth=(self.user, self.token))

        if res.status_code != 200:
            raise Exception(f"Failed to get data from {url}, status code: {res.status_code}, response: {res.text}")

        return res.json()


    def _fetch_queue_item_status(self, queue_item_url):  
        """
        Fetch the status of a queue item from Jenkins.
        Queue items represent tasks that have been triggered but not yet started executing.
        """      
        return self._fetch_data(f"{queue_item_url}api/json")


    def _fetch_build_item_status(self, build_item_url):            
        """
        Fetch the status of a build item from Jenkins. 
        Build items represent tasks that are currently executing or have completed execution.
        Once a queue item starts executing, it becomes a build item and gets a build URL.
        """
        return self._fetch_data(f"{build_item_url}api/json")    


    def _fetch_ready_dut_file(self, build_item_url):
        """
        Fetch the readyDut.json file from Jenkins, which contains the IP addresses of the boards once they are ready.
        This file is expected to be an artifact of the build item.
        """
        data = self._fetch_build_item_status(build_item_url)
        
        if data.get("result") == "FAILURE":
            raise BuildFailedException(f"Board setup failed, build item at {build_item_url} has status FAILURE")

        for artifact in data.get("artifacts", []):
            if artifact["fileName"] == "readyDut.json":
                return self._fetch_data(f"{build_item_url}artifact/{artifact['relativePath']}")

        return None


    def wait_boards_ready(self):
        """
        Wait until the boards are ready by periodically checking the status of the queue item and 
        then the build item in Jenkins. Once the boards are ready, retrieve their IP addresses from
        the readyDut.json file that is generated as an artifact of the build item.
        """

        # wait until the boards item gets an executor assigned
        while True:
            with open(self.control_file, "a+") as f:

                fcntl.flock(f, fcntl.LOCK_EX)

                data = self._read_control_data_locked(f)

                if data.get("board_ips") is not None:
                    break
                
                if data.get("queue_item_url") is None:
                    break
                
                if data.get("build_item_url") is not None:
                    break

                status = self._fetch_queue_item_status(data["queue_item_url"])

                if "executable" in status:
                    print("Boards setup started")
                    data["build_item_url"] = status["executable"]["url"]                        
                    print(f"Build item URL: {data['build_item_url']}")

                    # write back the data in case we get interrupted (the queue item may expire before we are called again)
                    self._write_control_data_locked(f, data)
                    break
                    
                print("Boards setup not started yet: " + status.get("why", "N/A"))

            time.sleep(10)

        # wait until the build item generates the readyDut.json artifact with the board IPs, and write them to the control file
        while True:
            with open(self.control_file, "a+") as f:
                fcntl.flock(f, fcntl.LOCK_EX)

                data = self._read_control_data_locked(f)

                if data.get("board_ips") is not None:
                    break

                if data.get("build_item_url") is None:
                    raise Exception("Missing build item URL, cannot track board setup progress")

                ready_dut_file = self._fetch_ready_dut_file(data["build_item_url"])

                if ready_dut_file is not None:
                
                    print("Boards powered up and ready to receive connections")

                    data["board_ips"] = [dut["ip"] for dut in ready_dut_file]
                    data["board_locks"] = [ False for _ in data["board_ips"] ]
                                    
                    for dut in ready_dut_file:
                        print(f"Board {dut['hostname']} is ready with IP {dut['ip']}")

                    self._write_control_data_locked(f, data)
                    break

            print("Boards are being powered up and flashed if necessary...")
            time.sleep(15)

        if not self.is_active():
            raise BuildFailedException("Jenkins build is no longer active")


    def _release_boards(self, build_id, general_result, reason):
        url = f"{self.jenkins_url}/view/debug_jobs/job/{self.jenkins_job}/{build_id}/input/WaitForGithubPytest/submit"

        proceed_reason = json.dumps({"generalResult": general_result, "summary": reason})

        json_param = json.dumps({
            "parameter": [
                {"name": "PROCEED_REASON", "value": proceed_reason}
            ]
        })

        res = requests.post(url=url, data={"proceed": "Proceed", "json": json_param}, auth=(self.user, self.token))
        
        if res.status_code != 200:
            raise Exception(f"Failed to release board, status code: {res.status_code}, response: {res.text}")


    def force_release_build(self, build_id):
        """
        Forcefully continue a given build id, this can be used in case the control file has been lost.
        """

        self._release_boards(build_id, general_result="pass", reason="Manually forced")


    def release_boards(self, general_result="pass", reason="Boards no longer needed"):
        """
        Release the boards by sending a request to Jenkins. This will allow the boards to be reused for future requests.

        WARNING: this will release the boards even if they are currently in use, any connection will be terminated.
        """

        # we can release only boards that are ready
        build_failed = False
        try:
            self.wait_boards_ready()        
        except BuildFailedException:
            build_failed = True
        
        with open(self.control_file, "a+") as f:
            
            fcntl.flock(f, fcntl.LOCK_EX)

            data = self._read_control_data_locked(f)                

            if not build_failed:

                if data.get("build_item_url") is None:
                    print("No build item URL found, cannot release boards")
                    return

                build_id = data["build_item_url"].rstrip("/").split("/")[-1]

                self._release_boards(build_id, general_result, reason)
                
                print("Board released successfully")

            if data.get("board_tmp_key_path"):
                try:
                    shutil.rmtree(data["board_tmp_key_path"])  # clean up the temporary key pair
                except Exception as e:
                    print(f"Failed to clean up temporary key pair at {data['board_tmp_key_path']}: {e}")

            self._write_control_data_locked(f, {})  # clear the control file data

    def connect_to_board(self, board_user, board_id, private_key_path=None, verbose=False):
        remote_address = self.get_remote_address(board_user, board_id)
        cmd = ["ssh", "-o", "StrictHostKeyChecking=no"]
        if verbose:
            cmd.append("-v")
        if private_key_path:
            cmd += ["-i", private_key_path]
        cmd.append(remote_address)
        subprocess.run(cmd)


    def is_active(self):
        """
        Check that the jenkins job for the boards is still running
        """

        with open(self.control_file, "a+") as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            data = self._read_control_data_locked(f)

            if data.get("build_item_url") is None:
                return False            

            try:
                status = self._fetch_build_item_status(data["build_item_url"])
                
                result = status.get("result")

                if result is None:
                    return True
                
            except Exception as e:                
                print("Failed to fetch build item status, assuming the build is not active")
                pass
            
            return False


    def list_queue(self):
        """
        List the current all tasks queued or running on the job SL261X_NPU_Test
        """
                
        url = f"{self.jenkins_url}/job/{self.jenkins_job}/api/json?tree=builds[building,number,url,actions[parameters[name,value]],why]"
        data = self._fetch_data(url)

        for build in data.get("builds", []):
            if build["building"]:
                status = "RUNNING"
            elif build.get("why") is not None:
                status = f"QUEUED ({build['why']})"
            else:
                status = "COMPLETED"

            params = {}
            for action in build.get("actions", []):
                for param in action.get("parameters", []):
                    params[param["name"]] = param["value"]

            print(f"Build #{build['number']} - Status: {status} - UUID: {params.get('UUID', 'N/A')} - URL: {build['url']}")

    
    def find_latest_build_version(self):
        """
        Query jenkins queue to find last job that specified a build version parameter and return the value
        """

        url = f"{self.jenkins_url}/job/{self.jenkins_job}/api/json?tree=builds[number,actions[parameters[name,value]]]"
        data = self._fetch_data(url)

        latest_build_version = None
        latest_build_number = -1

        for build in data.get("builds", []):
            build_number = build.get("number", -1)
            build_version = None

            for action in build.get("actions", []):
                for param in action.get("parameters", []):
                    if param.get("name") == "BuildVersion" and param.get("value"):
                        build_version = param["value"]
                        break
                if build_version is not None:
                    break

            if build_version is not None and build_number > latest_build_number:
                latest_build_number = build_number
                latest_build_version = build_version

        return latest_build_version


class LocalAdbBoardsControl(BoardsControl):

    backend_name = "adb"

    def __init__(self, control_file: str, adb_devices: list[str] | None = None):
        super().__init__(control_file)
        self.adb_devices = list(adb_devices or [])

    def _adb_cmd_prefix(self, device_id: str | None = None) -> list[str]:
        prefix = ["adb"]
        host = os.environ.get("ADB_SERVER_HOST")
        port = os.environ.get("ADB_SERVER_PORT")
        if host:
            prefix += ["-H", host]
        if port:
            prefix += ["-P", port]
        if device_id:
            prefix += ["-s", device_id]
        return prefix

    def _discover_connected_devices(self) -> list[str]:
        result = subprocess.run(
            self._adb_cmd_prefix() + ["devices"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

        devices = []
        for line in result.stdout.strip().splitlines()[1:]:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[1] == "device":
                devices.append(parts[0])
        return devices

    def _resolve_requested_devices(self) -> list[str]:
        connected_devices = self._discover_connected_devices()
        if self.adb_devices:
            missing_devices = [device for device in self.adb_devices if device not in connected_devices]
            if missing_devices:
                raise Exception(
                    "Requested ADB devices are not connected: " + ", ".join(missing_devices)
                )
            return self.adb_devices

        if not connected_devices:
            raise Exception("No local ADB devices detected. Connect a device and run 'adb devices'.")

        return connected_devices

    def create_boards(self, build_version=None, force=False):
        if build_version is not None:
            raise Exception("--build-version is not supported for the local ADB backend")
        del force

        with open(self.control_file, "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)

            data = self._initialize_control_data(self._read_control_data_locked(f))

            if data.get("board_ips") is not None:
                raise Exception("Control file already contains local ADB boards, cannot create a new allocation")

            board_ids = self._resolve_requested_devices()
            data["board_ips"] = board_ids
            data["board_locks"] = [False for _ in board_ids]
            data["board_user"] = ""
            data["board_private_key_path"] = None

            self._write_control_data_locked(f, data)

            print("Using local ADB boards: " + ", ".join(board_ids))

    def wait_boards_ready(self):
        with open(self.control_file, "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)

            data = self._initialize_control_data(self._read_control_data_locked(f))

            if data.get("board_ips") is None:
                board_ids = self._resolve_requested_devices()
                data["board_ips"] = board_ids
                data["board_locks"] = [False for _ in board_ids]
                data["board_user"] = ""
                data["board_private_key_path"] = None
                self._write_control_data_locked(f, data)

        if not self.is_active():
            raise BuildFailedException("One or more local ADB boards are no longer connected")

    def is_active(self):
        with open(self.control_file, "a+") as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            data = self._read_control_data_locked(f)

        board_ids = data.get("board_ips")
        if not board_ids:
            return False

        connected_devices = set(self._discover_connected_devices())
        return all(board_id in connected_devices for board_id in board_ids)

    def release_boards(self, general_result="pass", reason="Boards no longer needed"):
        del general_result
        del reason

        with open(self.control_file, "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            self._write_control_data_locked(f, {})

        print("Released local ADB board allocation")

    def connect_to_board(self, board_user, board_id, private_key_path=None, verbose=False):
        del board_user
        del private_key_path
        del verbose
        cmd = self._adb_cmd_prefix(board_id)
        cmd.append("shell")
        subprocess.run(cmd)


def _detect_control_backend(control_file: str | None) -> str | None:
    if not control_file or not os.path.exists(control_file):
        return None

    try:
        with open(control_file, "r") as f:
            raw = f.read().strip()
    except OSError:
        return None

    if not raw:
        return None

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None

    return data.get("backend")


def create_boards_control(control_file: str | None, backend: str | None = None, adb_devices: list[str] | None = None) -> BoardsControl:
    resolved_backend = backend or _detect_control_backend(control_file) or "jenkins"

    if resolved_backend == "jenkins":
        return JenkinsBoardsControl(control_file)

    if resolved_backend == "adb":
        return LocalAdbBoardsControl(control_file, adb_devices=adb_devices)

    raise ValueError(f"Unknown board control backend: {resolved_backend}")



def main():
    parser = argparse.ArgumentParser(description="Script to create and manage Torq boards using Jenkins or local ADB devices.")
    
    subcommands = parser.add_subparsers(dest="command", required=True)
    backend_parent = argparse.ArgumentParser(add_help=False)
    backend_parent.add_argument("--backend", choices=("jenkins", "adb"), default=None, help="Board backend to use. Defaults to the control file backend or Jenkins.")
    backend_parent.add_argument("--adb-device", action="append", default=None, help="ADB device ID to manage for the local backend. Repeat to pin multiple devices.")

    create_parser = subcommands.add_parser("acquire", help="Acquire a new set of boards and wait until they are ready", parents=[backend_parent])    
    create_parser.add_argument("--board-build-id", type=int, help="Do not acquire a new set of boards, use the boards from this existing build ID")    
    create_parser.add_argument("--build-version", type=str, help="Optional build version to specify when creating the board, if not provided the default build will be used")
    create_parser.add_argument("--private-key-path", type=str, help="Path to an existing private key to use for board creation, if not provided a new key pair will be generated")    
    create_parser.add_argument("--control-file", required=True, help="Path to the control file with allocation info")    

    subcommands.add_parser("list-queue", help="List the current queue items in Jenkins, this can be used to check the status of the board creation request", parents=[backend_parent])

    subcommands.add_parser("latest-installed-build-version", help="Get the latest build version installed on the board from Jenkins", parents=[backend_parent])

    wait_parser = subcommands.add_parser("wait", help="Wait for the boards to be ready using the allocation info saved from a previous acquire command", parents=[backend_parent])
    wait_parser.add_argument("--control-file", required=True, help="Path to the control file with allocation info")    

    connect_parser = subcommands.add_parser("connect", help="Connect to the boards using SSH or ADB shell, this is a blocking call that will keep the connection open until interrupted", parents=[backend_parent])    
    connect_parser.add_argument("--verbose", action="store_true", help="Print more verbose output when connecting to the board")
    connect_parser.add_argument("--control-file", required=True, help="Path to the control file with allocation info")

    release_parser = subcommands.add_parser("release", help="Release the board, this should be used after the board is no longer needed to free up resources", parents=[backend_parent])
    release_parser.add_argument("--control-file", required=True, help="Path to the control file with allocation info")

    setup_boards_parser = subcommands.add_parser("setup-boards", help="Install torq runtime and kernel driver to the boards", parents=[backend_parent])
    setup_boards_parser.add_argument("--control-file", required=True, help="Path to the control file with allocation info")
    setup_boards_parser.add_argument("--local-runner-path", required=True, help="Path to the local torq-run-module binary")
    setup_boards_parser.add_argument("--local-ko-path", help="Path to the local kernel module (.ko) file")

    setup_ci_parser = subcommands.add_parser("setup-for-ci", help="Perform all steps to setup a bench for CI testing", parents=[backend_parent])
    setup_ci_parser.add_argument("--force", action="store_true", help="Force update of image even if already present on board")
    setup_ci_parser.add_argument("--control-file", required=True, help="Path to the control file with allocation info")
    setup_ci_parser.add_argument("--local-runner-path", required=True, help="Path to the local torq-run-module binary")
    setup_ci_parser.add_argument("--local-ko-path", help="Path to the local kernel module (.ko) file")
    setup_ci_parser.add_argument("--build-version", type=str, help="Optional build version to specify when creating the board, if not provided the default build will be used")

    force_release_parser = subcommands.add_parser("force-release", help="Force release a board by build ID, this can be used if the control file is lost or corrupted", parents=[backend_parent])
    force_release_parser.add_argument("--build-id", required=True, help="The build ID to force release")

    args = parser.parse_args()

    if hasattr(args, "control_file"):
        control = create_boards_control(args.control_file, backend=getattr(args, "backend", None), adb_devices=getattr(args, "adb_device", None))
    else:
        control = create_boards_control(None, backend=getattr(args, "backend", None), adb_devices=getattr(args, "adb_device", None))

    if args.command == "acquire":
        if isinstance(control, JenkinsBoardsControl):
            if not args.build_version:
                raise Exception("--build-version is required for the Jenkins backend")

            if args.private_key_path:
                print(f"Using existing key pair with private key at {args.private_key_path}...")
                control.load_keypair(args.private_key_path)
            else:
                print("Creating key pair...")
                control.create_keypair()

            if args.board_build_id:
                print(f"Using existing boards build with ID {args.board_build_id}...")
                control.use_existing_build(args.board_build_id)
            else:
                print("Acquiring boards...")
                control.create_boards(args.build_version)
        else:
            if args.private_key_path or args.board_build_id:
                raise Exception("--private-key-path and --board-build-id are only supported for the Jenkins backend")
            print("Acquiring local ADB boards...")
            control.create_boards()

    elif args.command == "setup-for-ci":
        if isinstance(control, JenkinsBoardsControl):
            if not args.build_version:
                raise Exception("--build-version is required for the Jenkins backend")

            print("Creating key pair...")
            control.create_keypair()

            print("Creating board...")
            control.create_boards(args.build_version, args.force)
        else:
            print("Acquiring local ADB boards...")
            control.create_boards()
        
        try:
            print("Waiting for board to be ready...")
            control.wait_boards_ready()

            print("Setting up board...")
            control.setup_boards(args.local_runner_path, args.local_ko_path)

            print("Board setup for CI completed successfully")

        except BaseException:
            print("Boards creation job")
            control.release_boards()
            raise
        
    elif args.command == "list-queue":
        if not isinstance(control, JenkinsBoardsControl):
            raise Exception("list-queue is only supported for the Jenkins backend")
        control.list_queue()

    elif args.command == "latest-installed-build-version":
        if not isinstance(control, JenkinsBoardsControl):
            raise Exception("latest-installed-build-version is only supported for the Jenkins backend")
        latest_build_version = control.find_latest_build_version()
        if latest_build_version:
            print(f"Latest installed build version: {latest_build_version}")
        else:
            print("No build version found in recent builds")

    elif args.command == "wait":
        try:
            control.wait_boards_ready()
        except BuildFailedException:
            print("Board setup failed")
            exit(1)

    elif args.command == "connect":        

        with control.board() as (board_user, board_ip, board_private_key_path):            

            print(f"Connecting to board {board_ip}...")
            control.connect_to_board(board_user, board_ip, board_private_key_path, verbose=args.verbose)

    elif args.command == "release":                    
        control.release_boards()
        os.unlink(args.control_file)

    elif args.command == "setup-boards":
        control.setup_boards(args.local_runner_path, args.local_ko_path)

    elif args.command == "force-release":
        if not isinstance(control, JenkinsBoardsControl):
            raise Exception("force-release is only supported for the Jenkins backend")
        control.force_release_build(args.build_id)

    else:
        raise Exception(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()