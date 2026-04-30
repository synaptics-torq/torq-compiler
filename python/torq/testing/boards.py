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



class JenkinsBoardsControl:

    def __init__(self, control_file: str):
        self.user = os.environ.get("JENKINS_USER")
        self.token = os.environ.get("JENKINS_API_TOKEN")
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


    def load_keypair(self, private_key_path):
        """
        Load an existing SSH key pair from the given private key path. The public key is expected to be in the same directory with the same name but with .pub extension.
        """        

        with open(self.control_file, "a+") as f:

            fcntl.flock(f, fcntl.LOCK_EX)

            data = self._read_control_data_locked(f)

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

            data = self._read_control_data_locked(f)

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


    def create_board(self, build_version=None, force=False):
        """
        Requests Jenkins to create a board with the given public key. The board creation will be queued.
        """

        with open(self.control_file, "a+") as f:

            fcntl.flock(f, fcntl.LOCK_EX)

            data = self._read_control_data_locked(f)

            if data.get("board_public_key") is None:
                raise Exception("Missing board public key in control file, cannot create board")
            
            if data.get("queue_item_url") is not None:
                raise Exception(f"Control file already contains a queue item URL {data['queue_item_url']}, cannot create new board")
            
            if data.get("build_item_url") is not None:
                raise Exception(f"Control file already contains a build item URL {data['build_item_url']}, cannot create new board")

            trigger_url = "http://devops.synaptics.com:8080/jenkins/job/SL261X_NPU_Test/buildWithParameters"    

            data["board_uuid"] = uuid.uuid4().hex

            payload = {"UUID": data["board_uuid"], "PublicKey": data["board_public_key"]}

            if not force:
                current_version = self.find_latest_build_version()

                if current_version == build_version:
                    print(f"Requested build version {build_version} is the same as the current latest installed build version, skipping board creation")
                    build_version = None  # do not specify the build version to use the default one, which should be the same as the current latest 

            if build_version is not None:
                payload["BuildVersion"] = build_version   

            res = requests.post(trigger_url, data=payload, auth=(self.user, self.token))

            if res.status_code != 201:
                raise Exception(f"Failed to trigger board creation, status code: {res.status_code}, response: {res.text}")        
        
            data["queue_item_url"] = res.headers["Location"]            

            print(f"Board creation triggered successfully, queue item URL: {data['queue_item_url']}")

            self._write_control_data_locked(f, data)


    def use_existing_build(self, board_build_id: int):

        with open(self.control_file, "a+") as f:

            fcntl.flock(f, fcntl.LOCK_EX)

            data = self._read_control_data_locked(f)

            if data.get("queue_item_url") is not None:
                raise Exception(f"Control file already contains a queue item URL {data['queue_item_url']}, cannot use existing build")
            
            if data.get("build_item_url") is not None:
                raise Exception(f"Control file already contains a build item URL {data['build_item_url']}, cannot use existing build")
            
            data["build_item_url"] = f"http://devops.synaptics.com:8080/jenkins/job/SL261X_NPU_Test/{board_build_id}/"
            
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
                    print("Board IPs already present in control file, assuming boards are ready")
                    return
                
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

                if data.get("build_item_url") is None:
                    raise Exception("Missing build item URL, cannot track board setup progress")

                ready_dut_file = self._fetch_ready_dut_file(data["build_item_url"])

                if ready_dut_file is not None:
                
                    print("Boards setup completed")

                    data["board_ips"] = [dut["ip"] for dut in ready_dut_file]
                    data["board_locks"] = [ False for _ in data["board_ips"] ]
                                    
                    for dut in ready_dut_file:
                        print(f"Board {dut['hostname']} is ready with IP {dut['ip']}")

                    self._write_control_data_locked(f, data)
                    break

            print("Boards are being set up...")
            time.sleep(30)


    def release_boards(self, general_result="pass", reason="Boards no longer needed", force=False):
        """
        Release the boards by sending a request to Jenkins. This will allow the boards to be reused for future requests.

        WARNING: this will release the boards even if they are currently in use, any connection will be terminated.
        """

        # we can release only boards that are ready
        self.wait_boards_ready()        
        
        with open(self.control_file, "a+") as f:
            
            fcntl.flock(f, fcntl.LOCK_EX)

            data = self._read_control_data_locked(f)                

            if data.get("build_item_url") is None:
                print("No build item URL found, cannot release boards")
                return

            build_id = data["build_item_url"].rstrip("/").split("/")[-1]

            url = f"http://devops.synaptics.com:8080/jenkins/view/debug_jobs/job/SL261X_NPU_Test/{build_id}/input/WaitForGithubPytest/submit"

            proceed_reason = json.dumps({"generalResult": general_result, "summary": reason})

            json_param = json.dumps({
                "parameter": [
                    {"name": "PROCEED_REASON", "value": proceed_reason}
                ]
            })

            res = requests.post(url=url, data={"proceed": "Proceed", "json": json_param}, auth=(self.user, self.token))
            
            if res.status_code != 200:
                raise Exception(f"Failed to release board, status code: {res.status_code}, response: {res.text}")
            
            print("Board released successfully")

            if data.get("board_tmp_key_path"):
                shutil.rmtree(data["board_tmp_key_path"])  # clean up the temporary key pair

            self._write_control_data_locked(f, {})  # clear the control file data


    def get_board(self):
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
                        return ip, data["board_private_key_path"]                
            
            print("All boards are currently in use, waiting for a board to be available...")
            time.sleep(10)


    def board(self):
        """
        Context manager that acquires a board on entry and releases it on exit.

        Usage::

            with control.board() as ip:
                # use the board at `ip`
        """
        from contextlib import contextmanager

        @contextmanager
        def _board_ctx():
            ip, private_key = self.get_board()
            try:
                yield ip, private_key
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


    def list_queue(self):
        """
        List the current all tasks queued or running on the job SL261X_NPU_Test
        """
        
        url = "http://devops.synaptics.com:8080/jenkins/job/SL261X_NPU_Test/api/json?tree=builds[building,number,url,actions[parameters[name,value]],why]"
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

        url = "http://devops.synaptics.com:8080/jenkins/job/SL261X_NPU_Test/api/json?tree=builds[number,actions[parameters[name,value]]]"
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



def main():
    parser = argparse.ArgumentParser(description="Script to create a board on the Jenkins board farm and retrieve its status.")
    
    subcommands = parser.add_subparsers(dest="command", required=True)

    create_parser = subcommands.add_parser("acquire", help="Acquire a new set of boards and wait until they are ready")    
    create_parser.add_argument("--board-build-id", type=int, help="Do not acquire a new set of boards, use the boards from this existing build ID")    
    create_parser.add_argument("--build-version", type=str, help="Optional build version to specify when creating the board, if not provided the default build will be used")
    create_parser.add_argument("--private-key-path", type=str, help="Path to an existing private key to use for board creation, if not provided a new key pair will be generated")    
    create_parser.add_argument("--control-file", required=True, help="Path to the control file with allocation info")    

    subcommands.add_parser("list-queue", help="List the current queue items in Jenkins, this can be used to check the status of the board creation request")

    subcommands.add_parser("latest-installed-build-version", help="Get the latest build version installed on the board from Jenkins")

    wait_parser = subcommands.add_parser("wait", help="Wait for the boards to be ready using the allocation info saved from a previous acquire command")
    wait_parser.add_argument("--control-file", required=True, help="Path to the control file with allocation info")    

    connect_parser = subcommands.add_parser("connect", help="Connect to the boards using SSH, this is a blocking call that will keep the connection open until interrupted")    
    connect_parser.add_argument("--verbose", action="store_true", help="Print more verbose output when connecting to the board")
    connect_parser.add_argument("--control-file", required=True, help="Path to the control file with allocation info")

    release_parser = subcommands.add_parser("release", help="Release the board, this should be used after the board is no longer needed to free up resources")
    release_parser.add_argument("--control-file", required=True, help="Path to the control file with allocation info")
    

    args = parser.parse_args()

    if hasattr(args, "control_file"):
        control = JenkinsBoardsControl(args.control_file)
    else:
        control = JenkinsBoardsControl(None)

    if args.command == "acquire":
        
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
            print("Creating board...")
            control.create_board(args.build_version)    

    elif args.command == "list-queue":
        control.list_queue()

    elif args.command == "latest-installed-build-version":
        latest_build_version = control.find_latest_build_version()
        if latest_build_version:
            print(f"Latest installed build version: {latest_build_version}")
        else:
            print("No build version found in recent builds")

    elif args.command == "wait":
        control.wait_boards_ready()

    elif args.command == "connect":        

        with control.board() as (board_ip, board_private_key_path):            

            print(f"Connecting to board with IP {board_ip} using private key at {board_private_key_path}...")

            if args.verbose:
                subprocess.run(["ssh", "-o", "StrictHostKeyChecking=no", "-v", "-i", board_private_key_path, f"root@{board_ip}"])
            else:
                subprocess.run(["ssh", "-o", "StrictHostKeyChecking=no", "-i", board_private_key_path, f"root@{board_ip}"])

    elif args.command == "release":                    
        control.release_boards()
        os.unlink(args.control_file)

    else:
        raise Exception(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()