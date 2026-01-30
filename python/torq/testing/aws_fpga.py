from filelock import FileLock
import subprocess
from contextlib import contextmanager
from ..utils.remote_runner import SSHCommandRunner

def load_bitstream(agfi, bandwidth, force=False, runner: SSHCommandRunner | None = None):

    print(f"Loading FPGA bitstream {agfi} with bandwidth {bandwidth}...")
    
    def run(cmd):
        if runner:
            return runner.run_cmd(cmd)
        else:
            return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout

    if not force:
        # find the currently loaded AGFI
        result = run(["fpga-describe-local-image", "-S", "0"])
        if agfi in result:
            print("Bitstream already loaded.")
            return

    # load the bitstream
    run(["fpga-load-local-image", "-S", "0", "-I", agfi])

    # setup clocks
    run(["fpga-load-clkgen-recipe", "-S", "0", "-b", str(bandwidth)])

    # display current status
    clk_status = run(["fpga-describe-clkgen", "-S", "0"])

    lines = clk_status.splitlines()
    rate = None

    for i, line in enumerate(lines):
        if "Clock Group B Frequency" in line:
            for j in range(i + 1, len(lines)):
                if "clk_extra_b1" in lines[j]:
                    if j + 2 < len(lines):
                        values_line = lines[j + 2]
                        values = [v.strip() for v in values_line.strip('|').split('|')]
                        if len(values) > 1:
                            rate = values[1]
                    break
            break

    print(f"Configured clock at {rate} MHz")


@contextmanager
def FpgaSession(fpga_configuration):

    with FileLock("/tmp/fpga.lock"):
        try:
            load_bitstream(fpga_configuration["agfi"], fpga_configuration["bandwidth"])
            yield None
        except:
            load_bitstream(fpga_configuration["agfi"], fpga_configuration["bandwidth"], force=True)
            raise
        finally:
            pass

@contextmanager
def RemoteFpgaSession(fpga_configuration, remote_address, remote_port):

    with FileLock("/tmp/fpga.lock"):
        runner = SSHCommandRunner(remote_address, timeout=10*60, port=remote_port)
        try:
            load_bitstream(fpga_configuration["agfi"], fpga_configuration["bandwidth"], force=False, runner=runner)
            yield None
        except:
            load_bitstream(fpga_configuration["agfi"], fpga_configuration["bandwidth"], force=True, runner=runner)
            raise
        finally:
            pass
