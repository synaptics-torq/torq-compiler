from filelock import FileLock
import subprocess
from contextlib import contextmanager

def load_bitstream(agfi, bandwidth, force=False):

    print(f"Loading FPGA bitstream {agfi} with bandwidth {bandwidth}...")

    if not force:
        # find the currently loaded AGFI
        result = subprocess.run(["fpga-describe-local-image", "-S", "0"], capture_output=True, text=True)
        if agfi in result.stdout:
            print("Bitstream already loaded.")
            return

    # load the bitstream
    subprocess.run(["fpga-load-local-image", "-S", "0", "-I", agfi], check=True)

    # setup clocks
    subprocess.run(["fpga-load-clkgen-recipe", "-S", "0", "-b", str(bandwidth)], check=True)

    # display current status
    clk_status = subprocess.run(["fpga-describe-clkgen", "-S", "0"], check=True, capture_output=True, text=True)

    lines = clk_status.stdout.splitlines()
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