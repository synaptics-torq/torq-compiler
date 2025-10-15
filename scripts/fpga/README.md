# AWS FPGA testing environment setup

## Setup your IREE environment

See https://github.com/syna-astra-dev/iree-synaptics-synpu/blob/main/doc/development_guidelines.md

### Assuming that your environment is ready for that, generate a test model to the install location

```{code} shell
pytest ./iree-synaptics-synpu/tests/test_torqrt.py -v -k conv-stride1-56.mlir --basetemp=fpga/install
```

*Warning*: This last command will erase the content of fpga/install if it exists.

## Compile IREE runtime for FPGA environment 

### Create docker for AWS FPGA platform

```{code} shell
docker build -t fpga -f .github/workflows/Dockerfile.fpga .
```

First time will take a few minutes, following ones will be fast if you don´t modify the docker file.

### Run fpga docker interactively
This docker assumes that iree and this plugin source trees are alongside and it will place you in the iree source directory.

```{code} shell
cd .. # we are in the parent directory containing iree and this plugin
docker run -it --rm -v $(pwd):$(pwd) -u $(id -u):$(id -g) -w $(pwd) fpga
```

### Build IREE and torq_rt runtime within the fpga docker shell and exit

```{code} shell
./iree-synaptics-synpu/scripts/build_iree_rt.sh fpga/iree-build fpga/install
./iree-synaptics-synpu/scripts/build_torq_rt.sh fpga/torq_rt-build fpga/install
exit # exit the docker
```
./test_tosa_conv_stride1_56_mlir0/descriptors/compiler/main_di
First argument of both scripts is the build folder, the second is the install folder.
You can verify the correct installation:

```{code} shell
tree -L 2 fpga/install/
```
Which will give something like this:

fpga/install/
├── bin
│   ├── iree-run-module
│   ├── torq_rt_aws
│   └── torq_rt_cm
├── test_tosa_conv_stride1_56_mlir0
│   ├── descriptors
│   ├── in_rnd_0.npy
│   ├── output_torq.npy
│   ├── output_torq.vmfb
│   ├── output_torq.vmfb-phases
│   └── rt
└── test_tosa_conv_stride1_56_mlircurrent -> /home/ayounes/torq-dev/fpga/install/test_tosa_conv_stride1_56_mlir0

# archive the artifacts

```{code} shell
zip -qr fpga.zip fpga/install
```

## Prepare FPGA testing environment

### Enable prompt-less login in AWS FPGA server (only once)

To avoid login and password prompt you can add your ssh public key (~/.ssh/id_rsa.pub) to the authorized keys.
Assuming you have the same user as your Windows login you can do this for example:

```{code} shell
scp ~/.ssh/id_rsa.pub 10.128.81.84:/home/$USER/.ssh/authorized_keys
```

### Transfer the iree-run-module binary to the AWS FPGA server

```{code} shell
scp fpga.zip 10.128.81.84:/home/$USER/
```

## Test on FPGA platform

### Login to AWS FPGA server and unzip the package

```{code} shell
ssh 10.128.81.84
# You should see a user prompt like this:
# [user@ip-10-128-81-84 ~]$
unzip fpga.zip
cd fpga/install/
```

### Test the model using Cmodel simulator

This will execute the transfered model with a random input. If you the input shape is wrong iree will tell you the correct one.

```{code} shell
../iree-run-module --device=torq --module=./output_torq.vmfb --input=@./in_rnd_0.npy \
    --output=@./cm/out-iree.npy --torq_dump_mem_data_dir=cm
```

Upon success this will create a new cm/ directory

You can also test the HW test vector using the torq_rt_cm tool:

```{code} shell
./bin/torq_rt_cm -i ./main_dispatch_0 -o out_cm
```

Upon success this will create a new out/ directory

### load the FPGA image

```{code} shell
/home/share/bin/load_fpga /home/share/afi/torq-fpga-latest
```

### Tell iree to run against FPGA and execute the model again 

```{code} shell
../iree-run-module --device=torq --module=./output_torq.vmfb --input=@./in_rnd_0.npy \
  --output=@./aws/out-iree.npy --torq_dump_mem_data_dir=aws --torq_hw_type=aws_fpga
```
