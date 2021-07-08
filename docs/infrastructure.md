# Reference for setting up related infrastructure.

Although I set up my infrastructure using terraform and configs in a seperate repo, the following notes may be useful for others.

### terraform configs:
```yaml
resource "aws_instance" "gpu" {
  ami = "ami-00cc0fa9f9988be83" # US-west-2 AWS Deep Learning AMI (Amazon Linux 2)
  instance_type = "g4dn.xlarge"#=4vCPU, 16G RAM, 125 SSD,1_NVIDIAT4_16GB $0.53/hr
  ....
  }
}
```

### Setting up EC2:
```bash
sudo yum update -y
sudo yum install -y emacs-nox.x86_64 tmux htop
pip3 install --upgrade --user awscli
aws configure


# log into AWS ECR so that we can retrieve the deeplearning image:
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-west-2.amazonaws.com

# run image shell:
docker run -it -w=/local -v $PWD:/local 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:1.8.1-gpu-py36-cu111-ubuntu18.04 /bin/bash

# While model is training. To see if GPU is being used:
nvidia-smi
nvidia-smi -i 0 -l -q -d UTILIZATION
```
