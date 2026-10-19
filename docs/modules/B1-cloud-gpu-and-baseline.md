# B1 · Cloud GPU and full-information baselines

**Build track, after the proposal.** **Prerequisites:** [P0](P0-setup.md) (repo skeleton, local environment, `scripts/check_env.py`, commits pushed) and [P5](P5-which-model.md)–[P7](P7-why-appropriate.md) (`text.py`, `data.py`, `model.py`, `reveal.py`, `train_rdl.py`, all committed).

**Goal:** a GPU instance on AWS running the same code as your laptop, and the "fetch everything" baseline on the two primary datasets, `rel-event` and `rel-avito`, which are too big for the laptop: the upper bound every lookup budget is measured against.

You type every line yourself. Every block explains what it does and why. Each level ends with a **PASS** check (what you should see) and a **STOP** rule (what to do if you don't). Paste STOP output back into the chat.

Conventions:

- `PS>` blocks run in **PowerShell on your Windows laptop**, from the repo root.
- `ubuntu$` blocks run **on the EC2 instance** (VS Code Remote-SSH terminal).
- Replace anything in `<angle brackets>`.

---

## Level 0 — AWS account, CLI, and guardrails (laptop, ~40 min)

### 0.1 Make `aws` mean AWS CLI v2

In PowerShell, `aws` currently resolves to a broken Miniconda shim (`miniconda3\Scripts\aws.cmd`, which fails with `No module named 'awscli'`). AWS CLI v2 is already installed, just later on your PATH. Renaming the broken shims (reversible) makes v2 the default everywhere, with no profile scripts and no execution-policy changes.

```powershell
# Rename the two broken Miniconda shims so PATH falls through to AWS CLI v2.
# Reversible: rename them back if you ever need the old one.
Rename-Item "$env:USERPROFILE\miniconda3\Scripts\aws.cmd" "aws.cmd.broken"
Rename-Item "$env:USERPROFILE\miniconda3\Scripts\aws" "aws.broken"

# Confirm which executable PowerShell now finds, and its version.
(Get-Command aws).Source
aws --version
```

**PASS:** the path ends in `Amazon\AWSCLIV2\aws.exe` and the version starts with `aws-cli/2.`
**STOP:** anything else.

### 0.2 Sign in with a dedicated profile

`aws login` (CLI v2) opens a browser, uses your normal console sign-in, and stores short-lived credentials. No long-lived access keys sit on disk.

```powershell
# Create/refresh a profile named "wtl" (Worth The Lookup) via browser sign-in.
aws login --profile wtl

# Pin the region for this profile: every command below uses us-west-2 (Oregon).
aws configure set region us-west-2 --profile wtl

# Make "wtl" the active profile for this PowerShell window only.
$env:AWS_PROFILE = "wtl"

# Who am I? Prints account ID and caller ARN.
aws sts get-caller-identity

# Save the 12-digit account ID in a variable; later commands reuse it.
$ACCOUNT = aws sts get-caller-identity --query Account --output text
$ACCOUNT
```

**PASS:** `get-caller-identity` prints your account ID.
**Note:** if the ARN ends in `:root`, it works for tonight, but create an admin IAM Identity Center user this week. Root credentials shouldn't drive day-to-day work.
**STOP:** `aws login` is not recognized, or sign-in fails.

> Every new PowerShell window needs `$env:AWS_PROFILE = "wtl"` and the `$ACCOUNT` line again.

### 0.3 Check GPU quota first (it is the slowest thing to fix)

This quota **defaults to 0**, counted in vCPUs. `g6.2xlarge` (1× L4 GPU, 8 vCPU, 32 GiB RAM, ~$0.98/hr) needs 8.

```powershell
# L-DB2E81BA = "Running On-Demand G and VT instances" (vCPU count), EC2 in us-west-2.
aws service-quotas get-service-quota --service-code ec2 --quota-code L-DB2E81BA --query "Quota.Value"
```

If the value is **less than 8**, request exactly 8 now. Small requests are approved fastest (minutes to a day):

```powershell
# Ask for 8 G/VT vCPUs: one g6.2xlarge.
aws service-quotas request-service-quota-increase --service-code ec2 --quota-code L-DB2E81BA --desired-value 8 --query "RequestedQuota.Status"
```

**PASS:** quota ≥ 8, or the status prints `"PENDING"` / `"CASE_OPENED"`. If you're waiting on approval, Level 1 has a CPU fallback for tonight.

### 0.4 Budget alarm that counts credit-funded spend

A default AWS Budget includes credits, so with credits active your net cost reads ~$0 and it never fires. We exclude credits so it tracks real consumption.

Create the folder `infra/aws/` in VS Code, then create these two files.

`infra/aws/budget.json`:

```json
{
  "BudgetName": "wtl-monthly",
  "BudgetLimit": { "Amount": "300", "Unit": "USD" },
  "TimeUnit": "MONTHLY",
  "BudgetType": "COST",
  "CostTypes": { "IncludeCredit": false, "IncludeRefund": false }
}
```

- `"BudgetLimit"`: $300/month is far above what this project should spend, so any alert means something is left running.
- `"IncludeCredit": false`: measure gross usage, not what's left after credits.

`infra/aws/notify.local.json` (`*.local.json` is git-ignored because it holds your email):

```json
[
  {
    "Notification": { "NotificationType": "ACTUAL", "ComparisonOperator": "GREATER_THAN", "Threshold": 25, "ThresholdType": "PERCENTAGE" },
    "Subscribers": [ { "SubscriptionType": "EMAIL", "Address": "<your-email>" } ]
  },
  {
    "Notification": { "NotificationType": "FORECASTED", "ComparisonOperator": "GREATER_THAN", "Threshold": 100, "ThresholdType": "PERCENTAGE" },
    "Subscribers": [ { "SubscriptionType": "EMAIL", "Address": "<your-email>" } ]
  }
]
```

- First alert: actual spend passes 25% ($75).
- Second alert: AWS forecasts you'll exceed the full $300 this month.

```powershell
# Create the budget and attach both email alerts.
aws budgets create-budget --account-id $ACCOUNT --budget file://infra/aws/budget.json --notifications-with-subscribers file://infra/aws/notify.local.json

# Read it back to confirm it exists.
aws budgets describe-budget --account-id $ACCOUNT --budget-name wtl-monthly --query "Budget.[BudgetName,BudgetLimit.Amount]"
```

**PASS:** prints `["wtl-monthly", "300.0"]` (or similar).

### 0.5 Private S3 bucket for results

```powershell
# Bucket names are global, so include your account ID to make the name unique.
$BUCKET = "wtl-$ACCOUNT-usw2"

# Create it in us-west-2. Outside us-east-1, the LocationConstraint must be given explicitly.
aws s3api create-bucket --bucket $BUCKET --region us-west-2 --create-bucket-configuration LocationConstraint=us-west-2

# New buckets block all public access by default. Verify (all four should be true).
aws s3api get-public-access-block --bucket $BUCKET
```

**PASS:** four `true` values.

### 0.6 IAM role the instance uses (no keys on the server)

The EC2 instance gets temporary credentials from this role, so you never copy keys onto it. It may only touch **your one bucket**.

`infra/aws/ec2-trust.json` (who may assume the role: the EC2 service):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    { "Effect": "Allow", "Principal": { "Service": "ec2.amazonaws.com" }, "Action": "sts:AssumeRole" }
  ]
}
```

`infra/aws/wtl-s3-policy.json` (what the role may do). Replace `<ACCOUNT>` with your 12 digits:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    { "Effect": "Allow", "Action": ["s3:ListBucket"], "Resource": "arn:aws:s3:::wtl-<ACCOUNT>-usw2" },
    { "Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"], "Resource": "arn:aws:s3:::wtl-<ACCOUNT>-usw2/*" }
  ]
}
```

- Statement 1 permits listing the bucket itself.
- Statement 2 permits reading, writing, and deleting objects inside it, and nothing else in your account.

```powershell
# 1) Create the role with the trust policy.
aws iam create-role --role-name wtl-ec2-role --assume-role-policy-document file://infra/aws/ec2-trust.json --query "Role.Arn"

# 2) Attach our bucket-only S3 permissions as an inline policy.
aws iam put-role-policy --role-name wtl-ec2-role --policy-name wtl-s3 --policy-document file://infra/aws/wtl-s3-policy.json

# 3) Attach AWS's managed policy for Session Manager: a backup way in if SSH breaks.
aws iam attach-role-policy --role-name wtl-ec2-role --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore

# 4) EC2 attaches roles through an "instance profile" wrapper. Create it...
aws iam create-instance-profile --instance-profile-name wtl-ec2-profile

# 5) ...and put the role inside it.
aws iam add-role-to-instance-profile --instance-profile-name wtl-ec2-profile --role-name wtl-ec2-role
```

**PASS:** each command returns without an error.

### 0.7 SSH key pair and a firewall that only admits your IP

```powershell
# Make sure ~/.ssh exists.
New-Item -ItemType Directory -Force "$HOME\.ssh" | Out-Null
$KEY = "$HOME\.ssh\wtl-key.pem"

# Ask AWS to create an ed25519 key pair; AWS keeps the public half, we get the private half once.
$keyText = aws ec2 create-key-pair --key-name wtl-key --key-type ed25519 --query KeyMaterial --output text

# Write it with Unix newlines and no BOM; OpenSSH rejects some UTF-16/BOM/CRLF key files.
[IO.File]::WriteAllText($KEY, (($keyText -join "`n") + "`n"))

# Lock the file to your user only; Windows OpenSSH refuses keys others can read.
icacls $KEY /inheritance:r /grant:r "$($env:USERNAME):(R)"

# Find the default VPC (the network new instances join).
$VPC = aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query "Vpcs[0].VpcId" --output text
$VPC

# Create a security group (instance firewall) in that VPC.
$SG = aws ec2 create-security-group --group-name wtl-ssh --description "SSH from my IP only" --vpc-id $VPC --query GroupId --output text

# Look up your current public IP...
$MYIP = (Invoke-RestMethod https://checkip.amazonaws.com).Trim()

# ...and allow SSH (port 22) from exactly that address (/32 = one IP).
aws ec2 authorize-security-group-ingress --group-id $SG --protocol tcp --port 22 --cidr "$MYIP/32"
```

**PASS:** `$VPC` prints `vpc-…`, and the last command prints `"Return": true`.
**STOP:** `$VPC` prints `None` (no default VPC). Tell me and we'll create one.

> If your home IP changes later, SSH will time out. Re-run the last two lines with the new IP.

**Level 0 done.** You have an identity, a spend alarm, private storage, a least-privilege role, a key, and a firewall. Nothing costs money yet.

---

## Level 1 — Launch the GPU instance (laptop, ~20 min)

The instance clones your repo from GitHub, so make sure your P0–P8 commits are pushed. Commit anything new from Level 0 first (`infra/aws/*.json`; the `*.local.json` file stays ignored):

```powershell
git status            # NO .pem or *.local.json files should appear
git add infra/aws
git commit -m "Add AWS policy files"
git push
```

### 1.1 Variables for a fresh PowerShell window

Run this block at the start of **every** new laptop window:

```powershell
# Use the wtl profile and recompute the names we created in Level 0.
$env:AWS_PROFILE = "wtl"
$ACCOUNT = aws sts get-caller-identity --query Account --output text
$BUCKET = "wtl-$ACCOUNT-usw2"

# Look up our security group by name instead of remembering its ID.
$SG = aws ec2 describe-security-groups --filters Name=group-name,Values=wtl-ssh --query "SecurityGroups[0].GroupId" --output text

# Look up our instance by its Name tag. Prints None before the first launch.
$IID = aws ec2 describe-instances --filters Name=tag:Name,Values=wtl-gpu Name=instance-state-name,Values=pending,running,stopping,stopped --query "Reservations[0].Instances[0].InstanceId" --output text
"$ACCOUNT $BUCKET $SG $IID"
```

### 1.2 Pick the machine image

The AWS Deep Learning AMI ships the NVIDIA driver and CUDA. The newest one is PyTorch 2.14 on Ubuntu 26.04; AWS flags older PyTorch AMIs as having unpatchable CVEs. We use it for its **driver**, and install our own pinned Python environment on top in Level 2.

```powershell
# AWS publishes the latest AMI ID for this image family in a public SSM parameter.
$AMI = aws ssm get-parameter --name /aws/service/deeplearning/ami/x86_64/oss-nvidia-driver-gpu-pytorch-2.14-ubuntu-26.04/latest/ami-id --query Parameter.Value --output text

# The image's root disk device name; the disk-size file below must use it.
$ROOTDEV = aws ec2 describe-images --image-ids $AMI --query "Images[0].RootDeviceName" --output text
"$AMI $ROOTDEV"
```

**PASS:** `ami-… /dev/sda1`.

`infra/aws/block-device.json` makes a 200 GB encrypted gp3 root disk (datasets, Hugging Face cache, checkpoints). If `$ROOTDEV` isn't `/dev/sda1`, use its value:

```json
[
  { "DeviceName": "/dev/sda1", "Ebs": { "VolumeSize": 200, "VolumeType": "gp3", "Encrypted": true, "DeleteOnTermination": true } }
]
```

### 1.3 Launch

```powershell
# Each backtick (`) continues the command on the next line; it must be the last character.
# HttpTokens=required forces IMDSv2 (the hardened metadata service).
$IID = aws ec2 run-instances `
  --image-id $AMI `
  --instance-type g6.2xlarge `
  --key-name wtl-key `
  --security-group-ids $SG `
  --iam-instance-profile Name=wtl-ec2-profile `
  --block-device-mappings file://infra/aws/block-device.json `
  --metadata-options HttpTokens=required `
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=wtl-gpu},{Key=project,Value=wtl}]" `
  --query "Instances[0].InstanceId" --output text

# Block until AWS reports the instance as running.
aws ec2 wait instance-running --instance-ids $IID
$IID
```

- `--instance-type g6.2xlarge`: 1× L4 24 GB, 8 vCPU, 32 GiB.
- `--key-name wtl-key`: the SSH key from 0.7.
- `--security-group-ids $SG`: the firewall from 0.7.
- `--iam-instance-profile`: the bucket-only role from 0.6.
- `--block-device-mappings`: the 200 GB disk.
- `--tag-specifications`: the tags 1.1 uses to find the instance.

**STOP:**

- `VcpuLimitExceeded`: the quota isn't approved yet. Use the **CPU fallback**: re-run with `--instance-type m7i.2xlarge` (8 vCPU, 32 GiB, no GPU). Everything tonight works on CPU, just slower. Switch to `g6.2xlarge` later with a stop / modify / start.
- `Invalid IAM Instance Profile`: IAM is still propagating. Wait 30 s and retry.

### 1.4 Stable address, idle auto-stop, SSH shortcut

```powershell
# Allocate a fixed public IP (Elastic IP) so the SSH config never changes across stop/start.
$ALLOC = aws ec2 allocate-address --domain vpc --tag-specifications "ResourceType=elastic-ip,Tags=[{Key=project,Value=wtl}]" --query AllocationId --output text
aws ec2 associate-address --instance-id $IID --allocation-id $ALLOC
$IP = aws ec2 describe-addresses --allocation-ids $ALLOC --query "Addresses[0].PublicIp" --output text

# Auto-stop: if max CPU stays under 3% for 12 x 5 min (= 1 hour), CloudWatch stops the box.
# Training keeps at least one core busy, so this only fires when you have truly walked away.
aws cloudwatch put-metric-alarm --alarm-name wtl-idle-stop --namespace AWS/EC2 --metric-name CPUUtilization `
  --dimensions Name=InstanceId,Value=$IID --statistic Maximum --period 300 --evaluation-periods 12 `
  --threshold 3 --comparison-operator LessThanThreshold --alarm-actions arn:aws:automate:us-west-2:ec2:stop

# Add a "wtl" host alias to your SSH config (ASCII, so OpenSSH can read it).
Add-Content -Encoding ascii -Path "$HOME\.ssh\config" -Value "`nHost wtl`n    HostName $IP`n    User ubuntu`n    IdentityFile ~/.ssh/wtl-key.pem`n    ServerAliveInterval 60`n"

# First connection: answer "yes" to the host fingerprint, then check the GPU.
ssh wtl nvidia-smi
```

**PASS:** `nvidia-smi` prints a table showing an **NVIDIA L4** (on the CPU fallback it errors; that's fine).

**Daily habits.** When you stop for the night (unless a job is running), run `aws ec2 stop-instances --instance-ids $IID`. To resume, run `aws ec2 start-instances --instance-ids $IID`. A stopped instance costs only its disk (~$16/month) plus the Elastic IP (~$4/month).

### 1.5 Open the instance in VS Code

1. Install the **Remote - SSH** extension.
2. `Ctrl+Shift+P` → **Remote-SSH: Connect to Host…** → `wtl`.
3. Open a terminal in that window (``Ctrl+` ``). Everything in Levels 2–4 runs there.

---

## Level 2 — Python environment on the instance (~15 min)

We pin the exact stack RelBench's own GPU environment is tested with: **PyTorch 2.9.1 + CUDA 12.8**, Python 3.12. The AMI's newer driver runs CUDA 12.8 code fine. Why not the AMI's preinstalled torch 2.14? Nobody has tested it with `pytorch-frame` 0.3. Debugging version skew at midnight is not the research question.

```bash
# uv: a fast Python/env manager (the same one you used on the laptop in P0). Installs to ~/.local/bin.
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Your repo (public, so plain HTTPS clone works; VS Code forwards your Git credentials for push).
git clone https://github.com/MaxLuisRodriguez/CS224W-fraud-detection-machine-learning-with-graphs.git ~/wtl
cd ~/wtl

# A Python 3.12 virtualenv. uv downloads that interpreter if Ubuntu doesn't ship it.
uv venv --python 3.12 .venv
source .venv/bin/activate

# 1) PyTorch first, from PyTorch's CUDA 12.8 index, so nothing later pulls a different build.
uv pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128

# 2) RelBench 3.0.2 + its modeling extras: torch_geometric, pytorch-frame[full], sentence-transformers.
#    Never install the PyPI package named "torch-frame": it's an unrelated project.
uv pip install "relbench[example]==3.0.2"

# 3) pyg-lib: PyG's compiled sampler. Temporal NeighborLoader REQUIRES it (no pyg-lib = no time-aware sampling).
#    Its wheel must match torch 2.9 + cu128, so it comes from PyG's own wheel index, not PyPI.
uv pip install pyg-lib --find-links https://data.pyg.org/whl/torch-2.9.0+cu128.html
```

Smoke test: run the check you wrote in P0 (it came down with `git clone`):

```bash
python scripts/check_env.py
# The instance role (Level 0.6) should let you list your bucket with no keys configured.
aws s3 ls s3://wtl-<ACCOUNT>-usw2/ && echo "S3 OK"
```

**PASS:** `python build: linux-x86_64`, `torch 2.9.1+cu128`, `CUDA available: True`, `GPU: NVIDIA L4`, `pyg-lib loaded: True`, `relbench 3.0.2`, and `S3 OK`.
**STOP:**

- `pyg-lib loaded: False`: paste the step-3 output.
- `aws: command not found`: run `uv pip install awscli`, then retry.

### 2.1 Install your package

`pyproject.toml` and `src/wtl/` already exist from P0–P7, so one editable install makes `import wtl` work here too:

```bash
uv pip install matplotlib pytest   # figures and tests, as on the laptop (P0 step 6)
uv pip install -e .                # editable: code edits take effect without reinstalling
mkdir -p outputs                   # training results land here (Git-ignored; synced to S3)
python -m pytest tests             # P3 and P7's tests: 3 should pass here too
```

---

## Level 3 — The same graph code, on the GPU (~20 min, plus a long first build)

You already wrote the graph and model code on the laptop: `text.py` and `data.py` in P5, `model.py` and `reveal.py` in P6, `train_rdl.py` and `summarize_runs.py` in P7. They came down with `git clone`, so there's nothing new to type here. This level checks that the instance builds the same graph as the laptop, then builds the two big graphs once so their tensors are cached.

### 3.1 Same graph as the laptop

```bash
python scripts/inspect_graph.py --dataset rel-f1 --task driver-top3
```

**PASS:** the same output as P5 step 5.5: `races` with 820 nodes, 26 edge types, about 630 sampled race rows of which about 178 are distinct, and `model parameters: 5,074,945` (P6's model, with the `[MASK]` vectors).
**STOP:** any traceback. Paste it to me.

### 3.2 Build and cache the two primary graphs

The first build turns every table into tensors and embeds every text column with GloVe, then caches the result in `~/.cache/wtl/<dataset>/`. On `rel-avito` that includes ~6M ad titles, so it takes a while. Run both builds in `tmux` so they survive closing VS Code, and watch memory in a second terminal with `free -h`:

```bash
tmux new -s build                # a terminal session that lives on the server
source ~/wtl/.venv/bin/activate && cd ~/wtl
python -u scripts/inspect_graph.py --dataset rel-event --task user-ignore 2>&1 | tee outputs/build_rel-event.log
python -u scripts/inspect_graph.py --dataset rel-avito --task user-clicks 2>&1 | tee outputs/build_rel-avito.log
# Detach with Ctrl+b then d. Reattach later with: tmux attach -t build
```

`python -u` turns off output buffering, so the log fills as the run goes instead of all at the end.

**PASS:** each log ends with a mini-batch table and a parameter count. Look at the `rel-event` batch: how many sampled `events` rows are duplicates? That's P5's disjoint-sampling point on the dense dataset, and P4's sharing statistic seen from inside the model.
**STOP:** an out-of-memory kill (the log just stops, and `dmesg | tail` mentions `oom`). Stop the instance, resize it to `g6.4xlarge` (64 GiB) for this step only, and rerun.

> Levels 3.2 and 4.2 haven't been run yet: the laptop can't build these two graphs. Expect to debug once.

---

## Level 4 — Full-information baselines (~30 min, plus overnight)

`scripts/train_rdl.py --train_reveal full` is the "fetch everything" model, and it also evaluates that model at every revelation level *p*. Use **20 epochs**: in P7, 10-epoch runs sometimes never left the early loss plateau.

### 4.1 `rel-f1` on the GPU: compare with the laptop

```bash
python scripts/train_rdl.py --dataset rel-f1 --task driver-top3 --train_reveal full --hide drop --epochs 20 --seed 0
aws s3 sync outputs s3://wtl-<ACCOUNT>-usw2/outputs     # results (JSON) to S3
```

**PASS:** p = 1.0 validation AUROC inside the seed spread you measured on the laptop (P7's `summarize_runs.py` table). GPU and CPU runs won't match to the digit: the sampler isn't deterministic even on one machine.

### 4.2 The primary datasets, overnight

```bash
tmux new -s baselines
source ~/wtl/.venv/bin/activate && cd ~/wtl
for seed in 0 1 2; do
  python -u scripts/train_rdl.py --dataset rel-event --task user-ignore --train_reveal full --hide drop --epochs 20 --seed $seed 2>&1 | tee outputs/rel-event_full_seed$seed.log
  python -u scripts/train_rdl.py --dataset rel-avito --task user-clicks --train_reveal full --hide drop --epochs 20 --seed $seed 2>&1 | tee outputs/rel-avito_full_seed$seed.log
done
aws s3 sync outputs s3://wtl-<ACCOUNT>-usw2/outputs
```

The idle alarm stops the instance about an hour after the last job ends.

**PASS:** the next morning,

```bash
python scripts/summarize_runs.py --dataset rel-event --task user-ignore
python scripts/summarize_runs.py --dataset rel-avito --task user-clicks
```

prints three seeds per dataset. Compare the p = 1 column with the leaderboard's GNN numbers (test AUROC 0.804 and 0.666). Validation is a different day, so expect the same ballpark, not a match. The p < 1 columns are your first random-revelation curves on the primary datasets: the full-trained ("mask-naive") side of H2, at scale.

**STOP:** p = 1 AUROC at or below the leaderboard's LightGBM number (0.773 / 0.548). Paste the training log, and check whether the loss stayed near its epoch-2 value (P7's plateau).

**B1 done.** You have full-information numbers on all three datasets. Next, B2: the lookup environment (`list` / `get` actions keyed on `n_id`, one shared cache, budgets).
