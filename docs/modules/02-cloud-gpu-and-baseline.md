# Module 2 · Cloud GPU, data, and the full-information baseline

**Prerequisite:** [Module 1](01-orientation-and-setup.md) (repo skeleton, local environment, `scripts/check_env.py`, first commit pushed).

**Goal:** a GPU instance on AWS running the same code as your laptop, a PyG graph built from RelBench, and the "fetch everything" baseline: the upper bound every lookup budget is measured against.

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

The instance clones your repo from GitHub, so make sure Module 1's commit is pushed. Commit anything new from Level 0 first (`infra/aws/*.json`; the `*.local.json` file stays ignored):

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
# uv: a fast Python/env manager (the same tool SCOUT uses). Installs to ~/.local/bin.
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

Smoke test: run the check you wrote in Module 1 (it came down with `git clone`):

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

`pyproject.toml` and `src/wtl/__init__.py` already exist from Module 1, so one editable install makes `import wtl` work here too:

```bash
uv pip install -e .     # editable: code edits take effect without reinstalling
mkdir -p outputs        # training results land here (Git-ignored; synced to S3)
```

---

## Level 3 — From database to graph (~30 min)

### 3.1 `src/wtl/text.py`: text columns to vectors

RelBench tables contain text (names, ad titles). `pytorch-frame` needs a function that turns a list of strings into a tensor. RelBench's examples use 300-d averaged GloVe word vectors: fast, no GPU needed, ~480 MB downloaded once.

```python
"""Text embedder used when RelBench text columns are materialized into tensors."""
from typing import List, Optional

import torch
from sentence_transformers import SentenceTransformer
from torch import Tensor


class GloveTextEmbedding:
    """Callable that maps a batch of strings to 300-d averaged GloVe vectors."""

    def __init__(self, device: Optional[str] = None):
        # Downloads the model from Hugging Face on first use, then reuses the local cache.
        self.model = SentenceTransformer(
            "sentence-transformers/average_word_embeddings_glove.6B.300d", device=device
        )

    def __call__(self, sentences: List[str]) -> Tensor:
        # Text columns can hold None/NaN; the tokenizer needs real strings, so map missing values to "".
        cleaned = ["" if s is None or (isinstance(s, float) and s != s) else str(s) for s in sentences]
        return self.model.encode(cleaned, convert_to_tensor=True)
```

### 3.2 `src/wtl/data.py`: load, type, graph, and batch

```python
"""Load a RelBench database and task, and turn it into a PyG heterogeneous temporal graph.

Graph layout (RelBench "relational deep learning"):
  * one node type per table, one node per row, row features in data[table].tf (a TensorFrame)
  * one edge type per foreign key, both directions: (child, "f2p_<fkey>", parent) and its "rev_" reverse
  * data[table].time holds each row's timestamp (Unix seconds) wherever the table has a time column
"""
import json
from pathlib import Path

from relbench import load_dataset
from relbench.modeling.graph import get_node_train_table_input, make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.loader import NeighborLoader

from wtl.text import GloveTextEmbedding

# Our own cache for column types and materialized tensors (RelBench's raw data lives in ~/.cache/huggingface).
CACHE_ROOT = Path.home() / ".cache" / "wtl"


def load(dataset_name: str, task_name: str):
    """Download (first time only) and return the dataset, the task, and the database."""
    dataset = load_dataset(dataset_name)   # RelBench 3.x: fetched from the Hugging Face Hub
    task = dataset.load_task(task_name)    # task tables: (entity id, timestamp, label) rows per split
    db = dataset.get_db()                  # all tables, cut at the test timestamp (no future rows)
    return dataset, task, db


def column_stypes(db, dataset_name: str) -> dict:
    """Semantic type per column (numerical / categorical / text / timestamp ...), cached as JSON."""
    path = CACHE_ROOT / dataset_name / "stypes.json"
    if path.exists():
        raw = json.loads(path.read_text())
        # JSON stores each type's value (e.g. "numerical"); rebuild the torch_frame enum from it.
        return {table: {col: stype(s) for col, s in cols.items()} for table, cols in raw.items()}
    proposal = get_stype_proposal(db)      # RelBench's heuristic guess from each column's dtype and content
    path.parent.mkdir(parents=True, exist_ok=True)
    # stype is a plain Enum: save .value ("numerical"), not str() ("stype.numerical"), or reloading breaks.
    serializable = {table: {col: st.value for col, st in cols.items()} for table, cols in proposal.items()}
    path.write_text(json.dumps(serializable, indent=2))
    return proposal


def build_graph(db, dataset_name: str, task, col_to_stype: dict):
    """Materialize every table into tensors and connect rows by their primary/foreign keys."""
    data, col_stats = make_pkey_fkey_graph(
        db,
        col_to_stype_dict=col_to_stype,
        # Text columns go through GloVe on CPU, 256 strings at a time.
        text_embedder_cfg=TextEmbedderConfig(text_embedder=GloveTextEmbedding(device="cpu"), batch_size=256),
        # Materialized tensors are cached per dataset; the slow step only runs once.
        cache_dir=str(CACHE_ROOT / dataset_name / "materialized"),
        # Columns that would leak this task's label are dropped (applied after the cache, so one cache serves all tasks).
        remove_columns=task.hidden_columns(),
    )
    return data, col_stats   # col_stats: per-column statistics the encoders use for normalization


def make_loader(data, task, split: str, num_neighbors=(128, 64), batch_size: int = 512, shuffle: bool = False):
    """Mini-batches of target entities, each with its sampled 2-hop temporal neighborhood."""
    table = task.get_table(split)          # keep this: get_table is not cached in 3.x
    table_input = get_node_train_table_input(table=table, task=task)
    loader = NeighborLoader(
        data,
        num_neighbors=list(num_neighbors), # up to 128 neighbors per edge type at hop 1, 64 at hop 2
        time_attr="time",                  # sample only rows whose time <= the seed's prediction time
        input_nodes=table_input.nodes,     # (entity table name, row indices of the targets)
        input_time=table_input.time,       # each target's prediction timestamp
        transform=table_input.transform,   # attaches labels as batch[entity_table].y
        batch_size=batch_size,
        temporal_strategy="uniform",       # among eligible past neighbors, sample uniformly
        shuffle=shuffle,
        num_workers=0,
    )
    return loader, table
```

### 3.3 `scripts/explore.py`: look before you model

```python
"""Level 3 check: print the database schema, the task, and the resulting graph."""
import argparse

from wtl.data import build_graph, column_stypes, load

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="rel-f1")
parser.add_argument("--task", default="driver-top3")
args = parser.parse_args()

dataset, task, db = load(args.dataset, args.task)
print("tasks in this dataset:", dataset.get_task_names())

# The schema: each table's size, primary key, foreign keys (= graph edges), and time column.
for name, table in db.table_dict.items():
    print(f"{name:24s} rows={len(table.df):>10,}  pkey={table.pkey_col}  "
          f"fkeys={table.fkey_col_to_pkey_table}  time={table.time_col}")

# The task: which entity we score, what label, and over what horizon.
print(f"\nentity={task.entity_table}.{task.entity_col}  target={task.target_col}  horizon={task.timedelta}")
for split in ["train", "val"]:  # never peek at test
    df = task.get_table(split).df
    print(f"{split:5s} rows={len(df):,}  positive rate={df[task.target_col].mean():.3f}  "
          f"prediction times={df[task.time_col].nunique()}")

# The graph PyG will train on: node types with counts, and edge types (foreign keys, both directions).
data, _ = build_graph(db, args.dataset, task, column_stypes(db, args.dataset))
print("\n", data)
```

```bash
python scripts/explore.py --dataset rel-f1 --task driver-top3
```

**PASS:**

- 9 tables (races, circuits, drivers, constructors, results, standings, …) with foreign keys.
- `entity=drivers.driverId`.
- Train and val positive rates.
- A `HeteroData(...)` printout with `f2p_…` / `rev_f2p_…` edge types.

**STOP:** any traceback. Paste it to me.

---

## Level 4 — Full-information baseline (~30 min)

This is the **"fetch everything" upper bound** every budgeted policy is measured against. If it doesn't roughly match RelBench's number, nothing built later can be trusted.

### 4.1 `src/wtl/model.py`

```python
"""Relational deep learning model: per-table encoders -> time encoding -> hetero GraphSAGE -> MLP head."""
import torch
from relbench.modeling.nn import HeteroEncoder, HeteroGraphSAGE, HeteroTemporalEncoder
from torch_geometric.nn import MLP


class RDLModel(torch.nn.Module):
    """Scores target entities from their sampled relational neighborhood (RelBench's baseline design)."""

    def __init__(self, data, col_stats, channels: int = 128, num_layers: int = 2, aggr: str = "sum"):
        super().__init__()
        # One tabular encoder (torch_frame ResNet) per table: row columns -> a `channels`-d vector.
        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={nt: data[nt].tf.col_names_dict for nt in data.node_types},
            node_to_col_stats=col_stats,
        )
        # Encodes "how long before the prediction time" each row happened (recency matters).
        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[nt for nt in data.node_types if "time" in data[nt]], channels=channels,
        )
        # Message passing over every foreign-key edge type: one SAGEConv per edge type, summed per layer.
        self.gnn = HeteroGraphSAGE(
            node_types=data.node_types, edge_types=data.edge_types,
            channels=channels, aggr=aggr, num_layers=num_layers,
        )
        # Linear readout to one logit per target entity.
        self.head = MLP(channels, out_channels=1, norm="batch_norm", num_layers=1)

    def forward(self, batch, entity_table: str) -> torch.Tensor:
        seed_time = batch[entity_table].seed_time            # prediction time of each target in the batch
        x_dict = self.encoder(batch.tf_dict)                  # {table: [num_rows_in_batch, channels]}
        rel_time = self.temporal_encoder(seed_time, batch.time_dict, batch.batch_dict)
        for node_type, t in rel_time.items():
            x_dict[node_type] = x_dict[node_type] + t        # add recency information to each row
        x_dict = self.gnn(x_dict, batch.edge_index_dict)      # 2 rounds of message passing
        # The first len(seed_time) rows of the entity table are the targets themselves.
        return self.head(x_dict[entity_table][: seed_time.size(0)]).view(-1)
```

### 4.2 `scripts/train_full.py`

```python
"""Level 4: train the full-information RDL baseline and save validation results."""
import argparse
import json
import time
from pathlib import Path

import torch
from relbench.base import TaskType
from torch.nn import BCEWithLogitsLoss
from torch_geometric.seed import seed_everything
from tqdm import tqdm

from wtl.data import build_graph, column_stypes, load, make_loader
from wtl.model import RDLModel


@torch.no_grad()
def predict(model, loader, entity_table: str, device) -> "numpy.ndarray":
    """Probabilities for every row of the loader's table, in table order (the loader is unshuffled)."""
    model.eval()
    probs = [torch.sigmoid(model(batch.to(device), entity_table)).cpu() for batch in loader]
    return torch.cat(probs).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="rel-f1")
    parser.add_argument("--task", default="driver-top3")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--max_steps", type=int, default=2000)   # cap per epoch for big datasets
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)                                   # Python, NumPy, and torch RNGs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset, task, db = load(args.dataset, args.task)
    assert task.task_type == TaskType.BINARY_CLASSIFICATION, "this script handles binary tasks"
    data, col_stats = build_graph(db, args.dataset, task, column_stypes(db, args.dataset))
    train_loader, _ = make_loader(data, task, "train", batch_size=args.batch_size, shuffle=True)
    val_loader, val_table = make_loader(data, task, "val", batch_size=args.batch_size)

    model = RDLModel(data, col_stats).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = BCEWithLogitsLoss()                                # binary cross-entropy on raw logits

    best = {"val_roc_auc": -1.0}
    out_dir = Path("outputs") / args.dataset / args.task
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, count = 0.0, 0
        for step, batch in enumerate(tqdm(train_loader, desc=f"epoch {epoch}", leave=False)):
            if step >= args.max_steps:
                break
            batch = batch.to(device)
            logits = model(batch, task.entity_table)
            loss = loss_fn(logits, batch[task.entity_table].y.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * logits.numel()
            count += logits.numel()

        # RelBench computes the official metrics (roc_auc etc.) against the val labels.
        metrics = task.evaluate(predict(model, val_loader, task.entity_table, device), val_table)
        print(f"epoch {epoch:02d}  train_loss={loss_sum / count:.4f}  val={metrics}")

        if metrics["roc_auc"] > best["val_roc_auc"]:             # keep the best epoch, not the last
            best = {"val_roc_auc": metrics["roc_auc"], "epoch": epoch, **{f"val_{k}": v for k, v in metrics.items()}}
            torch.save(model.state_dict(), out_dir / f"full_seed{args.seed}.pt")

    best.update(vars(args), minutes=(time.time() - start) / 60, device=str(device))
    (out_dir / f"full_seed{args.seed}.json").write_text(json.dumps(best, indent=2, default=str))
    print("BEST:", json.dumps(best, default=str))


if __name__ == "__main__":
    main()
```

```bash
python scripts/train_full.py --dataset rel-f1 --task driver-top3

# Back up results (JSON + checkpoint) to S3 under a per-dataset prefix.
aws s3 sync outputs s3://wtl-<ACCOUNT>-usw2/outputs
```

**PASS:**

- Ten epochs run, and train loss falls.
- Best validation AUROC is clearly above 0.5. For reference, the leaderboard *test* AUROC is 0.68; val is a different period with only ~600 rows, so expect noise.
- `outputs/rel-f1/driver-top3/full_seed42.json` exists and is in S3.

### 4.3 Kick off the primary dataset overnight

`rel-avito` is ~450 MB plus GloVe embeddings for ~6M ad titles, so the first graph build takes a while. Run it inside `tmux` so it survives closing VS Code:

```bash
tmux new -s avito            # a terminal session that lives on the server
source ~/wtl/.venv/bin/activate && cd ~/wtl
python scripts/train_full.py --dataset rel-avito --task user-clicks --epochs 10 2>&1 | tee outputs/avito_full.log
# Detach with Ctrl+b then d. Reattach later with: tmux attach -t avito
```

Watch memory in a second terminal with `free -h`. If it dies with an out-of-memory error, tell me. We'll resize to `g6.4xlarge` (64 GiB) for this dataset only. The idle alarm stops the instance about an hour after the job ends.

**Module 2 done.** You have the full-information numbers for the proposal. Next, Module 3: revelation masks, the `[MASK]` embeddings, and mask-aware training.
