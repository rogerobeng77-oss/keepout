#!/usr/bin/env bash
# Give the Keepout App Runner service permission to write evidence frames to S3.
#
#   products/keepout/infra/instance-role.sh [--region eu-west-2]
#
# The shared infra/apprunner.sh creates the ECR *access* role, which is what lets
# App Runner pull the image. It does not create an *instance* role, which is what
# the running container assumes. Without one, boto3 inside the container has no
# credentials at all, every PutObject fails, and the analyzer spends minutes in
# retries. This script creates the instance role, attaches a policy scoped to the
# product's own prefix in the shared artefact bucket, and updates the service.
#
# Scoped deliberately narrowly: PutObject only, only under keepout/evidence/*.
# The service never needs to read, list or delete.

set -euo pipefail

REGION="${AWS_REGION:-eu-west-2}"
BUCKET="${KEEPOUT_EVIDENCE_BUCKET:-opencv26-artifacts-<aws-account-id>}"
PREFIX="${KEEPOUT_EVIDENCE_PREFIX:-keepout/evidence}"
ROLE="opencv26-keepout-instance"
POLICY="opencv26-keepout-evidence-write"
SERVICE="opencv26-keepout"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --region) REGION="$2"; shift 2 ;;
    --bucket) BUCKET="$2"; shift 2 ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 1 ;;
  esac
done

ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
[[ "$ACCOUNT" == "<aws-account-id>" ]] || { echo "wrong account: $ACCOUNT" >&2; exit 1; }
echo "==> account $ACCOUNT, region $REGION, bucket $BUCKET"

TRUST='{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "tasks.apprunner.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}'

if aws iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
  echo "  ok role $ROLE exists"
else
  echo "==> creating role $ROLE"
  aws iam create-role --role-name "$ROLE" \
    --assume-role-policy-document "$TRUST" \
    --description "Keepout App Runner instance role: write evidence frames to S3" \
    --tags Key=Project,Value=opencv26 Key=Product,Value=keepout >/dev/null
fi

POLICY_DOC="{
  \"Version\": \"2012-10-17\",
  \"Statement\": [{
    \"Sid\": \"WriteEvidenceFramesOnly\",
    \"Effect\": \"Allow\",
    \"Action\": [\"s3:PutObject\"],
    \"Resource\": \"arn:aws:s3:::${BUCKET}/${PREFIX}/*\"
  }]
}"

echo "==> inline policy $POLICY on $ROLE"
aws iam put-role-policy --role-name "$ROLE" \
  --policy-name "$POLICY" --policy-document "$POLICY_DOC"

ROLE_ARN="arn:aws:iam::${ACCOUNT}:role/${ROLE}"
echo "  ok $ROLE_ARN"

ARN="$(aws apprunner list-services --region "$REGION" \
  --query "ServiceSummaryList[?ServiceName=='$SERVICE'].ServiceArn | [0]" \
  --output text 2>/dev/null | grep -v '^None$' || true)"

if [[ -z "$ARN" ]]; then
  echo "  !! no App Runner service named $SERVICE in $REGION; deploy it first" >&2
  exit 1
fi

CURRENT="$(aws apprunner describe-service --region "$REGION" --service-arn "$ARN" \
  --query 'Service.InstanceConfiguration.InstanceRoleArn' --output text)"
if [[ "$CURRENT" == "$ROLE_ARN" ]]; then
  echo "  ok service already uses $ROLE"
  exit 0
fi

# IAM is eventually consistent; App Runner rejects a role it cannot yet assume.
echo "==> waiting 12s for the role to propagate"
sleep 12

CPU="$(aws apprunner describe-service --region "$REGION" --service-arn "$ARN" \
  --query 'Service.InstanceConfiguration.Cpu' --output text)"
MEM="$(aws apprunner describe-service --region "$REGION" --service-arn "$ARN" \
  --query 'Service.InstanceConfiguration.Memory' --output text)"

echo "==> attaching the instance role (keeping $CPU / $MEM)"
aws apprunner update-service --region "$REGION" --service-arn "$ARN" \
  --instance-configuration "Cpu=$CPU,Memory=$MEM,InstanceRoleArn=$ROLE_ARN" \
  --query 'Service.Status' --output text

echo "==> waiting for the update to finish"
for _ in $(seq 1 60); do
  STATUS="$(aws apprunner describe-service --region "$REGION" --service-arn "$ARN" \
    --query 'Service.Status' --output text)"
  [[ "$STATUS" == "RUNNING" ]] && break
  sleep 10
done
echo "  ok service status $STATUS"

URL="https://$(aws apprunner describe-service --region "$REGION" --service-arn "$ARN" \
  --query 'Service.ServiceUrl' --output text)"
echo "$URL"
