#!/bin/bash
# GCE startup script for VMs created with `create_vm.sh ... --auto`.
# Google runs it as root at every boot.
#
# Before anything else it installs /usr/local/bin/afd-cleanup (upload results,
# profile, and logs, then delete this VM) and a systemd timer that runs it at a
# deadline before --max-run-duration. So even if a download fails, a step
# hangs, or the VM reboots, the chips do not sit idle until the hard cap.
export PATH=/snap/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export HOME=/root
md() { curl -sf --max-time 10 -H 'Metadata-Flavor: Google' \
  "http://metadata.google.internal/computeMetadata/v1/instance/$1"; }

NAME=$(md name)
ZONE=$(md zone); ZONE=${ZONE##*/}
PROJECT_ID=$(curl -sf --max-time 10 -H 'Metadata-Flavor: Google' \
  http://metadata.google.internal/computeMetadata/v1/project/project-id)
RUN_URI=$(md attributes/afd-run-uri)
WORKDIR=$(md attributes/afd-workdir || echo /opt/afd)
DEADLINE_MIN=$(md attributes/afd-deadline-min || echo 200)
SELF_DELETE=$(md attributes/afd-self-delete || echo true)
mkdir -p "$WORKDIR/logs"

cat > /usr/local/bin/afd-cleanup <<EOF
#!/bin/bash
# Upload results, profile, and logs to $RUN_URI, then delete $NAME.
# --force: delete even if an upload failed (used by the deadline timer).
export PATH=$PATH HOME=/root
force=\${1:-}
echo "\$(date '+%F %T %Z') cleanup (\${force:-normal})" >> $WORKDIR/logs/status.txt
docker rm -f afd-vllm >/dev/null 2>&1 || true
cp /var/log/afd-autorun.log $WORKDIR/logs/autorun.log 2>/dev/null || true
ok=true
for d in results profile logs; do
  [ -d $WORKDIR/\$d ] || continue
  for try in 1 2 3; do
    timeout 900 gcloud storage rsync -r $WORKDIR/\$d/ $RUN_URI/\$d/ --quiet && break
    [ \$try = 3 ] && { ok=false; echo "upload of \$d failed" >> $WORKDIR/logs/status.txt; }
    sleep 20
  done
done
timeout 120 gcloud storage cp $WORKDIR/logs/status.txt $RUN_URI/status.txt --quiet || true
if [ "$SELF_DELETE" != true ]; then
  echo "afd-self-delete=false: VM left running" >> $WORKDIR/logs/status.txt
elif [ "\$ok" = true ] || [ "\$force" = --force ]; then
  timeout 600 gcloud compute instances delete $NAME --zone=$ZONE --quiet && exit 0
  # Fallback without gcloud: Compute REST API with the VM's own token.
  token=\$(curl -sf --max-time 10 -H 'Metadata-Flavor: Google' \\
    http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token \\
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')
  curl -sf --max-time 60 -X DELETE -H "Authorization: Bearer \$token" \\
    https://compute.googleapis.com/compute/v1/projects/$PROJECT_ID/zones/$ZONE/instances/$NAME
else
  echo "upload failed: VM kept for rescue until the deadline timer" >> $WORKDIR/logs/status.txt
  timeout 120 gcloud storage cp $WORKDIR/logs/status.txt $RUN_URI/status.txt --quiet || true
fi
EOF
chmod +x /usr/local/bin/afd-cleanup

# Rebooted after a run started: save what exists and delete.
if [ -e /var/lib/afd-autorun-started ]; then
  echo "$(date '+%F %T %Z') rebooted mid-run" >> "$WORKDIR/logs/status.txt"
  exec /usr/local/bin/afd-cleanup --force
fi
touch /var/lib/afd-autorun-started

systemd-run --unit=afd-deadline --on-active="${DEADLINE_MIN}m" \
  /usr/local/bin/afd-cleanup --force

DIR=/root/afd-scripts
mkdir -p "$DIR"
for i in 1 2 3 4 5; do
  timeout 300 gcloud storage cp "$RUN_URI/scripts/*" "$DIR/" && break
  if [ "$i" = 5 ]; then
    echo "$(date '+%F %T %Z') FAILED to download scripts" >> "$WORKDIR/logs/status.txt"
    exec /usr/local/bin/afd-cleanup --force
  fi
  sleep 15
done
exec bash "$DIR/autorun.sh" >> /var/log/afd-autorun.log 2>&1
