# syntax=docker/dockerfile:1
#
# Docker sidecar image for the dev-teraswitch ARC runner pods: rootless dockerd
# (uid 1001) with a systemd user manager beside it, so dockerd uses the systemd
# cgroup driver. Job containers then get their own cgroups with memory, CPU and
# pids limits, `docker pause` works, and kind accepts the daemon. Rootless
# dockerd without systemd runs with CgroupDriver=none, which kind rejects.
#
# The Docker binaries and entrypoint are the official docker:dind-rootless
# image's, on an Ubuntu base that has systemd. Rebuilt daily by
# .github/workflows/build-arc-runner-image.yml.

ARG DOCKER_VERSION=29.9.1
FROM docker:${DOCKER_VERSION}-dind-rootless AS docker

FROM ubuntu:24.04
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ca-certificates dbus-user-session fuse-overlayfs iproute2 iptables pigz \
      slirp4netns systemd uidmap xz-utils \
 && rm -rf /var/lib/apt/lists/*

COPY --from=docker /usr/local/bin/ /usr/local/bin/

# dockerd-entrypoint.sh puts these first on PATH when the node already has
# legacy iptables rules.
RUN mkdir /usr/local/sbin/.iptables-legacy \
 && for t in iptables iptables-restore iptables-save ip6tables ip6tables-restore ip6tables-save; do \
      ln -s "/usr/sbin/${t/tables/tables-legacy}" "/usr/local/sbin/.iptables-legacy/$t"; \
    done

# kind bind-mounts /lib/modules into its nodes; nothing here needs module files.
RUN groupadd --gid 1001 runner \
 && useradd --uid 1001 --gid 1001 --home-dir /home/runner --create-home --shell /bin/sh runner \
 && echo 'runner:100000:65536' > /etc/subuid \
 && echo 'runner:100000:65536' > /etc/subgid \
 && mkdir -p /lib/modules

# Job containers run without a session keyring of their own. By default runc
# creates one per container, and the kernel charges it to the uid that owns the
# container's root, which is uid 1001 for every runner pod on a node. That uid
# gets kernel.keys.maxkeys keys node-wide (200 by default), so once about 200
# job containers are up at once across a node's pods, every new one fails to
# start with "unable to join session keyring: unable to create session key:
# disk quota exceeded". With NoNewKeyring a container stays in the session
# keyring runc was started in and charges no key. Docker's default seccomp
# profile already denies keyctl, add_key and request_key, so a container could
# not use its own keyring anyway. dind-entrypoint passes this file with
# --config-file, because the default rootless path (~/.config/docker) is under
# the dind-home volume and would hide a file baked into the image.
COPY <<'EOF' /etc/docker/daemon.json
{
  "default-runtime": "runc-no-keyring",
  "runtimes": {
    "runc-no-keyring": {
      "runtimeType": "io.containerd.runc.v2",
      "options": { "NoNewKeyring": true, "SystemdCgroup": true }
    }
  }
}
EOF

COPY --chmod=0755 <<'EOF' /usr/local/bin/dind-entrypoint
#!/bin/sh
# Starts as root only to delegate this container's own cgroup to the runner user
# and set up its namespaces; no root process remains. The container is
# privileged, so it shares the node's cgroup namespace and /sys/fs/cgroup is the
# node's root. This touches only the container's own scope under kubepods.slice,
# never anything above it.
set -eu

if [ "${1:-}" != ns ]; then
  self=$(cut -d: -f3 /proc/self/cgroup)
  case "$self" in
    /kubepods.slice/*) ;;
    *) echo "dind: unexpected cgroup '$self', not delegating" >&2; exit 1 ;;
  esac
  cg="/sys/fs/cgroup$self"
  # The kernel's delegation set: the directory and these three files. The
  # scope's own limits (memory.max, cpu.max) stay root-owned.
  chown 1001:1001 "$cg" "$cg/cgroup.procs" "$cg/cgroup.subtree_control" "$cg/cgroup.threads"
  # kubelet sets memory.oom.group=1, which on any OOM in the scope kills
  # dockerd and every job container; 0 kills only the largest process.
  echo 0 > "$cg/memory.oom.group"
  # A cgroup namespace rooted at the scope gives systemd and dockerd the
  # /user.slice/user-1001.slice/user@1001.service layout they expect.
  exec unshare --cgroup --mount "$0" ns
fi

umount /sys/fs/cgroup
mount -t cgroup2 cgroup2 /sys/fs/cgroup
cd /sys/fs/cgroup
ctl=
for c in cpuset cpu io memory pids; do
  if grep -qw "$c" cgroup.controllers; then ctl="$ctl +$c"; fi
done
# A cgroup that has processes cannot enable controllers for its children, so
# every process here moves to a leaf first. kubelet's startup probe execs into
# this cgroup too, and one can land between the move and the write, hence the
# retries. Once controllers are enabled, later execs join the leaf instead.
mkdir init
for _ in 1 2 3 4 5 6 7 8 9 10; do
  while read -r pid; do echo "$pid" > init/cgroup.procs 2>/dev/null || :; done < cgroup.procs
  if echo "$ctl" > cgroup.subtree_control 2>/dev/null; then break; fi
  sleep 0.1
done
echo "$ctl" > cgroup.subtree_control
m=user.slice/user-1001.slice/user@1001.service
mkdir -p "$m"
for d in user.slice user.slice/user-1001.slice; do
  echo "$ctl" > "$d/cgroup.subtree_control"
done
chown 1001:1001 "$m" "$m/cgroup.procs" "$m/cgroup.subtree_control" "$m/cgroup.threads"

chown 1001:1001 /home/runner /run/user/1001
chmod 700 /run/user/1001
# systemd starts a user manager only when sd_booted(3) is true.
mkdir -p /run/systemd/system

env="HOME=/home/runner USER=runner LOGNAME=runner XDG_RUNTIME_DIR=/run/user/1001"
env="$env PATH=/usr/local/bin:/usr/local/sbin:/usr/bin:/usr/sbin:/bin:/sbin"
sh -c "echo \$\$ > $m/cgroup.procs && exec setpriv --reuid=1001 --regid=1001 --init-groups env -i $env /usr/lib/systemd/systemd --user" &
# No rootlesskit --pidns: runc passes container PIDs to systemd, so both must
# share a PID namespace. Without its own PID namespace, docker-init reaps
# dockerd's orphans only as a subreaper.
exec setpriv --reuid=1001 --regid=1001 --init-groups \
  env -i $env SYSTEMD_PID=$! TINI_SUBREAPER=1 \
  DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1001/bus sh -c '
    until busctl --user status >/dev/null 2>&1; do
      kill -0 "$SYSTEMD_PID" 2>/dev/null || { echo "dind: systemd --user exited" >&2; exit 1; }
      sleep 0.2
    done
    exec dockerd-entrypoint.sh dockerd --host=unix:///run/user/1001/docker.sock \
      --config-file /etc/docker/daemon.json \
      --exec-opt native.cgroupdriver=systemd'
EOF

ENTRYPOINT ["/usr/local/bin/dind-entrypoint"]
