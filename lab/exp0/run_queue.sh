#!/usr/bin/env bash
# Run sweep stages sequentially (each configuration still takes the GPU lock on its own).
# Usage: run_queue.sh "s1:probe" "s2:nommproj" "s2:quality" ...   logs -> ~/srv/brain-lab/exp0/logs/run_<script>_<stage>.log
cd "$(dirname "$0")"
for item in "$@"; do
  script=${item%%:*}; stage=${item#*:}
  echo "$(date -Is) start $item" >> ~/srv/brain-lab/exp0/logs/queue.log
  ./run_$script.sh $stage > ~/srv/brain-lab/exp0/logs/run_${script}_${stage}.log 2>&1
  echo "$(date -Is) done $item" >> ~/srv/brain-lab/exp0/logs/queue.log
done
