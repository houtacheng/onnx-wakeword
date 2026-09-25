#!/bin/sh
set -eu

OPTIONS_FILE=/data/options.json

option() {
    python -c 'import json, sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], sys.argv[3]))' \
        "${OPTIONS_FILE}" "$1" "$2"
}

threshold="$(option threshold 0.4)"
cooldown="$(option cooldown 1500)"
l1="$(option L1 1)"
l3="$(option L3 1)"
l5="$(option L5 0)"
debug="$(option debug false)"

set -- python /app/wyoming/wyoming_voicute_ha.py \
    --uri tcp://0.0.0.0:10400 \
    --model-info /app/models/model_info.json \
    --mel /app/models/melspectrogram.onnx \
    --threshold "${threshold}" \
    --cooldown "${cooldown}" \
    --L1 "${l1}" \
    --L3 "${l3}" \
    --L5 "${l5}"

if [ "${debug}" = "True" ] || [ "${debug}" = "true" ]; then
    set -- "$@" --debug
fi

exec "$@"
