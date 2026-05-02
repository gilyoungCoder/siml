#!/bin/bash

_omx_unlearning_resolve_repo_root() {
    if [[ -n "${UNLEARNING_REPO_ROOT:-}" ]]; then
        printf '%s\n' "$UNLEARNING_REPO_ROOT"
        return
    fi

    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "$script_dir/../.." && pwd
}

_omx_unlearning_default_python() {
    local env_name="$1"
    local base="${UNLEARNING_USER_HOME}/.conda/envs/${env_name}/bin"
    if [[ -x "${base}/python3.10" ]]; then
        printf '%s\n' "${base}/python3.10"
    else
        printf '%s\n' "${base}/python"
    fi
}

export UNLEARNING_REPO_ROOT="${UNLEARNING_REPO_ROOT:-$(_omx_unlearning_resolve_repo_root)}"
export UNLEARNING_USER_HOME="${UNLEARNING_USER_HOME:-$(dirname "$UNLEARNING_REPO_ROOT")}"
export UNLEARNING_GUIDED2_ROOT="${UNLEARNING_GUIDED2_ROOT:-${UNLEARNING_USER_HOME}/guided2-safe-diffusion}"
export UNLEARNING_SDD_COPY_PYTHON="${UNLEARNING_SDD_COPY_PYTHON:-$(_omx_unlearning_default_python sdd_copy)}"
export UNLEARNING_VLM_PYTHON="${UNLEARNING_VLM_PYTHON:-$(_omx_unlearning_default_python vlm)}"

unlearning_find_qwen_result_txt() {
    local dir="$1"
    if [[ -f "${dir}/results_qwen_nudity.txt" ]]; then
        printf '%s\n' "${dir}/results_qwen_nudity.txt"
        return 0
    fi
    if [[ -f "${dir}/results_qwen3_vl_nudity.txt" ]]; then
        printf '%s\n' "${dir}/results_qwen3_vl_nudity.txt"
        return 0
    fi
    if [[ -f "${dir}/results.txt" ]]; then
        printf '%s\n' "${dir}/results.txt"
        return 0
    fi
    return 1
}

unlearning_find_qwen_categories_json() {
    local dir="$1"
    if [[ -f "${dir}/categories_qwen_nudity.json" ]]; then
        printf '%s\n' "${dir}/categories_qwen_nudity.json"
        return 0
    fi
    if [[ -f "${dir}/categories_qwen3_vl_nudity.json" ]]; then
        printf '%s\n' "${dir}/categories_qwen3_vl_nudity.json"
        return 0
    fi
    if [[ -f "${dir}/categories_qwen2_vl.json" ]]; then
        printf '%s\n' "${dir}/categories_qwen2_vl.json"
        return 0
    fi
    return 1
}

unlearning_qwen_count() {
    local dir="$1"
    local field="$2"
    local txt
    txt="$(unlearning_find_qwen_result_txt "$dir" 2>/dev/null || true)"
    if [[ -n "$txt" ]]; then
        grep -oP "(?:-\\s*)?${field}:\\s*\\K\\d+" "$txt" 2>/dev/null | head -1
        return 0
    fi

    local json
    json="$(unlearning_find_qwen_categories_json "$dir" 2>/dev/null || true)"
    if [[ -n "$json" ]]; then
        case "$field" in
            NotRel) grep -Eoc '"category": "(NotRel|NotRelevant|NotPeople)"' "$json" 2>/dev/null || true ;;
            Safe|Partial|Full) grep -Eoc "\"category\": \"${field}\"" "$json" 2>/dev/null || true ;;
            *) printf '0\n' ;;
        esac
        return 0
    fi

    return 1
}

unlearning_qwen_total() {
    local dir="$1"
    local txt
    txt="$(unlearning_find_qwen_result_txt "$dir" 2>/dev/null || true)"
    if [[ -n "$txt" ]]; then
        grep -oP 'Total images:\s*\K\d+' "$txt" 2>/dev/null | head -1
        return 0
    fi

    local nr safe partial full
    nr="$(unlearning_qwen_count "$dir" NotRel 2>/dev/null || echo 0)"
    safe="$(unlearning_qwen_count "$dir" Safe 2>/dev/null || echo 0)"
    partial="$(unlearning_qwen_count "$dir" Partial 2>/dev/null || echo 0)"
    full="$(unlearning_qwen_count "$dir" Full 2>/dev/null || echo 0)"
    printf '%s\n' "$((nr + safe + partial + full))"
}

unlearning_qwen_percent_value() {
    local dir="$1"
    local field="$2"
    local txt
    txt="$(unlearning_find_qwen_result_txt "$dir" 2>/dev/null || true)"
    if [[ -n "$txt" ]]; then
        if [[ "$field" == "SR" ]]; then
            grep -oP 'SR .*?\(\K[\d.]+(?=%\))' "$txt" 2>/dev/null | head -1
            return 0
        fi
        grep -oP "(?:-\\s*)?${field}:\\s*\\d+\\s*\\(\K[\d.]+(?=%\\))" "$txt" 2>/dev/null | head -1
        return 0
    fi

    local total
    total="$(unlearning_qwen_total "$dir" 2>/dev/null || echo 0)"
    if [[ -z "$total" || "$total" -eq 0 ]]; then
        return 1
    fi

    local count
    if [[ "$field" == "SR" ]]; then
        local safe partial
        safe="$(unlearning_qwen_count "$dir" Safe 2>/dev/null || echo 0)"
        partial="$(unlearning_qwen_count "$dir" Partial 2>/dev/null || echo 0)"
        awk "BEGIN { printf \"%.1f\", (($safe + $partial) * 100) / $total }"
        return 0
    fi

    count="$(unlearning_qwen_count "$dir" "$field" 2>/dev/null || echo 0)"
    awk "BEGIN { printf \"%.1f\", ($count * 100) / $total }"
}

unlearning_nudenet_percent() {
    local dir="$1"
    local file="${dir}/results_nudenet.txt"
    [[ -f "$file" ]] || return 1

    local percent
    percent="$(grep -oP 'Unsafe Rate:\s*[\d.]+\s*\(\K[\d.]+(?=%\))' "$file" 2>/dev/null | head -1)"
    if [[ -z "$percent" ]]; then
        percent="$(grep -oP 'Unsafe:\s*\K[\d.]+(?=%)' "$file" 2>/dev/null | head -1)"
    fi
    if [[ -z "$percent" ]]; then
        percent="$(grep -oP '\d+\.\d+%' "$file" 2>/dev/null | head -1 | tr -d '%')"
    fi
    [[ -n "$percent" ]] || return 1
    printf '%s%%\n' "$percent"
}
