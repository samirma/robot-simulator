#!/usr/bin/env bash
# Fetch the official robot meshes that robots_specs/ references but git does not track.
#
#   ./fetch_robot_assets.sh                    every robot in robots_specs/robots.yml
#   ./fetch_robot_assets.sh so101 ainex        only these robots (a composite brings its base)
#
# Each mesh comes from its robot's upstream at the revision pinned in robots.yml: a
# sparse, blobless checkout for GitHub sources, and HTTP Range reads of the one nested
# zip it needs for Yahboom's Google Drive archive. Sources are cached under
# ${XDG_CACHE_HOME:-~/.cache}/robot-simulator. Files already matching
# robots_specs/meshes.sha256 are left alone, and every file is verified against it at
# the end, so re-running is a no-op.
set -euo pipefail

# Resolved without cd, like the engines' run.sh.
_self="${BASH_SOURCE[0]}"
case "$_self" in /*) ;; *) _self="$PWD/$_self" ;; esac
ROOT="$(dirname "$_self")"
ROOT="$(realpath "$ROOT" 2>/dev/null || echo "${ROOT%/.}")"
SPECS="$ROOT/robots_specs"
MANIFEST="$SPECS/meshes.sha256"
CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/robot-simulator"

die() { echo "error: $*" >&2; exit 1; }

command -v git >/dev/null || die "git not found"
command -v python3 >/dev/null || die "python3 not found"
[ -f "$MANIFEST" ] || die "missing $MANIFEST"

# Where each mesh folder comes from:
#   robot | robots.yml key naming the source | local folder (under robots_specs/) |
#   upstream folder (in the repository, or inside the Drive archive's nested zip)
SOURCES=(
  "so101|source|so101/assets|Simulation/SO101/assets"
  "myagv|source|myagv/urdf|myagv_urdf/urdf"
  "ainex|source|ainex/meshes|src/ainex_simulations/ainex_description/meshes"
  "myagv_mycobot280|source.arm_description|myagv_mycobot280/urdf/mycobot_280_pi|mycobot_description/urdf/mycobot_280_pi"
  "myagv_mycobot280|source.arm_mjcf|myagv_mycobot280/meshes_mujoco|meshes_mujoco"
  "rosmaster_x3_plus|source.code_download|rosmaster_x3_plus/meshes|yahboomcar_ws/src/yahboomcar_description/meshes"
)

# robots.yml flattened to "id<TAB>dotted.key<TAB>value" lines. A few lines of stdlib
# rather than PyYAML, which a fresh machine may not have; robots.yml is plain
# block-style YAML.
YAML_PY='
import re, sys
rid, stack = None, []
for raw in open(sys.argv[1]):
    line = re.sub(r"\s+#.*$", "", raw.rstrip("\n"))
    m = re.match(r"^(\s*)(- )?([A-Za-z_]\w*):\s*(.*)$", line)
    if not m or line.lstrip().startswith("#"):
        continue
    ind = len(m.group(1)) + (2 if m.group(2) else 0)
    key, val = m.group(3), m.group(4).strip()
    if m.group(2) and key == "id":
        rid, stack = val, []
        print(rid, "id", val, sep="\t")
        continue
    if rid is None:
        continue
    stack = [(i, k) for i, k in stack if i < ind]
    if val:
        print(rid, ".".join([k for _, k in stack] + [key]), val, sep="\t")
    else:
        stack.append((ind, key))
'
SPEC="$(python3 -c "$YAML_PY" "$SPECS/robots.yml")" || die "cannot read robots.yml"

spec() {  # spec ROBOT KEY -> value, or empty
  awk -F '\t' -v r="$1" -v k="$2" '$1 == r && $2 == k { print $3; exit }' <<<"$SPEC"
}

# Manifest lines under a local folder.
manifest_under() { awk -v p="$1/" 'index($2, p) == 1' "$MANIFEST"; }

# Paths (relative to robots_specs/) of the manifest lines on stdin whose file is missing
# or differs. Hashed in python rather than `sha256sum -c`, whose output macOS's own
# sha256sum does not share with GNU's.
STALE_PY='
import hashlib, os, sys
for line in sys.stdin:
    if not line.strip():
        continue
    want, path = line.rstrip("\n").split(None, 1)
    full = os.path.join(sys.argv[1], path)
    h = hashlib.sha256()
    try:
        with open(full, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
    except OSError:
        print(path)
        continue
    if h.hexdigest() != want:
        print(path)
'
stale() { python3 -c "$STALE_PY" "$SPECS"; }

# ---------------------------------------------------------------- sources

fetch_git() {  # fetch_git REPO_URL REVISION UPSTREAM_DIR LOCAL_DIR REL...
  local url="$1" rev="$2" up="$3" local_dir="$4"; shift 4
  local repo="${url%.git}"; repo="${repo##*/}"
  local src="$CACHE/src/$repo"
  if [ ! -d "$src/.git" ]; then
    mkdir -p "$src"
    git -C "$src" init -q
    git -C "$src" remote add origin "$url"
    git -C "$src" sparse-checkout init --no-cone
  fi
  local patterns=() rel
  for rel in "$@"; do patterns+=("/$up/$rel"); done
  git -C "$src" sparse-checkout add "${patterns[@]}"
  if ! git -C "$src" cat-file -e "$rev^{commit}" 2>/dev/null; then
    echo ">> fetching $repo @ ${rev:0:12}"
    git -C "$src" fetch -q --depth 1 --filter=blob:none origin "$rev" \
      || die "cannot fetch $url @ $rev"
  fi
  git -C "$src" checkout -q --detach "$rev" || die "cannot check out $repo @ $rev"
  for rel in "$@"; do
    [ -f "$src/$up/$rel" ] || die "$repo @ ${rev:0:12} has no $up/$rel"
    mkdir -p "$(dirname "$SPECS/$local_dir/$rel")"
    cp "$src/$up/$rel" "$SPECS/$local_dir/$rel"
  done
}

fetch_drive() {  # fetch_drive ROBOT KEY UPSTREAM_DIR LOCAL_DIR REL...
  local robot="$1" key="$2" up="$3" local_dir="$4"; shift 4
  local url id bytes member sha
  url="$(spec "$robot" "$key.url")"
  bytes="$(spec "$robot" "$key.bytes")"
  member="$(spec "$robot" "$key.workspace_zip")"
  sha="$(spec "$robot" "$key.workspace_zip_sha256")"
  id="$(sed -n 's|.*/file/d/\([^/?]*\).*|\1|p' <<<"$url")"
  [ -n "$id" ] && [ -n "$bytes" ] && [ -n "$member" ] && [ -n "$sha" ] \
    || die "$robot: $key in robots.yml needs url, bytes, workspace_zip and workspace_zip_sha256"
  python3 "$SPECS/tools/fetch_drive_zip_members.py" \
    --drive-id "$id" --outer-bytes "$bytes" \
    --member "$member" --member-sha256 "$sha" --cache "$CACHE/drive/$id/$member" \
    --prefix "$up/" --dest "$SPECS/$local_dir" "$@"
}

# ---------------------------------------------------------------- main

all_robots="$(awk -F '\t' '$2 == "id" { print $1 }' <<<"$SPEC")"
if [ $# -eq 0 ]; then
  robots="$all_robots"
else
  robots=""
  for r in "$@"; do
    grep -qx -- "$r" <<<"$all_robots" || die "unknown robot '$r'; robots.yml has: $(echo $all_robots)"
    robots+="$r"$'\n'
    base="$(spec "$r" base)"
    [ -z "$base" ] || robots+="$base"$'\n'
  done
fi
wanted() { grep -qx -- "$1" <<<"$robots"; }

# Every manifest line must belong to a source, or it could never be restored.
while read -r _ path; do
  covered=""
  for row in "${SOURCES[@]}"; do
    IFS='|' read -r _ _ local_dir _ <<<"$row"
    case "$path" in "$local_dir"/*) covered=1 ;; esac
  done
  [ -n "$covered" ] || die "$path is in meshes.sha256 but no source in $(basename "$0") provides it"
done <"$MANIFEST"

checked=""
for row in "${SOURCES[@]}"; do
  IFS='|' read -r robot key local_dir up <<<"$row"
  wanted "$robot" || continue
  checked+="$local_dir"$'\n'
  todo="$(manifest_under "$local_dir" | stale)"
  if [ -z "$todo" ]; then
    echo ">> $robot: $local_dir up to date"
    continue
  fi
  rels=()
  while read -r p; do rels+=("${p#"$local_dir"/}"); done <<<"$todo"
  echo ">> $robot: fetching ${#rels[@]} file(s) into robots_specs/$local_dir"
  if [ -n "$(spec "$robot" "$key.url")" ]; then
    fetch_drive "$robot" "$key" "$up" "$local_dir" "${rels[@]}"
  else
    repo="$(spec "$robot" "$key.repository")"
    rev="$(spec "$robot" "$key.revision")"
    [ -n "$repo" ] && [ -n "$rev" ] || die "$robot: robots.yml has no $key.repository/revision"
    fetch_git "$repo" "$rev" "$up" "$local_dir" "${rels[@]}"
  fi
done

# Verify everything this run was responsible for, fetched or not.
bad=""
total=0
while read -r local_dir; do
  [ -n "$local_dir" ] || continue
  lines="$(manifest_under "$local_dir")"
  total=$((total + $(grep -c . <<<"$lines")))
  mismatched="$(stale <<<"$lines")"
  [ -z "$mismatched" ] || bad+="$mismatched"$'\n'
done <<<"$checked"
[ -z "$bad" ] || die "sha256 mismatch against robots_specs/meshes.sha256:"$'\n'"$bad"
echo ">> $total mesh file(s) verified against robots_specs/meshes.sha256"
