#!/usr/bin/env bash
# RadioML 2018.01A完整数据下载；不启动训练，不使用GPU。
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'HELP'
用法：bash code/training/download_radioml2018.sh [--check-only] [输出目录]

默认目录：artifacts/external_datasets/RadioML2018_01A
依赖：curl、python3、flock（Linux util-linux）。
可选：h5py，用于检查HDF5的X/Y/Z结构；不安装则只检查ZIP CRC和文件签名。
重新运行相同命令可续传下载；解压中断后也可重试。建议预留50GB空间。
--check-only：仅测试512字节下载，不开始完整数据下载。
默认IPv4、HTTP/1.1；自动重试连接重置，支持两个同版本Kaggle入口。

环境变量：
  PYTHON_BIN              Python解释器，默认python3
  RADIOML_DOWNLOAD_URL     可替换的ZIP直链；默认Kaggle公开镜像v2
  HTTPS_PROXY / ALL_PROXY  按curl标准规则使用现有代理设置

镜像：https://www.kaggle.com/datasets/pinxau1000/radioml2018
原始发布：https://www.deepsig.ai/datasets/
数据许可：CC BY-NC-SA 4.0；保留压缩包内LICENSE.TXT。
HELP
    exit 0
fi
check_only=0
if [[ "${1:-}" == "--check-only" ]]; then
    check_only=1
    shift
fi
if (( $# > 1 )); then
    printf '只能提供一个输出目录；查看 --help。\n' >&2
    exit 2
fi

destination="${1:-artifacts/external_datasets/RadioML2018_01A}"
python_bin="${PYTHON_BIN:-python3}"
download_urls=(
    'https://api.kaggle.com/v1/datasets/download/pinxau1000/radioml2018?datasetVersionNumber=2'
    'https://www.kaggle.com/api/v1/datasets/download/pinxau1000/radioml2018?datasetVersionNumber=2'
)
if [[ -n "${RADIOML_DOWNLOAD_URL:-}" ]]; then
    download_urls=("$RADIOML_DOWNLOAD_URL")
fi
for program in curl flock "$python_bin"; do
    command -v "$program" >/dev/null || { printf '缺少依赖：%s\n' "$program" >&2; exit 1; }
done
mkdir -p -- "$destination"
destination="$(cd -- "$destination" && pwd -P)"
exec 9>"$destination/.download.lock"
flock -n 9 || { printf '同一目录已有下载程序运行：%s\n' "$destination" >&2; exit 1; }
archive="$destination/radioml2018_v2.zip"
partial="$archive.part"

printf 'RadioML 2018.01A完整数据集，默认使用Kaggle镜像v2\n输出：%s\n' "$destination"
printf '建议预留50GB；下载中断后重新执行相同命令即可续传。\n'

curl_options=(--ipv4 --http1.1 --fail --location --show-error --connect-timeout 15)
# curl普通--retry不覆盖所有TLS/连接重置错误。每次重启curl，下载时
# --continue-at -会重新读取当前文件长度，保留上次连接已写入的字节。
transfer() {
    local attempts="$1" status=0 attempt
    shift
    for ((attempt=1; attempt<=attempts; attempt++)); do
        if curl "${curl_options[@]}" "$@"; then
            return 0
        else
            status=$?
        fi
        case "$status" in
            5|6|7|18|28|35|52|55|56|92) ;;
            *) return "$status" ;;
        esac
        if ((attempt < attempts)); then
            printf '\n连接失败（curl=%s），第%s/%s次重试，2秒后继续。\n' \
                "$status" "$attempt" "$((attempts-1))" >&2
            sleep 2
        fi
    done
    return "$status"
}

if [[ ! -f "$archive" || "$check_only" == 1 ]]; then
    # 先请求512字节，拒绝登录/验证网页，避免把HTML保存成数据集。
    probe="$destination/.download_probe"
    download_url=''
    for candidate in "${download_urls[@]}"; do
        candidate_host="$("$python_bin" - "$candidate" <<'PY'
import sys
from urllib.parse import urlsplit
print(urlsplit(sys.argv[1]).hostname or '自定义地址')
PY
)"
        printf '检查入口：%s（IPv4 / HTTP1.1）\n' "$candidate_host"
        if transfer 3 --silent --max-time 30 --range 0-511 --max-filesize 1048576 \
            --output "$probe" "$candidate"; then
            if "$python_bin" - "$probe" <<'PY'
import sys
from pathlib import Path
raise SystemExit(0 if Path(sys.argv[1]).read_bytes()[:4] == b'PK\x03\x04' else 1)
PY
            then
                download_url="$candidate"
                break
            fi
            printf '入口未返回ZIP，可能是登录/验证网页；尝试其他入口。\n' >&2
        else
            printf '此入口连接失败；尝试其他入口（如有）。\n' >&2
        fi
    done
    if [[ -z "$download_url" ]]; then
        printf '\n所有入口均不可用；未开始完整下载，已有.part未改动。\n' >&2
        printf '请检查本机到Kaggle和storage.googleapis.com的HTTPS连通性；如需代理，配置HTTPS_PROXY后重试。\n' >&2
        printf '可通过RADIOML_DOWNLOAD_URL指定可访问且支持Range的ZIP直链；脚本不会关闭证书验证。\n' >&2
        exit 1
    fi
    printf 'ZIP分段检查通过：%s\n' "$candidate_host"
    rm -f -- "$probe"
    if ((check_only)); then
        printf '仅检查完成，未启动完整数据集下载。\n'
        exit 0
    fi
    # 若上次在下载完成后、改名前中断，直接进入完整ZIP检查。
    if ! "$python_bin" - "$partial" <<'PY'
import sys, zipfile
raise SystemExit(0 if zipfile.is_zipfile(sys.argv[1]) else 1)
PY
    then
        transfer 8 --speed-limit 1024 --speed-time 120 \
            --continue-at - --output "$partial" "$download_url" || {
                printf '\n下载中断，已保留.part；重新执行相同命令即可续传。\n' >&2
                exit 1
            }
    fi
    "$python_bin" - "$partial" <<'PY'
import sys, zipfile
if not zipfile.is_zipfile(sys.argv[1]):
    raise SystemExit('下载内容不是完整ZIP；已保留.part，请检查下载来源。')
PY
    mv -- "$partial" "$archive"
else
    printf '已有ZIP，跳过下载，检查并补齐解压文件。\n'
fi

"$python_bin" - "$archive" "$destination" <<'PY'
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import time
import zipfile
import zlib

archive, destination = Path(sys.argv[1]), Path(sys.argv[2])
marker = destination/'download_manifest.json'
marker.unlink(missing_ok=True)
with zipfile.ZipFile(archive) as package:
    entries = package.infolist()
    datasets = [e for e in entries if not e.is_dir() and e.filename.lower().endswith(('.hdf5','.h5'))]
    if len(datasets) != 1:
        raise SystemExit('ZIP中应有一个完整HDF5数据文件；请检查下载版本。')
    for entry in entries:
        relative = PurePosixPath(entry.filename)
        if relative.is_absolute() or '..' in relative.parts or '\\' in entry.filename:
            raise SystemExit('拒绝非法压缩包路径：'+entry.filename)
        if stat.S_ISLNK(entry.external_attr >> 16):
            raise SystemExit('拒绝压缩包中的符号链接：'+entry.filename)
        output = destination.joinpath(*relative.parts)
        if not output.resolve().is_relative_to(destination.resolve()):
            raise SystemExit('解压目标超出输出目录：'+entry.filename)
        if entry.is_dir():
            output.mkdir(parents=True,exist_ok=True)
            continue
        if output.exists():
            crc = 0
            with output.open('rb') as stream:
                for chunk in iter(lambda: stream.read(8*1024*1024),b''):
                    crc = zlib.crc32(chunk,crc)
            if output.stat().st_size != entry.file_size or crc != entry.CRC:
                raise SystemExit('已有文件与压缩包不匹配，未覆盖：'+str(output))
            print('已有文件CRC检查通过：',output,flush=True)
            continue
        required = entry.file_size + 1024**3
        if shutil.disk_usage(destination).free < required:
            raise SystemExit(f'解压空间不足，当前文件至少需约{required/1024**3:.1f}GiB。ZIP已保留。')
        output.parent.mkdir(parents=True,exist_ok=True)
        temporary = output.with_name(output.name+'.extracting')
        if temporary.is_symlink():
            raise SystemExit('拒绝符号链接临时文件：'+str(temporary))
        print(f'解压并校验CRC：{entry.filename} ({entry.file_size/1024**3:.2f} GiB)',flush=True)
        # ZipExtFile在到达EOF时检查成员CRC，不将20GB数据一次性读入内存。
        with package.open(entry) as source, temporary.open('wb') as sink:
            shutil.copyfileobj(source,sink,length=8*1024*1024)
        os.replace(temporary,output)
    hdf5_path = destination.joinpath(*PurePosixPath(datasets[0].filename).parts)

with hdf5_path.open('rb') as stream:
    if stream.read(8) != b'\x89HDF\r\n\x1a\n':
        raise SystemExit('文件未通过HDF5签名检查：'+str(hdf5_path))
shapes = None
try:
    import h5py
except ImportError:
    print('未安装h5py：已检查ZIP CRC与HDF5签名；未检查X/Y/Z结构。',flush=True)
else:
    with h5py.File(hdf5_path,'r') as data:
        shapes = {name:list(data[name].shape) for name in ['X','Y','Z']}
        count = shapes['X'][0]
        if count != 2555904 or shapes['X'] != [count,1024,2] or shapes['Y'] != [count,24] or shapes['Z'] != [count,1]:
            raise SystemExit('数据形状与完整RadioML2018不符：'+str(shapes))
        for name in ['X','Y','Z']:
            data[name][0:1]
            data[name][-1:]
    print('HDF5结构检查通过：',shapes,flush=True)
summary = {'dataset':'RadioML 2018.01A','source':'Kaggle pinxau1000/radioml2018 v2 (or URL override)',
           'original':'https://www.deepsig.ai/datasets/','license':'CC BY-NC-SA 4.0',
           'archive':str(archive),'hdf5':str(hdf5_path),'shapes':shapes,
           'zip_crc_verified':True,'hdf5_schema_verified':shapes is not None,'verified_at':time.time()}
marker.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print('完成。数据文件：',hdf5_path,flush=True)
print('检查记录：',marker,flush=True)
PY
