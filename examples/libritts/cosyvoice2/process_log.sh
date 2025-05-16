#!/bin/bash

# 获取脚本所在目录
path=$(cd `dirname $0`;pwd)
cd $path
echo "当前工作目录: $path"

# 创建日志存储目录
log_dir="$path/logs"
if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
    echo "创建日志目录: $log_dir"
fi

# 获取昨天的日期作为日志文件名前缀
current_date=$(date -d "-1 day" "+%Y%m%d")
echo "处理日期: $current_date"

# 定义需要处理的nohup日志文件路径
# 默认处理当前目录下的nohup.out，如需处理其他文件，请修改以下变量
nohup_file="$path/nohup.out"

# 检查日志文件是否存在
if [ ! -f "$nohup_file" ]; then
    echo "日志文件不存在: $nohup_file"
    exit 1
fi

# 检查日志文件大小
file_size=$(du -h "$nohup_file" | cut -f1)
echo "当前日志文件大小: $file_size"

# 按照指定大小切分日志文件（每个文件约20MB）
split -b 20971520 -d -a 4 "$nohup_file" "$log_dir/log_${current_date}_"
echo "日志文件已切分到: $log_dir/log_${current_date}_XXXX"

# 清空原日志文件，但保持文件存在，程序可继续写入
cat /dev/null > "$nohup_file"
echo "已清空原日志文件: $nohup_file"

# 可选：删除过期日志文件（保留最近7天的日志）
find "$log_dir" -type f -name "log_*" -mtime +7 -exec rm {} \;
echo "已删除7天前的日志文件"

echo "日志处理完成: $(date)"
