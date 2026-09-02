import datetime
from bisect import bisect_left, bisect_right

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymongo


# 设置中文字体，避免标题和坐标轴中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 建立 MongoDB 连接
myclient = pymongo.MongoClient("mongodb://192.168.1.126:27017/")

# 四种前缀和四种编号组合，共 16 个场景数据库
prefixes = ["AN", "ANR", "AS", "ASR"]
suffixes = ["900-0", "1000-0", "1100-0", "1200-0"]
db_names = [f"{prefix}-{suffix}" for prefix in prefixes for suffix in suffixes]

# 不同运行场景的运行容量
capacity_by_prefix = {
    "AN": 74,
    "ANR": 73,
    "AS": 82,
    "ASR": 82,
}


def get_direction(routing):
    """从 routing 中提取独立的方向段 A 或 D。

    数据中存在多种 routing 格式，例如：
    - GUDRO-ZHHH-A
    - ESMEB-D-ZHHH
    - ENLAB-A-ZHHH
    - ZHHH-D-OLMIB
    因此不能只取最后一个 '-' 后面的字段，而应扫描所有字段。
    """
    if not routing:
        return ''

    for part in routing.split('-'):
        if part in ('A', 'D'):
            return part
    return ''


for db_name in db_names:
    print(f"开始处理数据库: {db_name} ...")

    mydb = myclient[db_name]
    plan = list(mydb['FlightRecord'].find())

    if not plan:
        print(f"警告: 数据库 {db_name} 的 FlightRecord 为空，已跳过。")
        continue

    scenario_prefix = db_name.split('-')[0]
    capacity = capacity_by_prefix[scenario_prefix]
    threshold = capacity * 0.8  # 80% 容量阈值

    # 根据运行方向确定西跑道和东侧跑道组
    if scenario_prefix in ("AN", "ANR"):
        west_runways = {"ZHHH#04"}
        east_runways = {"ZHHH#05L", "ZHHH#05R"}
    else:
        west_runways = {"ZHHH#22"}
        east_runways = {"ZHHH#23L", "ZHHH#23R"}

    parsed_records = []
    west_landing_times = []
    east_landing_times = []

    for item in plan:
        # 按 pid 字段再次确认当前场景，避免混入其他场景数据
        if item.get('pid') != db_name:
            continue

        routing = item.get('routing', '')

        # 只统计与 ZHHH 相关的航班；A 表示进场，D 表示离场
        if 'ZHHH' not in routing:
            continue

        direction = get_direction(routing)
        if direction == 'A':
            actual_landing_time = item.get('actualLandingTime', 2147483647)

            # 进场时间采用 actualLandingTime，无效时回退到 scheduledLandingTime
            target_time_sec = actual_landing_time
            if target_time_sec == 2147483647:
                target_time_sec = item.get('scheduledLandingTime', 0)
            flight_type = '进场'

            # 五边冲突只使用有效的实际降落时间
            if actual_landing_time != 2147483647:
                arrival_runway = item.get('arrivalRunway', '')
                if arrival_runway in west_runways:
                    west_landing_times.append(actual_landing_time)
                elif arrival_runway in east_runways:
                    east_landing_times.append(actual_landing_time)

        elif direction == 'D':
            # 离场时间采用 actualTakeOffTime，无效时回退到 scheduledTakeOffTime
            target_time_sec = item.get('actualTakeOffTime', 2147483647)
            if target_time_sec == 2147483647:
                target_time_sec = item.get('scheduledTakeOffTime', 0)
            flight_type = '离场'
        else:
            continue

        target_time_sec = int(target_time_sec)
        time_hms = str(datetime.timedelta(seconds=target_time_sec))
        hour_bin = (target_time_sec // 3600) % 24

        parsed_records.append({
            'ID': item.get('id'),
            'Pid': item.get('pid'),
            'Type': flight_type,
            'Time_Sec': target_time_sec,
            'Time_HMS': time_hms,
            'Hour': hour_bin,
        })

    if not parsed_records:
        print(f"警告: 数据库 {db_name} 中没有解析到有效的航班数据，已跳过。")
        continue

    df = pd.DataFrame(parsed_records)

    # 按小时统计进离场总流量，并补齐 0-23 小时
    hourly_counts = df.groupby('Hour').size()
    hourly_counts = hourly_counts.reindex(range(24), fill_value=0).astype(int)

    total_all = hourly_counts.sum()

    # 计算每小时五边冲突数：以西跑道降落时刻为基准，统计东侧跑道组 ±90 秒内的降落航班数
    east_sorted = sorted(east_landing_times)
    conflict_counts = [0] * 24

    for west_time in west_landing_times:
        left = bisect_left(east_sorted, west_time - 90)
        right = bisect_right(east_sorted, west_time + 90)
        conflict_num = right - left

        hour_bin = (west_time // 3600) % 24
        conflict_counts[hour_bin] += conflict_num

    conflict_counts = np.array(conflict_counts)
    total_conflicts = int(conflict_counts.sum())

    # 开始绘图，适当增高图片以容纳底部图例
    plt.figure(figsize=(12, 7))

    max_y = hourly_counts.max()
    upper_y = max(max_y * 1.2 if max_y > 0 else 0, capacity * 1.15, threshold * 1.2, 10)
    plt.ylim(0, upper_y)

    x = np.arange(24)
    width = 0.5

    rects = plt.bar(
        x,
        hourly_counts,
        width,
        label=f'总流量 ({total_all})',
        color='#4CAF50',
    )

    plt.bar_label(
        rects,
        labels=[str(i) if i > 0 else '' for i in hourly_counts],
        padding=3,
        fontsize=9,
    )

    # 运行容量红色实线
    plt.axhline(
        y=capacity,
        color='red',
        linestyle='-',
        linewidth=2,
        label=f'运行容量 ({capacity})',
    )

    # 80% 容量阈值红色虚线
    plt.axhline(
        y=threshold,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'80%容量阈值 ({threshold:.1f})',
    )

    plt.title(f'场景"{db_name}"下的就近降落航班阈值分析', fontsize=16)
    plt.xlabel('时间 (小时)', fontsize=12)
    plt.ylabel('航班数量 (架次)', fontsize=12)

    x_labels = [f"{h:02d}:00-{(h + 1):02d}:00" for h in hourly_counts.index]
    plt.xticks(x, x_labels, rotation=45, ha='right')

    ax1 = plt.gca()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # 以左侧纵轴当前刻度为基准，先固定左轴，再让右轴完全复制
    left_ylim = ax1.get_ylim()
    left_ticks = list(ax1.get_yticks())
    left_tick_labels = [f"{v:g}" for v in left_ticks]

    ax1.set_yticks(left_ticks)
    ax1.set_yticklabels(left_tick_labels)

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        conflict_counts,
        color='#FF9233',
        marker='o',
        linewidth=2,
        label=f'五边冲突数 ({total_conflicts})',
        zorder=5,
    )
    ax2.set_ylabel('五边冲突数 (次)', fontsize=12)
    ax2.set_ylim(left_ylim)
    ax2.set_yticks(left_ticks)
    ax2.set_yticklabels(left_tick_labels)

    # 在每个非零折线点上标注数值，格式与柱形图数字标注一致
    for xi, yi in zip(x, conflict_counts):
        if yi > 0:
            ax2.annotate(
                str(int(yi)),
                (xi, yi),
                textcoords='offset points',
                xytext=(0, 7),
                ha='center',
                fontsize=9,
            )

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    # 将图例移到图表下方并横向排布，避免遮挡柱状图和参考线
    ax1.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.24),
        ncol=4,
        frameon=False,
    )

    plt.tight_layout()

    file_name = f'{db_name}-就近降落航班阈值分析.svg'
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    print(f"已成功生成并保存图表: {file_name}")
    print(f"该场景五边冲突总数为: {total_conflicts}\n")
    plt.close()


print("所有场景数据库的就近降落航班总流量及五边冲突数绘制工作全部完成！")
