# 髋部同轴双驱动轮腿：MuJoCo 台架与落地自平衡

## 工程边界

本目录只维护 MuJoCo 建模、控制、仿真配置、测试与结果。完整机器人硬件工程已迁至
[wheel_leg_hardware](../wheel_leg_hardware/README.md)，包括机械/电气结构、制造文件、BOM、打印试装包和历版资料。
仿真运行不依赖硬件目录；下文尺寸、质量、惯量和执行器参数是仿真模型的独立参数快照，不自动跟随硬件 V8.3 更新，也不能视为制造依据。

## 当前仿真布局：髋内置、膝外置

机身使用灰色半透明显示（RGBA=`0.28 0.32 0.38 0.35`），便于观察内部髋电机；仅改变外观，不改变质量、惯量和碰撞。

当前采用温和缩短方案：`OB=AC=BW=130 mm`，`OA=BC=35 mm`。保持连杆截面不变，估计连杆质量按长度缩放；电机、轮胎、轮辋和机身质量保持原参数。

机身保持 200×150×100 mm。左右髋电机中心位于 `Y=±50 mm`，电机包络落在各侧距中线 27–73 mm 范围，完整收进机身；定子刚性固定于机身。连接法兰中心位于侧壁 `Y=±75 mm`，髋转子通过法兰带动机身外部的膝电机，膝电机中心为 `Y=±100 mm`，内端面距中线 77 mm。

每侧电机及整条腿向内移动 48 mm，轮心位于 `Y=±137 mm`，轮距从 370 mm 缩至 **274 mm**。OA 仍连接膝转子，OB 仍连接膝定子。

两台髋电机之间净空约 54 mm。为避免与原先假设的电池/电子设备包络重叠，将二者估计宽度调整为 50 mm，电池/电子设备中心高度分别为 −20/+15 mm，质量保持 0.45/0.15 kg；惯量和机身质心已重算。该检查只针对简化包络，连接器、线缆和真实电池尺寸仍待实测确认。

最新布局通过 21 项测试，以及 `results/current/balance` 的站立、`results/current/leg_extension` 的屈伸、`results/current/self_righting` 的自扶正、`results/current/jump` 的跳跃落地和 `results/current/spin_extension` 的原地转向屈伸测试。此前不同参数和机构布局的结果统一保存在 `results/archive/`，不与当前基准混用。

## 工程组织与专项配置

仿真代码位于 `wheel_leg/`，检查/绘图工具位于 `tools/`，回归测试位于 `tests/`。MJCF 生成文件集中在 `models/generated/`，质量和验证报告集中在 `reports/`。

专项场景分别保存为独立 JSON 配置，并由自由基座仿真入口加载：

```text
configs/simulations/balance.json       # 落地自平衡与扰动
configs/simulations/leg_extension.json # 自平衡基础上的同步屈伸
configs/simulations/self_righting.json # 无动力躺地、自扶正、平衡交接
configs/simulations/jump.json          # 自平衡、下蹲、跳跃、落地再平衡
configs/simulations/spin_extension.json # 原地转向与同步伸缩腿
```

命令行参数可覆盖 JSON 中的同名参数；路径按项目根目录解析。MuJoCo 的 MJCF 也支持通过 `include` 做相对路径模块化，当前工程使用 Python 生成器集中生成最终模型，避免 GUI 运行时依赖隐含路径。参考 [MuJoCo XML 模块化文档](https://mujoco.readthedocs.io/en/latest/XMLreference.html#meta-elements)。

## 自平衡下双腿同步屈伸

```bash
cd /Users/lvjiaqing/MyProjects/MyRobot/wheel_leg_mujoco
.venv/bin/mjpython -m wheel_leg.simulate_balance --config configs/simulations/leg_extension.json
```

落地后先站稳 2 秒，再用 2 秒平滑过渡引入周期屈伸。默认目标腿长为约 `183.85±30 mm`（153.85–213.85 mm），周期 8 秒；腿长定义为髋轴 O 到轮心 W 在机身矢状面内的距离，不含横向电机安装偏置。双腿同步屈伸，机身随之升降，轮子保持接地并参与平衡。原定高模式仍可用 `--motion hold`（默认值）。

```bash
# 较小幅度：±20 mm，周期 6 秒
.venv/bin/mjpython -m wheel_leg.simulate_balance --config configs/simulations/leg_extension.json --leg-amplitude-mm 20 --leg-period 6 --duration 30

# 无窗口回归：较快屈伸期间施加反向推力
.venv/bin/python -m wheel_leg.simulate_balance --headless --config configs/simulations/leg_extension.json
.venv/bin/python tools/plot_balance.py results/current/leg_extension
.venv/bin/python -m unittest discover -s tests -p 'test_extension.py' -v
```

`--leg-amplitude-mm` 是半幅；当前要求整段目标腿长处于 120–225 mm 内，周期至少 4 秒，`--motion-start` 至少 2 秒。这些是当前控制器的使用边界，未验证任意幅度、周期和外推组合。启动时计算 9 个腿长的受载平衡点及 LQR 增益，可能需等待数秒。

控制器按腿长插值平衡姿态、支撑扭矩及反馈增益，并跟踪平滑目标速度；轮速参考考虑腿架转动和滚动关系。屈伸同样只通过六个电机力矩实现，不在运行中改写关节坐标。前馈为静态支撑前馈，动态偏差由反馈修正；受载闭环软约束使实际腿长与几何指令有小幅偏差。

日志新增目标腿长、左右实际腿长和目标机身高度；绘图生成 `leg_extension.png`。`motion_passed` 要求完成平滑启动后的至少一个完整周期，屈伸阶段双轮持续接地、最大腿长误差 <3 mm、俯仰/横滚 <5°，且满足原平衡判据和至少 75% 的目标峰峰行程。仿真过短会明确标记未完成周期，不算屈伸通过。

当前专项配置为 4 秒周期、60 秒运行、−5 N 推力；结果写入 `results/current/leg_extension`。历史对比结果位于 `results/archive/`。

## 原地转向并同步伸缩腿

该专项先保持站立平衡，再以五次曲线在 `3 秒` 内将机身航向转过 `90°`；转向期间腿长按 `183.85±10 mm` 周期变化，转向结束后继续保持 3 秒平衡。航向由左右轮毂电机的差动控制产生，机身前向位置/速度补偿使用轮毂共模力矩，未对机身施加外力或直接改写位姿。

```bash
cd /Users/lvjiaqing/MyProjects/MyRobot/wheel_leg_mujoco

# GUI
.venv/bin/mjpython -m wheel_leg.simulate_spin_extension \
  --config configs/simulations/spin_extension.json

# 无窗口回归
.venv/bin/python -m wheel_leg.simulate_spin_extension --headless \
  --config configs/simulations/spin_extension.json

# 专项曲线
.venv/bin/python tools/plot_spin_extension.py results/current/spin_extension
```

当前无窗口结果：`3 秒` 内实际转向 `88.82°`，最高航向角速度 `58.18°/s`，最大平面漂移 `64.1 mm`，最大俯仰 `3.00°`、最大横滚 `0.48°`，腿长目标范围 `20 mm`，最大腿长跟踪误差 `3.03 mm`，闭环结束后双轮持续接地并恢复平衡。默认工况是可重复的高速 `90°` 转向基准；在当前轮胎摩擦和质量估计下，直接将目标提高到 `120°` 以上会出现明显漂移或失稳，因此不把整圈 `360°` 作为当前通过标准。结果保存在 `results/current/spin_extension/summary.json`、`log.csv` 和 `spin_extension.png`。

## 自启动平衡后跳跃并落地

跳跃专项先用站立 LQR 稳定机身，再切换到约 `145 mm` 下蹲目标；随后双侧膝电机施加对称 `4.5 N·m` 蹬地脉冲，轮胎离地后六个电机释放，检测到双轮重新接地后恢复站立 LQR。只有轮地接触产生外力，没有直接对机身施加向上外力。

```bash
cd /Users/lvjiaqing/MyProjects/MyRobot/wheel_leg_mujoco

# GUI
.venv/bin/mjpython -m wheel_leg.simulate_jump \
  --config configs/simulations/jump.json

# 无窗口回归
.venv/bin/python -m wheel_leg.simulate_jump --headless \
  --config configs/simulations/jump.json
```

当前配置的无窗口结果：机身上升约 `106.6 mm`，离地约 `2.114 s`，约 `0.239 s` 后落地，最大俯仰约 `15.70°`，落地后重新平衡；膝关节限位最大超调约 `0.00123 rad`。相较此前 3.0 N·m / 155 mm 配置，跳高约提升 49%。结果保存在 `results/current/jump/summary.json` 和 `log.csv`。这是基于理想 MuJoCo 状态反馈的仿真结果，不代表实机已具备跳跃能力；真实系统仍需实测电机速度、冲击、结构强度、轮胎接触和热限制。

## 无动力躺地自启动

使用 `--startup self-right` 时，模型先以俯卧姿态放在地板上，前 `1 s` 不输出任何电机力矩；随后双侧膝电机以五次曲线渐增到 `+7 N·m`，通过腿部接触使机身回到接近直立姿态。检测到机身俯仰角小于 15°、俯仰角速度小于 3 rad/s 后，自动切换到原有全状态 LQR 平衡控制。自扶正超时后仍会切换控制器，但报告中的 `self_righting_passed` 为 false。

```bash
cd /Users/lvjiaqing/MyProjects/MyRobot/wheel_leg_mujoco
.venv/bin/mjpython -m wheel_leg.simulate_balance --config configs/simulations/self_righting.json
.venv/bin/python -m wheel_leg.simulate_balance --headless --config configs/simulations/self_righting.json
```

当前模型在无窗口测试中约 `1.98 s` 完成扶正并进入平衡，随后机身倾角收敛到 0°，双轮持续接地。自扶正阶段允许机身、连杆与地板接触；进入平衡后的接触判据恢复为仅允许车轮接地。该流程是基于理想状态反馈和已知初始躺姿的仿真验证，不等同于实机跌倒恢复控制。

## 落地自平衡（当前整机入口）

```bash
cd /Users/lvjiaqing/MyProjects/MyRobot/wheel_leg_mujoco
.venv/bin/python -m wheel_leg.build_ground_model
.venv/bin/mjpython -m wheel_leg.simulate_balance --config configs/simulations/balance.json
```

默认从轮底离地 1 cm、机身前倾 3° 启动，在第 5 秒施加 3 N、持续 0.2 秒的前向推力。程序在真实重力和轮地接触下运行，机身具有完整六自由度，仅通过六个电机输出力矩维持平衡。除了测试推力，没有对机身直接施加稳定力，没有把机身焊到世界，也没有运行时重写关节位置。

无窗口验证及曲线：

```bash
.venv/bin/python -m wheel_leg.simulate_balance --headless --config configs/simulations/balance.json --duration 60 --output results/archive/balance_60s
.venv/bin/python -m unittest discover -s tests -p 'test_ground_balance.py' -v
.venv/bin/python tools/plot_balance.py results/archive/balance_60s
```

更大扰动测试：

```bash
.venv/bin/mjpython -m wheel_leg.simulate_balance --config configs/simulations/balance.json --drop-height .03 --pitch-deg 5 --push-n 5 --duration 20
.venv/bin/mjpython -m wheel_leg.simulate_balance --config configs/simulations/balance.json --pitch-deg -5 --roll-deg 2 --push-n -5 --duration 20
```

`--push-n 0` 关闭外推。`--disable-feedback --push-n 0` 只保留静态前馈，用作失稳对照；已测约 0.43 秒触发跌倒判据。无窗口模式在跌倒、闭环误差 >5 mm、非车轮触地、数值警告或未达到稳定判据时返回非零退出码。稳定判据为完成时最后两秒双轮持续接地、俯仰/横滚均 <2°、前向速度 <0.05 m/s，且在推力结束至少两秒后评估。

### 质量和材质估计

已按用户授权使用明确标注的估计参数，统一保存于 `configs/model/physical_parameters.json`。修改后重新执行 `python -m wheel_leg.build_ground_model`；程序生成 `models/generated/ground_robot.xml` 和 `reports/MASS_REPORT.json`。

| 部件 | 建模依据 | 默认值 |
| --- | --- | --- |
| 机身 | 20×15×10 cm 闭合铝壳，2 mm 壁厚 | 0.6827 kg |
| 电池 | 10×5×3.5 cm 等效盒，中心位于机身中心下 2 cm | 0.45 kg |
| 电子设备 | 10×5×1.5 cm 等效盒，中心位于机身中心上 1.5 cm | 0.15 kg |
| 连杆、法兰、支架 | 按现有等效实体几何 × 铝密度 | 2700 kg/m³ |
| 橡胶轮胎 | 外半径 50 mm、内半径 43 mm、宽 32 mm 的环体 | 密度 1100 kg/m³（估计） |
| 铝轮辋 | 外半径 43 mm、内半径 34 mm、宽 24 mm 的环体 | 密度 2700 kg/m³ |
| 髋/膝电机 | 每台总质量保持已确认值 | 各 0.300 kg |
| 轮毂电机 | 每台总质量保持用户提供值 | 各 0.360 kg |
| 干燥刚性平面接触 | 轮胎滑动摩擦系数，软接触时间常数 | 0.8、0.008 s（估计） |

机身含电池/电子设备 **1.2827 kg**，整机 **4.3426 kg**。铝密度取工程近似 2700 kg/m³，参考 [Hydro 6061 数据表](https://www.hydro.com/globalassets/01-products--services/extruded-profiles/americas/ena-resources/alloy-data-sheets/hydro_2019_data_sheet_6061.pdf)。材料选择、壁厚、橡胶密度、摩擦、载荷位置均未经实物测量。连杆孔洞、紧固件和轴承尚未逐件建模；电机定/转子质量仍暂按 50/50 拆分，惯量为几何估计。

机身采用外盒减内盒的壳体惯量，并用平行轴定理合并电池/电子设备；轮胎/轮辋采用环体惯量。MuJoCo 编译器使用 `inertiafromgeom="auto"`，保留显式机身和车轮惯量。材料在这里用于质量、惯量及接触参数，结构为刚体，未建模弹性应力或橡胶有限元形变。

### 控制与验证边界

`balance_control.py` 先求静态接触平衡点，包括软接触压入量及闭环受力变形，再用 MuJoCo `mjd_transitionFD` 线性化完整离散动力学，通过 Riccati 迭代（必要时使用标准离散 Riccati 求解回退）获得 LQR。反馈量来自仿真真值，包含姿态、位移、关节角及速度；横向绝对位置和轮子绝对转角不作为普通站立调节目标，原地转向专项另加机身坐标系驻车补偿。控制器不需要预存增益，启动时按加载的模型计算。

峰值限幅为髋/膝 ±7 N·m、轮毂 ±2 N·m；日志另记超过额定 3/1 N·m 的时间比例。基准落地测试膝电机短暂达到约 3.35 N·m，超过额定约 11 ms，稳态约 1.54 N·m。未建模温升、真实电流环、轮毂转速上限、传感器噪声或通讯延迟，因此这些结果仅证明当前估计模型的局部自平衡，不能直接用作实机控制器。

已完成当前 130 mm 连杆模型的 30 秒站立、60 秒屈伸、10 秒无动力躺地自扶正、6 秒跳跃落地和 11 秒高速原地转向屈伸测试。自扶正约 1.98 秒完成姿态交接；当前增强跳跃配置约上升 106.6 mm、飞行 0.239 秒后落地；原地转向配置在 3 秒内完成约 90° 转向、最高航向角速度约 58.2°/s、最大漂移 64.1 mm，之后双轮持续接地并收敛到直立平衡。自扶正阶段允许机身/连杆接触地面，跳跃空中阶段和转向专项均无非车轮触地。

当前结果位于 `results/current/`；旧参数、旧布局和负向对照位于 `results/archive/`。参考 [MuJoCo 数值线性化接口](https://mujoco.readthedocs.io/en/3.2.7/APIreference/APIfunctions.html#mjd-transitionfd) 和 [接触模型](https://mujoco.readthedocs.io/en/3.2.7/computation/index.html#contact)。

## 适用拓扑与边界

髋电机与膝电机同轴，髋电机在内侧、膝电机在外侧。髋电机定子固定于髋部支架，髋电机转子通过法兰盘连接膝电机定子，因此髋电机旋转时会带动膝电机和整个腿部前后摆动。整个连杆机构挂在膝电机外侧：OA 主动臂连接膝电机转子，OB 固定臂连接膝电机定子，AC 和 BC 为被动连杆；B-C-W 是同一根刚性输出杆。闭环为 O-A-C-B-O，C 点用被动连接约束闭合。

髋电机定子通过无关节子 body 固连机身内部，没有中间固定方块。机身尺寸为 `20×15×10 cm`，髋部连接基准位于侧壁 `Y=±75 mm`。台架 `models/generated/biped_wheel_leg.xml` 保留固定、零机身质量；整机落地使用 `models/generated/ground_robot.xml`，二者都采用髋内置布局。

本工程以运动学/控制验证为目的；动力学质量采用等效几何体，电机外观由原生圆柱、端盖及转动标记绘制，不依赖 ROS 2。
以下台架章节中的 `wheel_leg.simulate` / `wheel_leg.simulate_biped` 默认髋部固定、车轮悬空、关闭重力；落地自平衡须使用 `wheel_leg.simulate_balance`。两类模型均为控制研究模型，不是制造尺寸或承载认证模型。

**验证状态：项目 `.venv` 已安装 Python 3.12、MuJoCo 3.2.7、NumPy 和 Matplotlib。当前模型已通过 21 项 Python 单元/动力学测试、1681 个单腿目标姿态闭环检查、441 组双腿姿态闭环检查，以及站立、屈伸、无动力自扶正、跳跃落地和原地转向屈伸测试。旧日志目录仅作历史记录。**

## 环境

本示例固定 `mujoco==3.2.7`，不是最新版本；该版本有 Apple Silicon 和 Intel macOS 二进制包。项目使用 Python 3.12 虚拟环境 `.venv`。
使用官方 `mujoco` 包，不是 `mujoco-py`。包内包含引擎和查看器，不需单独安装 MuJoCo App。

已安装 Homebrew、但缺少 Python 3.12 时：

```bash
brew install python@3.12
```

解压本目录后，在 macOS 本地终端执行，不要在 Remote SSH 终端中运行图形窗口：

```bash
cd wheel_leg_mujoco
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import platform, mujoco; print(platform.machine(), mujoco.__version__)"
which python
which mjpython
```

Apple Silicon 应使用原生 arm64 Python，不要在 Rosetta x86_64 环境下运行。原生 Intel Mac 使用 x86_64 Python。检查当前 Python 进程架构，不要只看硬件名称。

## 电机资料与仿真映射

资料源目录：

```text
/Users/lvjiaqing/MyProjects/MyRobot/references/DM-J4310-2EC
/Users/lvjiaqing/MyProjects/MyRobot/references/DM-H6215
```

当前工程使用 `motor_specs.py` 保存资料参数，并由 `build_model.py` 写入 MJCF 外形、质量分配和髋/膝驱动限幅。

| 电机 | 仿真位置 | 资料确认参数 |
| --- | --- | --- |
| DM-J4310-2EC V1.1 | 髋、膝 | 24 V；额定 3 N·m；峰值 7 N·m；额定输出 120 rpm；24 V 空载最大 200 rpm；减速比 10:1；外径 57 mm；轴向高度 46 mm；质量约 300 g |
| DM-H6215 | 轮毂 | 24 V；CAN 1 Mbps；外径 68 mm；轴向总宽 44.5 mm；质量 360 g；额定 1 N·m、峰值 2 N·m（均为用户提供值；说明书未列出扭矩和转速表） |

DM-J4310 的 `hip_motor`、`knee_motor` 控制限幅已改为峰值 `±7 N·m`；额定 3 N·m 仅作为连续工作参考。DM-H6215 的轮毂执行器控制限幅为用户提供的峰值 `±2 N·m`，额定 `1 N·m` 仅作为连续工作参考。当前仿真尚未建模温升、电流环、过载、扭矩-转速曲线和通讯丢失保护。

## 自绘电机模型

仿真不再加载 STEP/STL 或依赖 FreeCAD。尺寸与质量继续使用 `motor_specs.py` 中已确认的参数；外观是简化示意，并非完整制造模型。

| 电机 | 定子 | 转子 |
| --- | --- | --- |
| 髋电机 | 蓝色 | 橙色 |
| 膝电机 | 青色 | 黄色 |
| 轮毂电机 | 紫色 | 红色 |

髋/膝电机：外圆柱为定子，内圆柱为转子；轮毂电机相反，外圆柱为转子、内圆柱为定子。内圆柱半径和轴向高度均为外圆柱的 1/2，朝机身外侧的端面齐平。外圆柱的前半部采用同心环壳，为内圆柱留出空间；髋/膝定子半透明，便于观察内部转子。

J4310 外圆柱半径 28.5 mm、高 46 mm，内圆柱半径 14.25 mm、高 23 mm；H6215 外圆柱半径 34 mm、高 44.5 mm，内圆柱半径 17 mm、高 22.25 mm。左右电机朝外镜像，端面关系一致。环壳顶点由 `motor_visuals.py` 数学生成并直接嵌入 MJCF，无外部模型文件。

OA 通过同轴黄色输出轴连接膝转子；OB 的可见根部通过偏置支架连接膝定子的外环，避开中心转子。膝转子与 OA 属同一刚体；OB 与膝定子刚性连接，膝关节转动不会驱动 OB。髋转子通过中心连接件带动膝定子；轮毂外转子随车轮转动，内定子固定于 BC/BW 输出杆。

本次调整的是外观与连接示意，质量/惯量继续由原来的透明等效几何体承担，外观部件质量为 0；每台髋/膝电机 300 g、轮毂电机 360 g，定转子质量仍为暂定 50/50 分配。新简化外形不代表厂商内部真实结构或实测惯量。

历史 CAD 转换脚本和网格文件已迁至硬件工程的 `tools/convert_cad_models.py` 和 `assets/meshes/`；运行仿真不读取它们。几何体定义参考 [MuJoCo 官方文档](https://mujoco.readthedocs.io/en/3.2.7/XMLreference.html#body-geom)。

## 文件

```text
wheel_leg_mujoco/
├── README.md
├── requirements.txt
├── wheel_leg/                 # 可复用建模、控制和仿真入口
│   ├── geometry.py            # 纯 Python 几何正逆解
│   ├── motor_specs.py         # DM-J4310-2EC V1.1 / DM-H6215 参数
│   ├── motor_visuals.py       # 程序化定子/转子外观
│   ├── build_*.py             # 单腿、双腿、落地 MJCF 生成器
│   ├── balance_control.py     # 平衡点求解、LQR、自扶正
│   ├── simulate*.py           # 台架、双腿、落地、跳跃、转向 GUI/无窗口入口
│   ├── paths.py               # 工程目录唯一来源
│   └── config.py              # JSON 专项配置加载
├── configs/
│   ├── model/physical_parameters.json
│   └── simulations/           # balance / leg_extension / self_righting / jump / spin_extension
├── models/generated/          # 生成的 wheel_leg.xml 等 MJCF
├── tools/                     # 仅 check、plot 仿真检查与绘图工具
├── tests/                     # 21 项单元和动力学回归测试
├── reports/                   # MASS_REPORT.json、VALIDATION.json
└── results/
    ├── current/               # 当前基准结果
    └── archive/               # 历史布局、参数和失败对照
```

## 按顺序运行

### 1. 几何与模型编译

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python tools/check_model.py
```

`tools/check_model.py` 实际调用 MuJoCo 编译器和 `mj_forward`，扫描 41×41 个姿态，并检查 body 层级：`leg_carrier` 挂在固定髋部，`knee_motor` 挂在腿架，`linkage_base` 挂在膝电机定子，OA/AC 与 BC/BW 两个分支都挂在该连杆基座下。
预期模型包含 `nq=5, nv=5, nu=3, neq=1`。由于闭环点连接在这个平面系统中的有效约束秩是 2，理想独立运动自由度为 5-2=3，即整体腿架摆动、腿部伸缩和车轮自转。`nv=5` 不是五个独立驱动。
这是运动学自检，不是动力学稳定性或承载测试。

双腿模型检查：

```bash
.venv/bin/python -m wheel_leg.build_biped_model
.venv/bin/python tools/check_biped_model.py
```

双腿模型预期为 `nq=10, nv=10, nu=6, neq=2`；两个 C 点闭环分别提供平面约束，理想独立自由度为 `10-4=6`。

### 2. 可选：纯运动学预览

```bash
.venv/bin/mjpython -m wheel_leg.simulate --kinematic --test extend --duration 16 --output results/current/kinematic
```

本模式直接设置关节坐标，仅检查形态、方向和运动轨迹。没有用力矩驱动，不能用来证明电机能力。不要把这个模式与 `--gravity`、`--load-n` 混用。

### 3. 真正的动力学：无重力、车轮悬空

先执行无窗口测试，再执行 GUI：

```bash
.venv/bin/python -m wheel_leg.simulate --headless --test extend --duration 16 --output results/current/extend
.venv/bin/mjpython -m wheel_leg.simulate --test extend --duration 24 --output results/current/extend_gui
.venv/bin/mjpython -m wheel_leg.simulate --test swing --duration 24 --output results/current/swing
.venv/bin/mjpython -m wheel_leg.simulate --test combo --duration 24 --output results/current/combo
```

`extend`：摆角维持 0，默认腿长约在 149~219 mm 往复。
`swing`：腿长保持约 212.1 mm，虚拟摆角在 ±15° 内往复。
`combo`：两种动作组合。
`hold`：维持初始姿态。
轨迹有两秒平滑启动，运动周期默认为八秒。动力学模式只发送电机输出力矩，不逐帧强制设置关节角。轮子默认以零相对转速为目标，通过限幅速度反馈控制。

双腿 GUI：

```bash
.venv/bin/python -m wheel_leg.build_biped_model
.venv/bin/mjpython -m wheel_leg.simulate_biped --test combo --duration 24 --output results/current/biped_combo_gui
```

双腿无窗口测试：

```bash
.venv/bin/python -m wheel_leg.simulate_biped --headless --test combo --duration 8 --output results/current/biped_combo
```

### 4. 加入重力、等效轮轴载荷与车轮反力矩

```bash
# 仅支承腿部自身质量；不是整机承重测试
.venv/bin/mjpython -m wheel_leg.simulate --test extend --gravity --output results/current/gravity

# 可选：用理想模型进行偏置力补偿，便于对比 PD 的稳态误差
.venv/bin/mjpython -m wheel_leg.simulate --test extend --gravity --bias-comp --output results/current/compensated

# 轮轴 W 处逐渐增加到世界向上 20 N 的外力；仍然不接触地面
.venv/bin/mjpython -m wheel_leg.simulate --test hold --gravity --load-n 20 --output results/current/load20

# 轮子相对 BC/BW 输出杆转速目标 3 rad/s，观察驱动反力矩
.venv/bin/mjpython -m wheel_leg.simulate --test hold --gravity --wheel-speed 3 --output results/current/wheel
```

外载由 `mj_applyFT` 施加于 BC/BW 输出杆的轮轴位置，不能解释为真实地面接触或机身载荷已被验证。
髋部被固定时，增加固定机身的质量不会产生真实的腿部支撑载荷。整机支撑测试需要可动机身和轮地接触。

### 5. 检查日志

```bash
.venv/bin/python tools/plot_log.py results/current/extend/log.csv
open results/current/extend/length.png
open results/current/extend/wheel_trajectory.png
cat results/current/extend/summary.json
```

每个输出目录包含 `log.csv`、`summary.json`。绘图脚本生成腿长、摆角、命令转矩、闭环误差、轮轴轨迹五张独立图。
程序记录 C 点两端距离、目标跟踪误差、平行关系误差及力矩饱和比例；闭环误差超过 5 mm 时主动停止。5 mm 是防止继续输出明显无效运动的中止值，不是设计验收标准。

对这个 130 mm 杆长的入门模型，可把零重力慢速运动下闭环误差 <0.1 mm、轮轴跟踪误差约 1~2 mm 作为最初调试目标，而非已实测结果或通用标准。关注误差对步长、载荷、控制增益及约束参数的敏感性。

## 机构坐标、角度和闭环建模

长度单位 m，质量 kg，时间 s，角度 rad，力矩 N·m。
世界坐标 X 向前、Y 向左、Z 向上。腿部铰链轴统一为局部 -Y，因此角度增加使局部向下的杆向 +X 转动。车轮转轴单独取 +Y。

关节角直接按四杆几何定义：

```text
theta_OB = q_hip
theta_OA = q_hip + q_knee_drive     # 膝转子驱动的主动臂 OA
theta_AC = theta_OA + q_bearing_A  # A 被动轴承后的 AC
theta_BW = q_hip + q_bearing_B     # B 被动轴承后的 BW
```

OA 与 BW 反向共线，AC 与 OB 平行；A、B 是树内被动铰链，C 是第三个被动轴承。所有杆件转轴平行于 Y 轴，连杆平面位于膝电机外侧 +Y。

对于等长杆 l 和虚拟腿长 L、虚拟摆角 beta：

```text
alpha = acos(L / (2*l))
q_hip   = beta - alpha
q_hip = beta - alpha
q_knee_drive = 2*alpha - pi         # 膝转子相对膝定子的主动角
q_bearing_A = pi - 2*alpha           # A 被动轴承
q_bearing_B = 2*alpha                # B 被动轴承
W - O   = (L*sin(beta), 0, -L*cos(beta))
```

MJCF 树在 C 处切开，连杆平面沿 +Y 放到膝电机外侧：

```text
固定髋部
└── leg_carrier [hip 驱动]
    └── knee_motor [外侧定子，随髋转动]
        └── linkage_base [固定臂 OB]
            ├── active_arm_OA [knee_drive 驱动]
            │   └── coupler_AC [A 被动轴承]
            └── output_link_BCW [B 被动轴承；BC/BW 同一刚杆]
                └── wheel [wheel_spin 驱动]

C_AC <== equality/connect ==> C_BC
```

单个 `connect` 是点重合约束，本质上是 C 处的被动球铰；本例所有树内转轴已经平行、运动已经被限制在同一个平面内，所以在 C 补一个点连接即可。一般三维机构不能不加分析地照搬这个做法。不要用 `weld` 把 AC 和 BC/BW 锁成刚体；A、B 仍是被动关节，真正的膝电机驱动关节名为 `knee_drive`。

初始 body 姿态预先处于闭合构型，通过 joint 的 `ref` 令初始 `qpos` 与上面的几何角一致。
改零位时必须同步理解 `euler`、`ref` 和目标角，否则初始闭合或跟踪关系会出错。

## 示例参数及其含义

- OB = AC = BW = 0.130 m；OA = BC = 0.035 m；轮胎半径 0.050 m。
- 髋高度 0.360 m，因此默认车轮不接地。
- 在连杆截面不变的估计下，固定臂 OB、刚性输出杆 BC/BW、主动臂 OA、被动连杆 AC 质量分别约为 0.104、0.143、0.040、0.043 kg；髋-膝法兰盘 0.06 kg，B 处轴承座 0.012 kg，轮轴支架 0.015 kg。
- 髋电机总质量 0.30 kg，膝电机总质量 0.30 kg，均按定子/转子各 0.15 kg 分配；末端轮毂电机总质量 0.36 kg，按定子/转子各 0.18 kg 分配。
- 原有 0.40 kg 轮体质量拆为轮胎 0.26 kg 和轮辋 0.14 kg；因此末端轮系总质量约 0.76 kg。上述均为可替换占位参数，不是厂家数据。
- 按当前默认 `MassProperties`，固定髋部以下的单腿建模总质量约 1.777 kg，双腿机构约 3.554 kg（不含机身、电池和电子设备）。`check_model.py` 会校验该质量汇总。
- DM-J4310 髋/膝驱动限幅为峰值 ±7 N·m，额定扭矩为 3 N·m；DM-H6215 轮毂驱动限幅为用户提供的峰值 ±2 N·m，额定扭矩为 1 N·m；两项 H6215 扭矩值尚未在所给说明书中交叉确认。
- 自绘电机外观部件质量为 0，质量与惯量由透明等效几何体承担。H6215 定子归属 BC/BW 输出杆，转子归属轮子 body；当前只有 2 个数学生成的内嵌环壳网格，不加载 CAD/STL。
- 理想输出力矩电机 `gear=1`。这里已经建在减速器输出轴一侧，不要再把输出力矩乘一次减速比。
- 默认 kp=25 N·m/rad、kd=0.7 N·m·s/rad，都是调试起点。
- `armature`、关节阻尼同样是假设值，没有建模电流环、扭矩-转速曲线、温升、通信时延和齿隙。
- 连杆/电机壳体的碰撞禁用，只有轮胎和地面具有碰撞属性。因此动画不能证明 CAD 中无干涉。

## 修改杆长

不要只改 XML 里某一个 0.15。通过生成器一起更新所有依赖位置和几何：

```bash
.venv/bin/python -m wheel_leg.build_model --length-mm 140 --crank-mm 35 --wheel-radius-mm 50
.venv/bin/python tools/check_model.py
.venv/bin/mjpython -m wheel_leg.simulate --test extend
```

这是等长 OB/AC、OA/BC 示例；改为不等长四杆时还需修改逆解与轨迹生成，不能只替换一个长度。
`tests/test_geometry.py` 包含默认文件一致性检查，修改参数生成模型后可运行 `tools/check_model.py`；恢复默认后再运行全部默认文件测试。

## 后续扩展

已具备固定台架和三维自由基座落地自平衡。后续优先用实测质量、质心和惯量替换估计，再加入传感器估计、驱动时延和电机转速/热限制。地形行驶、真实自碰撞及跌倒恢复需要额外控制和测试。

关闭重力只能隔离几何和控制问题；不能把零重力结果当作承载能力。MuJoCo 的软约束残差也不是连杆真实弹性变形，需要与有限元、实物载荷与温升试验区分。

## 常见问题

`launch_passive requires mjpython`：图形命令改用激活环境中的 `mjpython`。

`mjpython: command not found`：检查 `source .venv/bin/activate`、`python -m pip show mujoco`、`which python`、`which mjpython`。

Apple Silicon 架构错误：检查 `python -c "import platform; print(platform.machine())"`，应为 arm64。

`unrecognized attribute site1`：先检查 `python -c "import mujoco; print(mujoco.__version__)"`，确认实际使用本工程虚拟环境。

模型突然抖动或散开：先跑 `tools/check_model.py`。检查尺寸、C 两端初始位置、被动关节是否误驱动、限位和力矩饱和；然后考虑减小步长。不要只盲目加大增益或约束刚度。

模型下垂：重力下纯 PD 有稳态误差；检查目标可达性、关节零位和饱和，再比较 `--bias-comp`。参数变化后仍需重新验证。

固定台架里轮子转、机器人不前进：这是预期现象；底座被固定，轮子悬空，不能测试行驶或平衡。

## 官方参考

- Python 安装与 macOS mjpython：https://mujoco.readthedocs.io/en/3.2.7/python.html
- 本例固定版本及平台安装包：https://pypi.org/project/mujoco/3.2.7/
- 闭环点连接：https://mujoco.readthedocs.io/en/3.2.7/XMLreference.html#equality-connect
- 动力学、执行器和约束：https://mujoco.readthedocs.io/en/3.2.7/computation/index.html
- 约束参数：https://mujoco.readthedocs.io/en/3.2.7/modeling.html#solver-parameters
- Homebrew Python 3.12：https://formulae.brew.sh/formula/python@3.12
