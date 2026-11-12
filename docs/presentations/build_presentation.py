"""Build the C01 technical briefing as editable PPTX and matching previews.

Dependencies are isolated in /tmp/c01_presentation_deps for this task.
All technical figures are code-native diagrams or plots of official sample IQ.
"""
from pathlib import Path
import sys
sys.path.insert(0, '/tmp/c01_presentation_deps')
import json
import math
import subprocess
from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ASSETS = OUT / 'assets'
PREVIEWS = OUT / 'previews'
for folder in [ASSETS, PREVIEWS]:
    folder.mkdir(parents=True, exist_ok=True)
W, H = 1600, 900
SCALE = 0.6  # design px to PowerPoint points, 13 1/3 by 7.5 inches
REG = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
BOLD = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
C = dict(navy='#13283F', teal='#009D92', cyan='#4AD4D1', blue='#4779DE',
         orange='#D78A26', bg='#F3F6FA', white='#FFFFFF', ink='#183047',
         muted='#61748A', line='#DCE5EF', pale='#E4F4F1', lightblue='#E9F0FC',
         lightorange='#FFF1DE', darkmuted='#ACC0D1', red='#BD5C4B')
AUDIT = json.loads((ROOT/'artifacts/c01_data_audit/data_audit.json').read_text())
PARAM = json.loads((ROOT/'artifacts/main_config_verified/parameter_report.json').read_text())
STATUS = json.loads((ROOT/'artifacts/config_v2_pipeline/smoke_status.json').read_text())
assert STATUS['complete'] and len(STATUS['stages']) == 17
assert all(s['returncode'] == 0 for s in STATUS['stages'])
assert AUDIT['rows'] == 1179000
SLIDES = []
FONT_CACHE = {}


def font(size, bold=False):
    key = (round(size), bold)
    if key not in FONT_CACHE:
        FONT_CACHE[key] = ImageFont.truetype(BOLD if bold else REG, round(size), index=2)
    return FONT_CACHE[key]


def measure(text, size, bold=False):
    return font(size, bold).getlength(text)


def wrap(text, width, size, bold=False):
    result = []
    for paragraph in text.split('\n'):
        line = ''
        for ch in paragraph:
            if line and measure(line+ch, size, bold) > width:
                result.append(line)
                line = ch
            else:
                line += ch
        result.append(line)
    return result


class Slide:
    def __init__(self, section, title, subtitle='', dark=False, source='', notes='', appendix=False):
        self.title, self.notes, self.source = title, notes, source
        self.dark, self.appendix = dark, appendix
        self.elements = []
        self.number = len(SLIDES)+1
        self.rect(0, 0, W, H, C['navy'] if dark else C['bg'])
        if section:
            self.text(80, 48, 1400, 34, section, 19, C['cyan'] if dark else C['teal'], True)
        if title:
            self.text(80, 103, 1440, 77, title, 51, C['white'] if dark else C['ink'], True)
        if subtitle:
            self.text(80, 187, 1440, 55, subtitle, 25, C['darkmuted'] if dark else C['muted'])
        self.line(80, 831, 1520, 831, '#355064' if dark else C['line'], 1)
        self.text(80, 848, 1370, 30, source or 'C01 · 电磁信号通用感知 · 技术路线与阶段进展', 17,
                  C['darkmuted'] if dark else C['muted'])
        self.text(1450, 844, 70, 36, f'{self.number:02d}', 22,
                  C['cyan'] if dark else C['teal'], True, align='right')
        SLIDES.append(self)

    def rect(self, x, y, w, h, fill, border=None, radius=0):
        self.elements.append(dict(kind='rect', x=x, y=y, w=w, h=h, fill=fill, border=border, radius=radius))

    def circle(self, x, y, diameter, fill, border=None):
        self.elements.append(dict(kind='circle', x=x, y=y, w=diameter, h=diameter, fill=fill, border=border))

    def text(self, x, y, w, h, text, size=30, color=None, bold=False, align='left', link=None):
        lines = wrap(str(text), w, size, bold)
        need = len(lines)*size*1.28
        if need > h+2:
            raise ValueError(f'Slide {self.number} text overflow {need:.1f}>{h}: {text}')
        self.elements.append(dict(kind='text', x=x, y=y, w=w, h=h, text='\n'.join(lines),
                                  size=size, color=color or C['ink'], bold=bold, align=align, link=link))

    def line(self, x1, y1, x2, y2, color=None, width=3):
        self.elements.append(dict(kind='line', x=x1, y=y1, x2=x2, y2=y2, color=color or C['teal'], width=width))

    def arrow(self, x1, y1, x2, y2, color=None, width=3):
        self.line(x1, y1, x2, y2, color, width)
        self.elements.append(dict(kind='arrowhead', x=x2, y=y2, angle=math.degrees(math.atan2(y2-y1, x2-x1)),
                                  color=color or C['teal']))

    def image(self, x, y, w, h, path):
        self.elements.append(dict(kind='image', x=x, y=y, w=w, h=h, path=str(path)))

    def pill(self, x, y, text, fill=None, color=None, w=None):
        width = w or measure(text, 21, True)+30
        self.rect(x, y, width, 38, fill or C['pale'], radius=12)
        self.text(x+15, y+5, width-30, 29, text, 21, color or C['teal'], True)

    def box(self, x, y, w, h, title, body='', accent=None, fill=None, size=28):
        self.rect(x, y, w, h, fill or C['white'], C['line'], 16)
        if accent:
            self.rect(x, y+18, 5, h-36, accent, radius=2)
        self.text(x+26, y+24, w-52, 52, title, 30, accent or C['ink'], True)
        if body:
            self.text(x+26, y+87, w-52, h-105, body, size, C['muted'])

    def takeaway(self, text):
        self.rect(80, 764, 1440, 48, C['pale'], radius=8)
        self.text(100, 772, 1400, 34, text, 23, C['teal'], True)


def grid_table(s, x, y, widths, rows, row_h=65, sizes=None):
    for r, row in enumerate(rows):
        xx = x
        for col, (width, value) in enumerate(zip(widths, row)):
            fill = C['navy'] if r == 0 else C['white'] if r % 2 else '#EAF0F6'
            s.rect(xx, y+r*row_h, width, row_h, fill)
            size = (sizes or [24]*len(widths))[col]
            s.text(xx+16, y+r*row_h+14, width-32, row_h-18, value, size,
                   C['white'] if r == 0 else C['ink'], r == 0)
            xx += width


def stage(s, x, y, w, h, num, title, body, active=True):
    s.rect(x, y, w, h, C['white'], C['line'], 14)
    s.pill(x+22, y+20, num, C['pale'] if active else C['lightorange'], C['teal'] if active else C['orange'])
    s.text(x+22, y+83, w-44, 52, title, 31, C['ink'], True)
    s.text(x+22, y+150, w-44, h-167, body, 26, C['muted'])


def build_signal_plot():
    sample_json = ASSETS/'official_radar_sample.json'
    if not sample_json.exists():
        script = """import json,pyarrow.parquet as pq
p='/mnt/disk1/wangxinbei_contest_dataset/C01/rf_signal_sample_io_v1.0/rf_signal_sample_io_input_v1.0.parquet'
print(json.dumps(pq.read_table(p).to_pylist()[1]))
"""
        result = subprocess.run(['/home/yangqilin/anaconda3/envs/py313/bin/python','-c',script],
                                check=True,capture_output=True,text=True)
        sample_json.write_text(result.stdout)
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    arr = np.asarray(json.loads(sample_json.read_text())['iq'], dtype=np.float32)
    z = arr[:,0]+1j*arr[:,1]
    rms = np.sqrt(np.mean(abs(z)**2))
    z = z/max(float(rms), 1e-12)
    windows = np.lib.stride_tricks.sliding_window_view(z,64)[::16]
    spec = abs(np.fft.fftshift(np.fft.fft(windows*np.hanning(64),axis=1),axes=1))**2
    db = 10*np.log10(np.maximum(spec,1e-12)/max(float(spec.max()),1e-12))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.edgecolor':'#CCD7E3','axes.labelcolor':'#61748A',
                         'xtick.color':'#61748A','ytick.color':'#61748A'})
    fig, axes = plt.subplots(2,1,figsize=(8.6,5.3),dpi=180,gridspec_kw={'height_ratios':[1,1.25]})
    fig.patch.set_facecolor('white')
    axes[0].plot(np.arange(len(z)), z.real, color=C['teal'],lw=1.0,label='I / RMS')
    axes[0].plot(np.arange(len(z)), z.imag, color=C['blue'],lw=.8,alpha=.7,label='Q / RMS')
    axes[0].set_xlim(0,len(z)); axes[0].set_ylabel('Amplitude');axes[0].legend(loc='upper right',frameon=False,ncol=2,fontsize=10)
    axes[1].imshow(db.T,origin='lower',aspect='auto',extent=[31.5,len(z)-32.5,-.5,.5],
                   vmin=-60,vmax=0,cmap='viridis')
    axes[1].set_xlabel('Sample index');axes[1].set_ylabel('Cycles / sample')
    fig.subplots_adjust(left=.12,right=.985,top=.97,bottom=.12,hspace=.32)
    fig.savefig(ASSETS/'radar_iq_stft.png',facecolor='white')
    plt.close(fig)


def compose():
    # 01
    s = Slide('C01 / TECHNICAL BRIEFING', '', dark=True,
              source='2026.11.10  ·  技术路线与阶段进展  ·  双 A100 80GB',
              notes='开场说明：本工作面向 C01 多类型电磁信号通用感知，目标是形成一个共享电磁基座，同时完成体制识别、调制与波形识别和参数估计。当前已经完成全量数据审计、模型与训练代码、小样本完整流程测试。下面重点汇报路线选择和验证安排，全量长期训练与比赛精度仍待验证。')
    s.text(80,177,1040,116,'电磁信号通用感知',76,C['white'],True)
    s.text(80,285,1120,110,'从时频融合到结构化预训练',55,C['white'],True)
    s.text(84,423,900,104,'一个共享基座，连接体制识别、调制识别与物理参数估计',32,C['darkmuted'])
    for i,(num,label) in enumerate([('8','体制类别'),('23','调制与波形'),('7','物理参数')]):
        x=84+i*280
        s.text(x,596,210,90,num,65,C['cyan'],True)
        s.text(x,691,240,42,label,27,C['darkmuted'])
    for i in range(5):
        s.rect(1160+i*16,230+i*47,245,120,'#1C3C55','#416076',14)
    s.text(1240,438,160,50,'I / Q',35,C['cyan'],True)
    s.pill(1134,666,'数据已审计 · 流程已打通','#23483F',C['cyan'],360)
    # 02
    s=Slide('01 / OVERVIEW','工作主线：统一表征、结构学习、多任务适配',
            '以可复现的实验闭环，逐步建立比赛模型能力。',
            source='依据：技术路线文档、当前仓库与初步测试报告。',
            notes='这页先给出完整工作图景。第一步用全量审计明确数据分布和分组边界；第二步构建时频双流共享编码器；第三步比较连续重建和结构化预训练；第四步共同微调、评估并导出。当前数据和工程流程已经完成基础验证，后续重点是全量训练和各个模块的收益证据。')
    contents=[('01','可信数据','内容哈希分组\n有效位与量纲核验\n训练侧拟合统计'),('02','共享表征','时域 / 频域双流\n样点位置对齐\n动态门控融合'),('03','结构预训练','Atomic + Emergent\n两轮词表优化\n连续信息辅助'),('04','任务与交付','三任务共同微调\n逐项指标与消融\n单模型统一推理')]
    for i,(n,t,b) in enumerate(contents):
        stage(s,80+i*365,290,345,360,n,t,b)
        if i<3:s.arrow(431+i*365,470,440+i*365,470)
    s.takeaway('研究问题：怎样学习可迁移的信号结构，同时保留物理参数所需的精细信息？')
    # 03
    s=Slide('02 / TASK','评分决定重点：参数估计与两项识别同步推进',
            '初赛三项权重为 30 / 30 / 40；所有任务共用同一基座。',
            source='来源：C01 赛题 PDF，第 1–2、7–9 页。',
            notes='初赛体制和调制各占30分，参数项占40分，所以不能先只优化分类再补回归。比赛要求总参数不超过1B，跨任务共享基座权重超过80%，最终不采用多个任务基座或多模型预测集成。复赛还考察推理效率和专家评审，因此当前结构也保留标准算子与统一部署路径。')
    for i,(num,title,detail,color) in enumerate([('30','体制识别','8 类 · Accuracy',C['blue']),('30','调制 / 波形','23 类 · Accuracy',C['teal']),('40','参数估计','7 项 · 有效标签误差',C['orange'])]):
        x=80+i*490
        s.rect(x,278,460,330,C['white'],C['line'],16)
        s.text(x+28,300,390,110,num,78,color,True)
        s.text(x+28,433,390,50,title,34,C['ink'],True)
        s.text(x+28,510,390,44,detail,26,C['muted'])
    for x,text in [(80,'参数上限 ≤ 1B'),(570,'共享权重 > 80%'),(1060,'单模型 · 单推理入口')]:
        s.pill(x,655,text,C['lightblue'],C['blue'],460)
    s.takeaway('以两项识别准确率 + 七项原单位误差共同选模；本地代理分与官方成绩分开报告。')
    # 04
    s=Slide('03 / DATA','已完成 117.9 万条训练记录的全量核验',
            '三份 Parquet 全部读取；数据分布直接影响抽样和输出头设计。',
            source='来源：artifacts/c01_data_audit/data_audit.json，全量审计。',
            notes='全量数据为1179000条，调制207000，体制144000，参数828000。两类分类内部均衡，调制每类9000，体制每类18000，但任务之间不均衡，参数占70.2%。因此联合训练采用任务均衡更新预算，并在任务内按长度组织批次。没有发现非有限或异常长度I/Q。')
    for i,(name,key,color) in enumerate([('调制 / 波形','modulation',C['blue']),('体制识别','protocol',C['teal']),('物理参数','parameter',C['orange'])]):
        count=AUDIT['task_counts'][key]; y=320+i*128
        s.text(90,y,235,45,name,29,C['ink'],True)
        s.rect(340,y+8,600,30,'#E2EAF2',radius=10)
        s.rect(340,y+8,600*count/828000,30,color,radius=10)
        s.text(970,y,220,45,f'{count:,}',30,color,True)
    s.box(1230,283,290,180,'512 / 1024','采样点数',C['teal'])
    s.box(1230,486,290,180,'[T, 2]','Float32 I / Q',C['blue'])
    s.text(90,705,1100,42,'分类内部均衡；参数任务占全部记录的 70.2%。',28,C['muted'])
    # 05
    s=Slide('04 / DATA DISCOVERY','关键发现：相同波形跨任务出现，划分必须先锁组',
            '同一段 I/Q 可拥有不同任务标签；逐行随机划分会造成验证泄漏。',
            source='来源：全量 Float32 内容哈希审计；重复记录索引 duplicate_rows.parquet。',
            notes='sample_id全部唯一，但内容哈希发现13129个跨任务重复组，共26305条记录。同组必须进入同一侧，预训练同样不能接触验证波形。在训练侧，这些记录也提供了可靠的跨任务配对监督，可比较合并标签和保留原记录两种方式。精确内容分组尚不能排除加噪、缩放或切窗的同源副本。')
    s.text(80,280,600,120,'13,129',88,C['teal'],True)
    s.text(85,406,540,90,'跨任务重复波形组\n涉及 26,305 条记录',32,C['ink'])
    labels=[('调制 + 参数','10,851'),('体制 + 参数','2,080'),('含其他组合','198')]
    for i,(a,b) in enumerate(labels):
        y=555+i*51;s.text(86,y,310,42,a,25,C['muted']);s.text(410,y,170,42,b,29,C['ink'],True,align='right')
    s.box(700,280,345,150,'同内容波形组','不同 ID / 不同任务标签',C['blue'],size=24)
    s.arrow(1045,355,1110,355)
    s.box(1120,280,395,150,'内容哈希分组','完整保留组内记录',C['teal'],size=24)
    s.line(1320,430,1320,470,C['teal']);s.line(910,470,1320,470,C['teal'])
    s.arrow(910,470,910,520);s.arrow(1320,470,1320,520)
    s.box(700,530,360,150,'训练侧','统计 / 词表 / 参数学习',C['teal'],size=24)
    s.box(1110,530,405,150,'验证侧','固定清单，独立选模',C['blue'],size=24)
    s.takeaway('先按内容分组，再平衡任务与类别；训练侧配对标签合并作为独立实验。')
    # 06
    s=Slide('05 / RESEARCH BASIS','研究切入点：观测条件会改变波形外观',
            '“恢复得像”与“结构可迁移”需要通过下游任务分别检验。',
            source='依据：三场讲座；Observations Drift, Structures Remain 用户提供的成果介绍。',
            notes='成果介绍指出，波形由生成、传播、采集共同决定，重建目标可能学习观测条件相关线索。对C01我们将这一思想转化为实验：比较原始数值重建和结构身份预测。讲座1的长序列方法按实际需求后置，讲座2的结构学习作为关键候选，讲座3的统一基座、相位和掩码先验各自做消融。研究介绍的分类结果不能直接外推为参数回归收益。')
    s.box(80,280,450,340,'波形形成的三个环节','信号生成：局部与组合规律\n\n环境传播：信道与噪声\n\n信号采集：采样坐标',C['blue'],size=26)
    s.box(575,280,450,340,'两个需要处理的失配','表观失配\n相同结构，观测外观变化\n\n时间失配\n相同点数，物理时长不同',C['orange'],size=26)
    s.box(1070,280,450,340,'对应的项目方向','Atomic / Emergent\n学习局部与组合结构\n\n样点对齐 / RoPE-fs\n分别处理当前与已知采样率',C['teal'],size=26)
    s.takeaway('采用条件：结构目标改善分类时，七项参数估计仍需单独验证。')
    # 07
    s=Slide('06 / MODEL','总体架构：时频双流融合，一个基座服务三个任务',
            '前端、融合层与 Transformer 全部共享；输出头保持轻量。',
            source='依据：code/models/rf_model.py；主配置参数报告（真实数据三步短测实例）。',
            notes='模型接收512或1024点I/Q，保留共用归一化和能量尺度。时域支路处理波形、包络、差分相位，频域支路处理双边局部谱与全窗频谱。按样点位置对齐后通过门控融合，再进入8层384宽共享Transformer。分类使用全局读出，参数使用全局和局部特征。主配置短测实例实测17081539参数，共享97.20%，完整训练取值表不同会使头部规模略变。')
    s.box(80,405,180,150,'I / Q','512 / 1024',C['ink'],size=23)
    s.box(315,290,265,145,'时域流','波形 · 包络 · 相位',C['teal'],size=23)
    s.box(315,525,265,145,'频域流','局部 STFT · 全窗谱',C['blue'],size=23)
    s.line(282,480,282,360,C['teal']);s.line(282,480,282,600,C['blue']);s.line(260,480,282,480)
    s.arrow(282,360,310,360);s.arrow(282,600,310,600,C['blue'])
    s.box(645,400,210,160,'对齐与融合','动态通道门控',C['teal'],size=23)
    s.line(580,360,610,360);s.line(610,360,610,480);s.arrow(610,480,640,480)
    s.line(580,600,610,600,C['blue']);s.line(610,600,610,480,C['blue'])
    s.box(915,375,255,210,'共享编码器','Transformer\n8 层 · 384 宽\n6 个注意力头',C['ink'],size=25)
    s.arrow(855,480,910,480)
    for y,title,body,color in [(280,'体制头','8 类',C['blue']),(450,'调制头','23 类',C['teal']),(620,'参数头','7 个数值',C['orange'])]:
        s.rect(1250,y,270,125,C['white'],C['line'],12)
        s.text(1270,y+15,230,45,title,29,color,True);s.text(1270,y+65,230,40,body,25,C['muted'])
        s.line(1200,480,1200,y+62);s.arrow(1200,y+62,1245,y+62,color)
    s.line(1170,480,1200,480)
    s.pill(80,700,'17.08M 参数',C['lightblue'],C['blue'],280)
    s.pill(390,700,'97.20% 共享',C['pale'],C['teal'],280)
    s.text(710,705,800,40,'主配置短测实例；全量网格头规模以训练报告为准。',23,C['muted'])
    # 08
    s=Slide('07 / TIME–FREQUENCY','两路表示在同一时间位置汇合',
            '时域看局部变化，频域看能量分布；用样点中心建立对应关系。',
            source='图示：官方 Radar/Barker Code 接口样例；STFT 为项目绘制，非模型预测。',
            notes='左图来自官方六条接口样例中的雷达波形，上面是共同RMS归一化的I和Q，下面是双边STFT，相对功率以颜色展示。这只是输入表示示例。时域前端总步长16，频域64点窗和16点步长，融合时按中心样点对齐。幅度归一化保留原尺度旁路，相位特征降低低幅位置权重；绝对频谱位置不自动居中消除。')
    s.rect(80,270,815,440,C['white'],C['line'],14)
    s.image(96,282,780,415,ASSETS/'radar_iq_stft.png')
    s.pill(110,720,'真实输入示例',C['lightblue'],C['blue'])
    s.box(945,270,575,140,'时域：保留精细变化','包络边缘、相位增量、脉冲起止',C['teal'],size=25)
    s.box(945,435,575,140,'频域：保留谱形与位置','双边频谱、占用带宽、扫频轨迹',C['blue'],size=25)
    s.box(945,600,575,140,'融合：按中心样点对齐','每 16 点形成 token；动态门控汇合',C['ink'],size=25)
    # 09
    s=Slide('08 / PARAMETER HEADS','参数任务：离散网格与连续时间量分别建模',
            '参数头同时读取全局语义与带位置的局部特征。',
            source='来源：全量参数统计；code/models/rf_model.py；当前方案中的输出头对照。',
            notes='通信参数训练取值有限，码元率6值、带宽21值、频移13值，所以主配置实现了概率期望加残差，同时保留连续回归配置作为对照。雷达脉宽、PRI、时延连续回归，脉冲数比较计数期望和连续回归。所有字段按训练侧中心尺度处理，使用valid选取有效标签。频移有15131个真零必须计入。隐藏测试是否沿用网格未知，所以不默认硬量化。')
    s.box(80,278,700,390,'通信三参数','码元率 6 值  ·  带宽 21 值  ·  频移 13 值\n\n候选：连续回归 / 网格概率期望 + 残差\n\n数值预测：ŷ = Σ pₖvₖ + δ',C['blue'],size=28)
    s.box(820,278,700,390,'雷达四参数','脉宽 / PRI / 时延：连续回归\n\n脉冲数：2–6；回归与计数期望对照\n\n局部位置读出保留起点、边沿和间隔',C['teal'],size=28)
    for i,txt in enumerate(['按 *_valid 选样本','每个字段单独标准化','输出恢复原始数值单位']):
        s.pill(80+i*490,695,txt,C['lightorange'],C['orange'],460)
    s.takeaway('频移有 15,131 个有效真零；缺失值与不适用的占位零必须区分。')
    # 10
    s=Slide('09 / PRETRAINING','关键对照：补齐连续数值，还是预测结构身份？',
            '使用同一共享架构与可见上下文，比较预训练目标的迁移价值。',
            source='依据：用户提供的 Structural Pretraining with Time Alignment 成果介绍。',
            notes='连续重建让网络补原始I/Q数值，结构预测让网络猜被遮挡位置的局部及组合模式编号。两者需要保持骨干、可见输入、掩码预算和下游微调一致，才能解释收益来源。我们也设置结构加连续辅助目标，目的是照顾C01的参数回归；这是参赛适配，不是原工作已经证明的组合。')
    for x,title,color,desc in [(80,'连续重建目标',C['blue'],'从上下文恢复被遮挡的 I/Q'),(820,'结构身份目标',C['teal'],'从上下文预测 Atomic / Emergent')]:
        s.rect(x,278,700,370,C['white'],C['line'],16)
        s.text(x+28,303,644,60,title,34,color,True)
        for i,l in enumerate(['可见','MASK','MASK','可见']):
            xx=x+30+i*159;s.rect(xx,398,143,65,C['lightblue'] if l!='MASK' else '#E6EAF0',radius=7)
            s.text(xx+8,414,127,38,l,25,C['muted'] if l=='MASK' else color,True,align='center')
        s.arrow(x+350,483,x+350,526,color)
        s.text(x+30,550,640,60,desc,29,C['ink'],True,align='center')
    s.takeaway('比赛选型：两项分类收益与七项参数误差一起判断，重建损失只作训练诊断。')
    # 11
    s=Slide('10 / STRUCTURAL UNITS','Atomic 描述局部模式，Emergent 描述组合规律',
            '结构标签由训练数据生成；编号本身不等同于真实物理符号。',
            source='来源：用户提供的成果介绍；讲座 2 原图 07–08；项目结构词表实现。',
            notes='原介绍使用512个原子单元，再进行512次合并，形成包含原子及组合单元的1024目标词表，最长展开5个原子位置。右侧只是示意：A2A3A4合并为E1后，把E1回填到这三个原位置，网络仍处理相同token序列。实际词表需要诊断占用率、幅度和长度偏差，并检查弱脉冲覆盖。')
    s.rect(80,278,580,445,C['white'],C['line'],16)
    s.text(110,302,520,52,'局部特征聚类 → Atomic',31,C['teal'],True)
    for cx,cy,color,letter in [(235,442,C['teal'],'A1'),(467,445,C['blue'],'A2'),(348,582,C['orange'],'A3')]:
        s.circle(cx-65,cy-65,130,{'A1':C['pale'],'A2':C['lightblue'],'A3':C['lightorange']}[letter])
        for j in range(8):
            ang=j*2.4;s.circle(cx+math.cos(ang)*38-4,cy+math.sin(ang)*31-4,8,color)
        s.text(cx-38,cy-19,76,42,letter,29,color,True,align='center')
    s.text(110,655,520,44,'示意聚类；参考词表为 512 个单元。',23,C['muted'])
    s.rect(700,278,820,445,C['white'],C['line'],16)
    s.text(730,302,760,52,'相邻编号合并 → Emergent',31,C['blue'],True)
    for y,title,seq in [(390,'原子标签',['A1','A2','A3','A4','A5','A6']),(555,'回填标签',['A1','E1','E1','E1','A5','A6'])]:
        s.text(730,y,160,42,title,25,C['muted'])
        for i,val in enumerate(seq):
            x=890+i*96;col=C['teal'] if val=='E1' else C['blue']
            s.rect(x,y-3,82,60,C['pale'] if val=='E1' else C['lightblue'],radius=7)
            s.text(x+6,y+8,70,40,val,27,col,True,align='center')
    s.text(890,480,570,44,'A2 + A3 + A4 → E1（组合示意）',25,C['teal'],True)
    s.text(730,655,740,44,'512 次合并 · 1024 个目标 · 最长 5 个原子位置',23,C['muted'])
    s.takeaway('组合标签回填到原位置，学习跨 Patch 规律；下游仍输出比赛类别和参数。')
    # 12
    s=Slide('11 / ITERATIVE CODEBOOK','两轮词表优化：让结构目标随表征能力改进',
            '词表在每轮内固定；第二轮用冻结编码器提取训练侧特征。',
            source='依据：成果介绍的两轮策略；code/training/build_codebook.py 与短测日志。',
            notes='第一轮从原始Patch局部特征构建词表，训练结构预测编码器。第二轮冻结第一轮编码器，提取训练侧局部表示并重建词表，再继续结构预训练。项目中第二轮共享主干继承第一轮，结构头重新初始化，因为新旧编号未必对应。所有词表只在训练侧拟合，教师推理和聚类也计入计算预算。参考规模词表与三步结构训练已经执行，但收益还需全量实验。')
    for y,label,items in [(290,'第一轮',['原始 Patch 特征','词表 V1','结构预训练']), (520,'第二轮',['冻结编码器特征','词表 V2','结构继续训练'])]:
        s.pill(80,y+26,label,C['pale'],C['teal'],145)
        for i,item in enumerate(items):
            x=280+i*420
            s.rect(x,y,370,140,C['white'],C['line'],14)
            s.text(x+22,y+40,326,64,item,31,C['ink'],True,align='center')
            if i<2:s.arrow(x+375,y+70,x+414,y+70)
    s.line(1305,430,1305,475,C['teal']);s.line(1305,475,465,475,C['teal']);s.arrow(465,475,465,514)
    s.takeaway('已生成 512 / 1024 参考规模词表并完成结构短测；两轮增益需与同预算单轮方案比较。')
    # 13
    s=Slide('12 / TIME ALIGNMENT','物理时间对齐：有采样率时，位置应表达实际时间',
            '时频分支的样点对齐现在可做；跨采样率的物理对齐需要 fs。',
            source='依据：讲座 2、用户提供的 RoPE-fs 介绍；C01 官方输入 Schema。',
            notes='这页区分两个问题。当前时域和STFT按中心样点对齐，不需要知道采样率。而RoPE-fs需要采样率，把token中心映射到物理时间。示例中16个点在1MHz对应16微秒，在2MHz对应8微秒。C01输入没有fs，所以目前使用样点坐标，不由512或1024长度猜采样率。后续若赛方提供fs，再比较索引、索引加条件和物理时间RoPE。')
    s.rect(80,278,890,430,C['white'],C['line'],16)
    for y,label,spacing,color in [(360,'1 MHz：16 点 = 16 μs',110,C['teal']),(535,'2 MHz：16 点 = 8 μs',55,C['blue'])]:
        s.text(110,y-42,750,45,label,29,color,True)
        for i in range(6):
            xx=145+i*spacing;s.circle(xx,y+28,15,color)
            if i<5:s.line(xx+15,y+35,xx+spacing,y+35,color,3)
        s.text(110,y+78,780,40,'相同 token 间隔，对应不同物理时长。',24,C['muted'])
    s.box(1020,278,500,195,'物理坐标候选','tᵢ = [nᵢ + (P−1)/2] / fs\n用于旋转位置编码',C['teal'],size=26)
    s.box(1020,500,500,208,'当前 C01 实现','输入缺少采样率\n使用样点坐标；fs 明确后再验证',C['orange'],size=26)
    s.takeaway('序列长度不等于采样率；保持窗口起点与局部位置，支持时延及脉冲参数估计。')
    # 14
    s=Slide('13 / TRAINING PIPELINE','训练闭环：建立基线，再证明结构学习带来的收益',
            '从相同数据划分出发，比较目标、迭代和任务协同。',
            source='依据：当前训练代码、配置与项目实验方案。',
            notes='先运行联合监督建立可提交基线，再在相同共享结构上比较重建与结构预训练，最后共同微调并统一导出。任务损失独立归一，参数按有效位和训练侧尺度计算。预训练先在原始I/Q施加掩码，再生成全部时域和频域特征，完整I/Q只进入目标分支。这样避免频谱或归一化泄露遮挡值。')
    for i,(n,t,b) in enumerate([('01','联合监督','建立三任务基线\n定位主要误差'),('02','预训练对照','重建 / Atomic / 多层结构\n验证第二轮词表'),('03','共同微调','共享主干 + 三类任务头\n检查参数与分类协同'),('04','单模型导出','移除训练辅助头\n统一 JSONL 推理')]):
        stage(s,80+i*365,280,345,320,n,t,b)
    for i,txt in enumerate(['划分与词表：训练侧拟合','输入与目标：掩码链路隔离','任务与字段：损失分别归一']):
        s.box(80+i*490,640,460,106,txt,'',C['teal'])
    # 15
    s=Slide('14 / EXPERIMENTS','用小矩阵区分收益来源，用完整指标决定采用',
            '固定骨干、划分、可见输入与微调协议；重要差异重复训练。',
            source='依据：结构化预训练适配方案；当前本地评估器。',
            notes='实验依次覆盖无预训练监督、重建、Atomic、Atomic加Emergent、第二轮词表、结构连续混合。另将融合、相位和配对合并作为独立对照。每次同时看体制和调制准确率、宏平均召回、七项原单位RMSE和误差尾部，并记录GPU小时与时延。官方参数尺度和聚合未知，本地代理分只是选模辅助。')
    grid_table(s,80,270,[150,430,860],[['编号','方案','主要回答的问题'],['T0','联合监督','当前架构的比赛基线是什么？'],['T1','I/Q 掩码重建','连续重建有多少下游价值？'],['T2 / T3','Atomic / + Emergent','局部结构与组合规律各贡献多少？'],['T4','第二轮词表优化','词表迭代是否优于同预算单轮？'],['T5','结构 + 连续辅助','能否兼顾分类与七项参数估计？']],row_h=69,sizes=[25,25,25])
    s.takeaway('评估面板：2 项分类 + 7 项有效 RMSE / MAE / P95 + 训练成本与端到端时延。')
    # 16
    s=Slide('15 / ENGINEERING PROGRESS','阶段进展：完整流程已跑通，进入全量实验阶段',
            '现有证据覆盖可执行性、关键数值语义与统一推理。',
            source='来源：docs/self_test_report.md；docs/config_v2_test_report.md；机器可读测试状态。',
            notes='仓库当前已有模型、数据准备、两类预训练、词表更新、微调、评估、导出和统一推理。592条小子集的17阶段完整流程通过，包含GPU/CPU和同组双卡。主配置另完成三步监督，参考规模结构词表和三步预训练也通过。当前导出权重属于短测，不能作为比赛精度成果。下一步是全量长期收敛、消融和部署验收。')
    for i,(big,label) in enumerate([('17','全流程阶段通过'),('592','小规模测试记录'),('97.20%','主配置共享比例')]):
        x=80+i*490;s.rect(x,274,460,180,C['white'],C['line'],14)
        s.text(x+25,290,410,90,big,65,C['teal'],True)
        s.text(x+25,389,410,40,label,25,C['muted'])
    s.box(80,495,700,225,'已完成','数据审计与分组逻辑 · 模型与训练实现\n两轮词表与联合微调短测\nCPU / GPU 推理 · 双卡同步 · 格式校验',C['teal'],size=26)
    s.box(820,495,700,225,'下一阶段','全量监督与预训练对照 · 多次独立复测\n隐藏测试泛化与实际技术收益\n离线镜像 · CUDA 12.x / 国产平台验收',C['orange'],size=26)
    s.takeaway('当前是小样本短测与参数统计；全量收敛、比赛精度及目标平台性能待验证。')
    # 17
    s=Slide('16 / EXECUTION PLAN','双 A100 支撑对照实验，先形成可提交版本',
            '常规消融每卡一个实验；主模型使用双卡同步训练。',
            source='计划依据：2 × A100 80GB；按 12 月底提交安排，正式日期以赛事通知为准。',
            notes='资源采用两张A10080GB。先用6层256宽小模型筛选目标，再把有效方案扩展到8层384宽主模型，扩大模型需要有实测收益。按工作窗口安排，11月完成全量基线和主要预训练对照，12月上旬补词表迭代与参数专项，12月中旬准备完整可交付包，余下时间复测和优化。先实测数据加载、STFT和前向反向吞吐，再锁定总预算。')
    for i,(title,body,color) in enumerate([('S：快速消融','6 层 · 256 宽\n筛选目标与输入表示',C['blue']),('M：主模型','8 层 · 384 宽\n全量训练与主要对照',C['teal']),('L：按收益扩展','12 层 · 512 宽\n性能 / 成本实测后决定',C['orange'])]):
        s.box(80+i*490,275,460,215,title,body,color,size=26)
    milestones=[('11 月中下旬','全量基线与目标对照'),('12 月上旬','词表迭代与参数优化'),('12 月中旬','完整提交包与报告'),('截止前','冻结、复测、部署')]
    s.line(160,565,1440,565,C['line'],7)
    for i,(date,txt) in enumerate(milestones):
        x=130+i*365;s.circle(x+12,552,26,C['teal']);s.text(x-30,610,325,46,date,28,C['ink'],True)
        s.text(x-30,670,325,75,txt,25,C['muted'])
    s.takeaway('提交物：一个共享模型 + 统一推理入口 + 配置与日志 + 离线环境 + 技术报告。')
    # 18
    s=Slide('17 / DISCUSSION','一个共享基座，贯通结构学习与多任务感知',
            '研究贡献由消融结果支撑，工程能力由可复现交付验证。',dark=True,
            source='当前工作：数据审计、代码与短测已完成；重点推进全量训练和对照验证。',
            notes='收束为三点：第一，按样点对齐的时频共享表征；第二，Atomic与Emergent结构学习及两轮目标优化；第三，面向分类和连续参数的共同适配。已经具备运行全量实验的工程基础，下一阶段用相同验证协议回答各模块是否有效。欢迎围绕结构目标、参数信息保留和部署成本讨论。')
    for y,num,title,body in [(305,'01','时频对齐','联合波形与频谱，形成跨任务共享表征'),(455,'02','结构学习','从局部单元到组合规律，验证词表迭代价值'),(605,'03','共同适配','同时评估类别判断与七项物理参数估计')]:
        s.text(85,y,120,65,num,45,C['cyan'],True)
        s.text(235,y,300,60,title,37,C['white'],True)
        s.text(560,y+5,920,65,body,31,C['darkmuted'])
    # 19 appendix
    s=Slide('APPENDIX A / TARGETS','附录：七项参数的有效值与输出策略',
            '范围使用文件原始数值；采样率与详细单位定义仍需赛方确认。',
            source='来源：全量审计 data_audit.json。每项有效样本均为 414,000 条。',appendix=True,
            notes='这页用于问答。通信参数为离散训练网格，但测试是否同网格未知，因此保留回归和残差。三个时间参数连续，采用训练侧尺度归一和原单位评价。每项有效414000。不能通过参数是否为0判断适用性，也不对所有字段做统一硬截断。')
    rows=[['字段','训练原始数值范围','取值结构','输出策略'],['symbol_rate','62.5 – 2000','6 个值','回归 / 概率期望 + 残差'],['bandwidth','65 – 3500','21 个值','回归 / 概率期望 + 残差'],['frequency_shift','−50 – 1000','13 个值，含真零','带符号回归 / 网格头'],['pulse_width','约 1.0e−5 – 1.6e−5','连续','局部特征 + 连续回归'],['pri','约 1.7e−5 – 2.3e−5','连续','间隔表征 + 连续回归'],['time_delay','约 1.0e−6 – 1.0e−5','连续','起点位置 + 连续回归'],['number_of_pulses','2 – 6','5 个计数值','连续回归 / 计数期望']]
    grid_table(s,80,260,[320,400,300,420],rows,row_h=59,sizes=[22,23,23,23])
    # 20 appendix
    s=Slide('APPENDIX B / EVIDENCE','附录：研究参考与本地证据入口',
            '外部方法、项目设计与实测进展分别标注，避免混用结论。',
            source='完整来源与讲解提示同时保存在各页备注和配套讲稿中。',appendix=True,
            notes='外部方法来源包括赛题PDF、三场讲座和用户提供的结构预训练成果介绍，以及EMind、SpectrumFM等原论文。本文不是结构论文完整复现；混合连续目标属于C01适配。实测数据来自全量审计JSON、主配置参数报告和两份初测报告。若问创新，回答需要先完成同骨干同预算消融；若问性能，当前仅报告流程验证，不给正式比赛精度。')
    refs=[('赛题与数据','C01 任务 PDF；全量审计 JSON / 样本索引',''),
          ('核心研究','Observations Drift, Structures Remain\n用户提供成果介绍 + 讲座 2 原始 PPT',''),
          ('预训练参照','EMind（2025）','https://arxiv.org/abs/2508.18785'),
          ('编码与目标参照','SpectrumFM（2025）','https://arxiv.org/abs/2505.06256'),
          ('实现与进展','README；docs/self_test_report.md\ndocs/config_v2_test_report.md','')]
    for i,(label,body,url) in enumerate(refs):
        y=272+i*93;s.text(90,y,295,43,label,25,C['teal'],True)
        s.text(420,y,1060,81,body,27,C['ink'],link=url or None)
    s.text(90,752,1400,46,'备注含逐页讲解与来源；PPT 中的模型和流程图可直接编辑。',23,C['muted'])


def reorder_for_management():
    """Keep 11 briefing pages; put nine technical/evidence pages in backup."""
    original = list(SLIDES)
    main = [0, 1, 2, 3, 4, 6, 10, 8, 15, 16, 17]
    backup = [5, 7, 9, 11, 12, 13, 14, 18, 19]
    SLIDES[:] = [original[i] for i in main + backup]
    durations = [15, 35, 35, 40, 45, 65, 65, 60, 55, 45, 25]
    for i, slide in enumerate(SLIDES):
        old_number = slide.number
        slide.number = i + 1
        slide.appendix = i >= len(main)
        if not slide.appendix:
            slide.notes = f'建议用时：{durations[i]} 秒。\n\n' + slide.notes
        else:
            slide.notes = '备查页：根据提问选择展开，不计入 8–10 分钟主讲。\n\n' + slide.notes
        for element in slide.elements:
            if element['kind'] != 'text':
                continue
            if element['x'] == 1450 and element['y'] == 844:
                element['text'] = f'{i+1:02d}'
            if element['x'] == 80 and element['y'] == 48:
                suffix = element['text'].split(' / ', 1)[-1]
                if i == 0:
                    element['text'] = 'C01 / MANAGEMENT BRIEFING'
                elif slide.appendix:
                    element['text'] = f'BACKUP {chr(65+i-len(main))} / {suffix}'
                else:
                    element['text'] = f'{i:02d} / {suffix}'


def rgb(value):
    return RGBColor.from_string(value.lstrip('#'))


def render_pptx():
    prs=Presentation()
    prs.slide_width=Inches(13.333333)
    prs.slide_height=Inches(7.5)
    prs.core_properties.title='电磁信号通用感知：技术路线与阶段进展'
    prs.core_properties.subject='C01 时频共享基座与结构化预训练'
    prs.core_properties.author=''
    prs.core_properties.keywords='C01, I/Q, 时频融合, Atomic, Emergent, 多任务学习'
    for data in SLIDES:
        slide=prs.slides.add_slide(prs.slide_layouts[6])
        for e in data.elements:
            kind=e['kind']
            if kind in ['rect','circle']:
                shape_type=MSO_SHAPE.OVAL if kind=='circle' else MSO_SHAPE.ROUNDED_RECTANGLE if e.get('radius') else MSO_SHAPE.RECTANGLE
                sh=slide.shapes.add_shape(shape_type,Pt(e['x']*SCALE),Pt(e['y']*SCALE),Pt(e['w']*SCALE),Pt(e['h']*SCALE))
                if kind=='rect' and e.get('radius'):
                    sh.adjustments[0]=min(.2,e['radius']/min(e['w'],e['h']))
                sh.fill.solid();sh.fill.fore_color.rgb=rgb(e['fill'])
                if e.get('border'):
                    sh.line.color.rgb=rgb(e['border']);sh.line.width=Pt(.7)
                else:sh.line.fill.background()
            elif kind=='text':
                sh=slide.shapes.add_textbox(Pt(e['x']*SCALE),Pt(e['y']*SCALE),Pt(e['w']*SCALE),Pt(e['h']*SCALE))
                tf=sh.text_frame;tf.clear();tf.word_wrap=False
                tf.margin_left=tf.margin_right=tf.margin_top=tf.margin_bottom=0
                tf.vertical_anchor=MSO_ANCHOR.TOP
                for i,line in enumerate(e['text'].split('\n')):
                    p=tf.paragraphs[0] if i==0 else tf.add_paragraph()
                    p.alignment={'left':PP_ALIGN.LEFT,'center':PP_ALIGN.CENTER,'right':PP_ALIGN.RIGHT}[e['align']]
                    p.space_before=p.space_after=Pt(0)
                    p.line_spacing=Pt(e['size']*SCALE*1.28)
                    r=p.add_run();r.text=line
                    r.font.name='Microsoft YaHei';r.font.size=Pt(e['size']*SCALE)
                    r.font.bold=e['bold'];r.font.color.rgb=rgb(e['color'])
                    pr=r._r.get_or_add_rPr()
                    ea=OxmlElement('a:ea');ea.set('typeface','Microsoft YaHei');pr.append(ea)
                    if e.get('link'):r.hyperlink.address=e['link']
            elif kind=='line':
                sh=slide.shapes.add_connector(1,Pt(e['x']*SCALE),Pt(e['y']*SCALE),Pt(e['x2']*SCALE),Pt(e['y2']*SCALE))
                sh.line.color.rgb=rgb(e['color']);sh.line.width=Pt(e['width']*SCALE)
            elif kind=='arrowhead':
                sh=slide.shapes.add_shape(MSO_SHAPE.ISOSCELES_TRIANGLE,Pt((e['x']-8)*SCALE),Pt((e['y']-8)*SCALE),Pt(16*SCALE),Pt(16*SCALE))
                sh.rotation=e['angle']+90;sh.fill.solid();sh.fill.fore_color.rgb=rgb(e['color']);sh.line.fill.background()
            elif kind=='image':
                slide.shapes.add_picture(e['path'],Pt(e['x']*SCALE),Pt(e['y']*SCALE),Pt(e['w']*SCALE),Pt(e['h']*SCALE))
        slide.notes_slide.notes_text_frame.text=f'{data.title or "封面"}\n\n{data.notes}\n\n来源与范围：{data.source}'
    path=OUT/'C01_电磁模型技术路线汇报.pptx'
    prs.save(path)
    return path


def render_preview(data, scale=1.5):
    im=Image.new('RGB',(round(W*scale),round(H*scale)),C['bg'])
    d=ImageDraw.Draw(im)
    for e in data.elements:
        kind=e['kind'];x,y=e['x']*scale,e['y']*scale
        if kind in ['rect','circle']:
            box=(x,y,x+e['w']*scale,y+e['h']*scale)
            if kind=='circle':d.ellipse(box,fill=e['fill'],outline=e.get('border'),width=max(1,round(scale)))
            else:d.rounded_rectangle(box,radius=e.get('radius',0)*scale,fill=e['fill'],outline=e.get('border'),width=max(1,round(scale)))
        elif kind=='text':
            f=font(e['size']*scale,e['bold'])
            for i,line in enumerate(e['text'].split('\n')):
                xx=x
                length=f.getlength(line)
                if e['align']=='center':xx+=(e['w']*scale-length)/2
                if e['align']=='right':xx+=e['w']*scale-length
                d.text((xx,y+i*e['size']*1.28*scale),line,font=f,fill=e['color'],anchor='lt')
        elif kind=='line':d.line((x,y,e['x2']*scale,e['y2']*scale),fill=e['color'],width=max(1,round(e['width']*scale)))
        elif kind=='arrowhead':
            angle=math.radians(e['angle']);back=(x-math.cos(angle)*14*scale,y-math.sin(angle)*14*scale)
            points=[(x,y),(back[0]+math.sin(angle)*7*scale,back[1]-math.cos(angle)*7*scale),
                    (back[0]-math.sin(angle)*7*scale,back[1]+math.cos(angle)*7*scale)]
            d.polygon(points,fill=e['color'])
        elif kind=='image':
            pic=Image.open(e['path']).convert('RGB').resize((round(e['w']*scale),round(e['h']*scale)),Image.Resampling.LANCZOS)
            im.paste(pic,(round(x),round(y)))
    return im


def export_assets():
    pdf=canvas.Canvas(str(OUT/'C01_电磁模型技术路线汇报_预览.pdf'),pagesize=(960,540))
    pdf.setTitle('C01 电磁模型技术路线汇报')
    thumbs=[]
    for i,s in enumerate(SLIDES,1):
        im=render_preview(s)
        path=PREVIEWS/f'slide_{i:02d}.png';im.save(path)
        pdf.drawImage(ImageReader(im),0,0,width=960,height=540);pdf.showPage()
        thumbs.append(im.resize((480,270),Image.Resampling.LANCZOS))
    pdf.save()
    sheet=Image.new('RGB',(480*4,310*5),'#DCE5EF');d=ImageDraw.Draw(sheet)
    for i,im in enumerate(thumbs):
        x=(i%4)*480;y=(i//4)*310
        sheet.paste(im,(x,y));d.text((x+12,y+274),f'{i+1:02d}  {SLIDES[i].title[:22]}',font=font(18),fill=C['ink'])
    sheet.save(OUT/'C01_汇报PPT_总览.png')
    notes=['# C01 电磁模型技术路线汇报讲稿','',
           '面向项目负责人，建议时长：8–10 分钟。第 1–11 页为主讲，第 12–20 页为备查。主讲建议总用时约 9 分钟。',
           '当前依据：2026 年 11 月 10 日仓库与审计/初测产物。全量长期训练和正式比赛精度仍待验证。','']
    for i,s in enumerate(SLIDES,1):
        notes += [f'## {i:02d}｜{s.title or "封面：电磁信号通用感知"}','',s.notes,'',f'来源：{s.source}','']
    notes += ['## 常见问答','',
              '- 为什么采用结构目标？以同架构、同可见输入比较结构预测和连续重建的下游价值。',
              '- 是否已经证明提高比赛成绩？目前证据为数据审计、流程短测和参数统计，待全量训练与重复消融。',
              '- 没有采样率怎样做 RoPE-fs？当前使用样点坐标；有可信推理采样率后再启用物理时间对齐。',
              '- 结构词表是否是外挂知识库？词表只用于训练目标生成，最终推理不加载词表或教师。',
              '- 分类提升但参数下降怎么办？比较连续辅助目标、目标权重和局部读出，以完整比赛面板选模。','']
    (OUT/'C01_逐页讲稿.md').write_text('\n'.join(notes))
    (OUT/'README.md').write_text('''# C01 技术路线汇报材料

- `C01_电磁模型技术路线汇报.pptx`：20 页，16:9，可编辑正文与架构/流程图；各页含讲者备注。
- `C01_电磁模型技术路线汇报_预览.pdf`：同版面预览，便于快速查看与分享。
- `C01_汇报PPT_总览.png`：全部幻灯片缩略图。
- `C01_逐页讲稿.md`：逐页讲解、来源与常见问答。
- `build_presentation.py`：生成源码。真实输入示例和单页预览位于 `assets/`、`previews/`。

面向项目负责人，8–10 分钟；第 1–11 页为主讲，第 12–20 页为备查。PPT 字体指定微软雅黑，预览使用 Noto Sans CJK SC。PDF 是同源版面渲染预览，PPT 中的主体文字与图形保持可编辑。

内容来自当前仓库、全量数据审计、主配置参数统计、流程初测及已提供的研究材料。参数量与共享比例为小子集三步主配置实例，未呈现为正式训练精度。未改动原始数据、模型代码或权重。

复现依赖位于 `/tmp/c01_presentation_deps`，执行 `python docs/presentations/build_presentation.py`。若临时目录已清理，需先在隔离环境安装 python-pptx、Pillow、reportlab、matplotlib，并保证中文字体可用。
''')


if __name__=='__main__':
    build_signal_plot()
    compose()
    reorder_for_management()
    path=render_pptx()
    export_assets()
    print(json.dumps({'pptx':str(path),'slides':len(SLIDES),'editable_text_and_diagrams':True},ensure_ascii=False))
