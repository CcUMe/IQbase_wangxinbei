from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
import math,json
root=Path.cwd()
# These boxes extract the photographed slides without reconstructing covered content.
boxes={
1:[(0,.30,.85,.81),(.07,.475,.92,.99),(.10,.415,.94,.95),(.07,.435,.90,.975),(.105,.465,.94,.97),(.085,.45,.925,.965),(.105,.35,.94,.85)],
2:[(.04,.42,.89,.935),(.04,.33,.875,.83),(.075,.40,.925,.91),(.025,.425,.88,.965),(.065,.325,.92,.835),(.09,.365,.91,.85),(.025,.395,.88,.905),(.06,.365,.88,.86),(.035,.385,.91,.89),(.085,.42,.915,.94)],
3:[(.16,.435,.835,.84),(.085,.345,.92,.845),(.09,.335,.925,.845),(.03,.35,.86,.85),(.035,.25,.87,.755),(.09,.40,.935,.91),(.04,.385,.89,.89),(.025,.39,.86,.89),(.075,.41,.91,.92),(.08,.33,.91,.83),(.04,.46,.87,.965),(.08,.41,.88,.91),(0,.36,.81,.855),(.02,.35,.83,.845),(.055,.34,.865,.845),(.04,.335,.855,.835),(.01,.295,.835,.80),(.07,.38,.89,.88),(0,.335,.81,.84),(.02,.315,.85,.82),(.035,.405,.87,.91)]}
for lec,rects in boxes.items():
 out=root/f'讲座{lec}'/'配图';out.mkdir(exist_ok=True)
 manifest=[]
 for n,b in enumerate(rects,1):
  src=root/f'讲座{lec}'/f'{n}.jpg';im=Image.open(src);w,h=im.size
  px=(max(0,int(b[0]*w)-12),max(0,int(b[1]*h)-12),min(w,int(b[2]*w)+12),min(h,int(b[3]*h)+12))
  crop=im.crop(px);crop.save(out/f'原图_{n:02d}.jpg',quality=96,subsampling=0)
  manifest.append({'file':f'原图_{n:02d}.jpg','source':f'../{n}.jpg','crop_pixels':px,'note':'只裁取画面，未修复遮挡或改写PPT内容'})
 (out/'图片来源.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
# Draw clean teaching diagrams with Chinese fonts, suitable for ordinary Markdown viewers.
FONT='C:/Windows/Fonts/msyh.ttc';BOLD='C:/Windows/Fonts/msyhbd.ttc'
C={'bg':'#f4f7fb','ink':'#172c45','muted':'#52657b','blue':'#245dc4','bluebg':'#e9f0ff','teal':'#087d75','tealbg':'#e3f4ef','orange':'#ac5a17','orangebg':'#fff0df','line':'#a0b1c4','purple':'#7554ad','purplebg':'#eee7fa'}
def font(s,b=False):return ImageFont.truetype(BOLD if b else FONT,s)
def wrap(t,f,maxw,d):
 lines=[]
 for para in t.split('\n'):
  line=''
  for ch in para:
   if d.textlength(line+ch,font=f)>maxw and line:lines.append(line);line=ch
   else:line+=ch
  lines.append(line)
 return lines
class Fig:
 def __init__(self,h,title,sub):
  self.im=Image.new('RGB',(1800,h),C['bg']);self.d=ImageDraw.Draw(self.im);self.h=h
  self.d.text((60,38),title,font=font(45,True),fill=C['ink'])
  self.d.text((60,104),sub,font=font(25),fill=C['muted'])
 def txt(self,xy,t,size=28,color=None,b=False,maxw=None):
  f=font(size,b);lines=wrap(t,f,maxw,self.d) if maxw else t.split('\n');x,y=xy
  for line in lines:self.d.text((x,y),line,font=f,fill=color or C['ink']);y+=int(size*1.45)
  return y
 def panel(self,box,title,color='blue'):
  self.d.rounded_rectangle(box,24,fill='white',outline='#dce4ef',width=2)
  self.d.text((box[0]+28,box[1]+18),title,font=font(30,True),fill=C[color])
 def box(self,box,title,body='',color='blue',ts=30,bs=24):
  bg=C.get(color+'bg',C['bluebg']);self.d.rounded_rectangle(box,18,fill=bg,outline=C[color],width=2)
  x,y,x2,y2=box;f=font(ts,True);ls=wrap(title,f,x2-x-32,self.d)
  yy=y+20
  for line in ls:
   tw=self.d.textlength(line,font=f);self.d.text(((x+x2-tw)/2,yy),line,font=f,fill=C[color]);yy+=ts+12
  if body:
   yy+=6;f=font(bs);ls=wrap(body,f,x2-x-32,self.d)
   for line in ls:
    tw=self.d.textlength(line,font=f);self.d.text(((x+x2-tw)/2,yy),line,font=f,fill=C['ink']);yy+=bs+10
  assert yy<=y2+4,('Text exceeds box',title,yy,y2)
 def arr(self,a,b,color='muted',width=4,dashed=False):
  x,y=a;xx,yy=b;dx=xx-x;dy=yy-y;dist=math.hypot(dx,dy)
  if dashed:
   for k in range(0,int(dist),20):
    t=k/dist;u=min(k+11,dist)/dist;self.d.line((x+dx*t,y+dy*t,x+dx*u,y+dy*u),fill=C[color],width=width)
  else:self.d.line((a,b),fill=C[color],width=width)
  ang=math.atan2(dy,dx);size=13
  pts=[b,(xx-size*math.cos(ang-.45),yy-size*math.sin(ang-.45)),(xx-size*math.cos(ang+.45),yy-size*math.sin(ang+.45))]
  self.d.polygon(pts,fill=C[color])
 def foot(self,t):self.txt((60,self.h-55),t,22,C['muted'])
 def save(self,lec,name):self.im.save(root/f'讲座{lec}'/'配图'/name)
# 1. SigMA
f=Fig(1120,'SigMA：先在短信号上学规律，再适配长信号','讲解示意 · 上半部分学通用编码器，下半部分用任务标签继续调整模型')
f.panel((40,165,1760,575),'第一阶段｜用短/中等信号做预训练')
xs=[75,505,935,1365]
for x,t,b,c in zip(xs,['I/Q信号切成小块','Mamba顺序编码','MSP多尺度预测','对照真实目标'],['给每个小块位置信息','边读边更新有限状态\n保留对后续有用的信息','同一编码器接受\n不同时间尺度的监督','计算误差，调整参数\n不需要人工类别标签'],['blue','blue','teal','orange']):
 f.box((x,255,x+355,420),t,b,c)
for a,b in zip(xs,xs[1:]):f.arr((a+355,337),(b-8,337))
f.txt((85,475),'训练完成后保留编码器参数；MSP提供学习要求，Mamba负责组织序列信息。',29)
f.panel((40,620,1760,1015),'第二阶段｜将预训练能力迁移到更长输入任务','teal')
for x,t,b,c in zip(xs,['更长的I/Q信号','APA分块适配','预训练编码器','任务头＋微调'],['直接细切会产生很多块\n计算和内存开销增大','调整分块粒度\n映射原嵌入层权重','处理适配后的序列\n利用已经学到的规律','识别、检测或参数估计\n用任务标签继续训练'],['blue','teal','blue','orange']):
 f.box((x,715,x+355,880),t,b,c)
for a,b in zip(xs,xs[1:]):f.arr((a+355,797),(b-8,797))
f.txt((85,935),'APA控制输入开销；更粗的分块可能损失细节，长输入迁移仍需验证。',29)
f.foot('依据照片3—5整理；省略网络层细节，多尺度目标的具体构造以报告原实现为准。')
f.save(1,'00_方法流程示意.png')
# 2. Structural pretraining
f=Fig(1400,'结构化预训练：对齐时间，再预测信号的结构编号','讲解示意 · 标签由无标注信号统计得到；结构编号不等于人工类别或已知物理含义')
f.panel((40,165,1760,570),'准备训练目标｜先建立局部与组合结构词表')
xs=[75,505,935,1365]
for x,t,b,c in zip(xs,['信号切成patch','提特征并聚类','Atomic编号','Emergent组合'],['收集很多局部小块','相似局部特征归为一组','每块得到一个局部编号\n例如 A1、A2、A3','高频相邻编号逐步合并\n再回填到每个patch'],['blue','blue','teal','purple']):
 f.box((x,255,x+355,425),t,b,c)
for a,b in zip(xs,xs[1:]):f.arr((a+355,340),(b-8,340))
f.txt((85,480),'得到两套对齐标签：每一块是什么局部结构，以及它属于什么较长组合。',29)
f.panel((40,610,1760,1150),'正式预训练｜遮住一部分输入，让模型预测两套编号','teal')
f.box((75,760,400,925),'I/Q切块并遮挡','藏起一段，保留上下文','blue')
f.box((475,760,800,925),'RoPE-fs时间坐标','按采样率换算位置\n表达实际时间间隔','teal')
f.box((875,760,1200,925),'Transformer编码','从可见片段提取信息','blue')
f.box((1320,705,1720,845),'Atomic预测头','预测被遮挡块的局部编号','teal',29,24)
f.box((1320,875,1720,1015),'Emergent预测头','预测该位置所属的组合编号','purple',29,24)
f.arr((400,842),(467,842));f.arr((800,842),(867,842));f.arr((1200,842),(1310,775));f.arr((1200,842),(1310,945))
f.arr((1540,570),(1540,695),'purple',dashed=True)
f.arr((1540,600),(1740,600),'purple',dashed=True)
f.arr((1740,600),(1740,945),'purple',dashed=True)
f.arr((1740,945),(1725,945),'purple',dashed=True)
f.txt((1050,635),'两套标签分别核对两头预测',23,C['purple'])
f.txt((85,1068),'用两级预测误差更新编码器；鼓励保留能跨观测条件复用的结构。',29)
f.box((275,1195,1525,1310),'下游使用：新信号 → 已学编码器 → 任务头 → 用少量或全量标签微调','结构词表和掩码练习用于预训练，不是每次预测都重新聚类。','orange',28,24)
f.foot('依据照片6—8整理；词表特征提取器、训练超参数及损失细节仍需原论文确认。')
f.save(2,'00_方法流程示意.png')
# 3. IQFM overall, keep method levels distinct.
f=Fig(1370,'IQFM、KRMAE、差分相位：分别改了哪个环节？','讲解示意 · 通用基座、预训练练习方式、下游输入表征属于不同层面的设计')
f.panel((40,165,1760,525),'① IQFM｜学习一个可用于多个任务的I/Q编码器')
xs=[75,505,935,1365]
for x,t,b,c in zip(xs,['无标注I/Q','Transformer编码','轻量解码器重建','学到通用编码器'],['采样、归一化、分块\n遮住部分输入','利用可见片段\n提取上下文特征','恢复缺失波形\n与原信号比较误差','保存参数\n为后续任务提供起点'],['blue','blue','teal','orange']):f.box((x,260,x+355,430),t,b,c)
for a,b in zip(xs,xs[1:]):f.arr((a+355,345),(b-8,345))
f.panel((40,565,1760,865),'② KRMAE｜改变预训练时“遮哪里”，让练习更有针对性','teal')
f.box((75,655,560,805),'从波形能量找关键区域','利用雷达脉冲通常较集中的特点','teal',29,25)
f.box((655,655,1140,805),'在关键区域中选择遮挡位置','不是把关键区域永久丢弃','teal',29,25)
f.box((1235,655,1720,805),'继续做编码—解码重建练习','改变监督选择，不另造一套分类骨干','teal',29,25)
f.arr((560,730),(645,730));f.arr((1140,730),(1225,730))
f.panel((40,905,1760,1250),'③ RW-DPhase｜改变特定下游任务的输入表征','purple')
for x,t,b,c in zip(xs,['任务I/Q回波','构造更稳的输入','使用预训练编码器','接分类头微调'],['先取幅度和相位信息','相邻点相位变化\n按幅度估计可靠性','输入适配后的表征\n继续学习任务差异','输出目标类别\n效果需按场景验证'],['blue','purple','blue','orange']):f.box((x,1000,x+355,1170),t,b,c)
for a,b in zip(xs,xs[1:]):f.arr((a+355,1085),(b-8,1085))
f.foot('依据照片10、14、17整理；各项实验分别论证，不代表图中所有方案已联合训练并验证。')
f.save(3,'00_方法流程示意.png')
# 4. KRMAE intuition with synthetic pulse waveform.
f=Fig(1140,'KRMAE：优先让模型练习恢复有信息的脉冲区域','教学波形 · 仅表达“找区域—选遮挡—学重建”的思想，不是报告的实验数据或阈值规则')
f.panel((40,165,1760,760),'同一段信号：先找脉冲，再在其中遮住部分小块','teal')
for y,label,masked in [(345,'原始信号',False),(600,'用于训练的输入',True)]:
 f.txt((75,y-95),label,28,b=True)
 f.d.line((285,y,1700,y),fill='#c6d1df',width=2)
 pts=[]
 for k in range(1400):
  env=.12
  for center in [170,460,760,1100]:
   if abs(k-center)<85:env=.85
  val=math.sin(k*.25)*env*62+math.sin(k*.91)*3
  pts.append((300+k,y-val))
 f.d.line(pts,fill=C['blue'],width=3)
 for center in [170,460,760,1100]:
  x=300+center
  if not masked:
   f.d.rounded_rectangle((x-85,y-78,x+85,y+78),10,outline=C['teal'],width=3)
  else:
   f.d.rounded_rectangle((x-40,y-80,x+30,y+80),7,fill='#d4dce8',outline='#8494aa',width=2)
   f.txt((x-30,y-18),'?',30,C['muted'],True)
f.txt((290,704),'绿色框：候选关键区域；灰块：选中的遮挡位置。其余可见部分提供恢复线索。',26)
f.panel((40,800,1760,1030),'用原信号作答案，反复调整模型参数','orange')
for x,t,b,c in zip([75,505,935,1365],['可见输入','编码器提取特征','解码器恢复波形','比较重建误差'],['保留上下文','学习脉冲之间的规律','预测缺失部分','更新参数；之后再接任务头'],['blue','blue','teal','orange']):f.box((x,875,x+355,1000),t,b,c,28,22)
for a,b in zip([75,505,935],[505,935,1365]):f.arr((a+355,938),(b-8,938))
f.foot('关键区域的检测并非必然正确；弱目标与强背景可能使能量规则失效，需单独验证。')
f.save(3,'01_KRMAE原理示意.png')
# 5. Differential phase and reliability.
f=Fig(1190,'可靠性加权差分相位：看变化，并降低不可靠相位的影响','教学示意 · 精确权重函数未在照片中给出；图中没有假定具体加权公式')
f.panel((40,165,1760,705),'把幅度与相位分工使用，再合成任务输入','purple')
f.box((75,365,385,530),'复数I/Q回波','包含幅度与相位','blue')
f.box((485,245,905,395),'幅度：信号有多强','用来估计相位的可靠性','teal')
f.box((485,475,905,625),'相邻采样点的相位变化','尽量减少恒定初相的影响','purple')
f.box((1030,365,1390,530),'可靠性加权','减弱低幅不稳定位置\n保留更可信的变化','purple')
f.box((1470,365,1725,530),'任务输入','幅度＋加权差分\n再接编码器分类','orange',29,22)
f.arr((385,410),(475,320));f.arr((385,490),(475,550));f.arr((905,320),(1020,410));f.arr((905,550),(1020,490));f.arr((1390,447),(1460,447))
f.txt((85,650),'差分不等于没有噪声：低幅相位、频偏和相位噪声仍然会影响结果。',27)
f.panel((40,750,1760,1075),'一个直观例子：角度跨过 ±180°，不代表信号真的突然转了一大圈','teal')
f.txt((85,840),'记录的角度：    170°  →  179°  →  −179°  →  −170°',35,b=True)
f.txt((85,915),'直接数值相减：   +9°         −358°          +9°',31,C['orange'])
f.txt((85,985),'按圆周求相邻变化：+9°           +2°          +9°',31,C['teal'])
f.foot('依据照片16—18整理；差分角仍有边界，图示仅说明跨角度边界时为什么不能直接相减。')
f.save(3,'02_差分相位原理示意.png')
print('Created 38 source slide crops and 5 teaching diagrams. Originals unchanged.')
