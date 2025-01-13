<strong>机器学习期末大作业</strong>  
<br>
<strong >作者</strong>:汝鑫(224081200050)，孙宇(224081200053)，陈思康(224081200029)<br>

<strong>简洁</strong><br>
    本文件是机器学习课程的期末大作业，由汝鑫、孙宇和陈思康共同完成。该作业旨在通过构建和训练一个对话代理系统来深入理解和实践机器学习算法的应用，特别是因子分解机（FM）、主动采样器及负采样器的预训练过程，并最终对所开发的对话代理进行评估。项目要求使用Python 3.6及以上版本，结合Numpy、PyTorch等库进行开发。整个项目的实施分为四个主要步骤：首先是环境准备和知识图谱的构建，接着是对FM模型以及主动和负采样器的预训练，然后是对话代理的训练，最后是对对话代理性能的评估。所有训练模型和日志都将保存在特定的目录中，以便于后续分析和论文撰写。此项目不仅展示了团队成员在机器学习领域的技能与积累，同时也为未来的研究提供了有价值的参考和改进方向
以下是具体配置设置，请自行完成配置运行。

Environment Requirement<br>
```bash
$ Python >= 3.6<br>
$ Numpy >= 1.12<br>
$ PyTorch >= 1.0<br>
```
<strong>运行代码的示例</strong><br>
训练模型保存在文件夹 run-log/<数据名称>/<训练模型名称>-model 中。训练日志记录在文件夹 run-log/<数据名称>/<训练模型名称>-log 和文件 run-log/<训练步骤>.out 中。<br>
<strong>0. Preparation</strong><br>
```bash
$ python base_config.py<br>
```
参数设置的解析函数在 configuration/base_config.py 文件中配置。<br>
```bash
$ python knowledge_graph.py <br>
```
知识图谱在 KG/knowledge_graph.py 文件中准备。<br>
<strong>1.预训练FM</strong><br>
```bash
$ python 1_fm_train.py<br>
```
FM模型的实现代码位于FM文件夹中。<br>
<strong>2.预训练主动采样器和负采样器</strong><br>
```bash
$ python 2_active_sampler_train.py<br>
$ python 2_negative_sampler_train.py<br>
```
采样器的实现代码分别位于 active-sampler 和 negative-sampler 文件夹中。<br>
<strong>3.训练对话代理</strong><br>
```bash
$ python 3_run.py<br>
```
对话代理的实现代码位于 conversational-policy/conversational_policy.py 文件中<br>
<strong>4.评估对话代理</strong><br>
```bash
$ python 4_policy_evaluate.py<br>
```
对话代理的评估代码位于 conversational-policy/conversational_policy_evaluate.py 文件中<br>
