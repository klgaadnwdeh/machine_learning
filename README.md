<strong>机器学习期末大作业</strong>  
<br>
<strong >作者</strong>:汝鑫(224081200050)，孙宇(224081200053)，陈思康(224081200029)<br>


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
