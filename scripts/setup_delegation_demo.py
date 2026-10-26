"""Create a three-member delegation demo through the local console API."""
import http.cookiejar
import json
from pathlib import Path
import urllib.request

BASE = 'http://127.0.0.1:9899'
SESSION = 'session_multihop_demo_20261104'
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))

def api(path, data=None, method=None):
    request = urllib.request.Request(BASE + path,
        data=None if data is None else json.dumps(data).encode('utf-8'),
        headers={'Content-Type': 'application/json'}, method=method)
    with opener.open(request, timeout=90) as response:
        result = json.load(response)
    if result.get('status') != 'success':
        raise RuntimeError(str(result.get('message', 'API failed')))
    return result

config = json.loads(Path('config.json').read_text(encoding='utf-8-sig'))
api('/auth/login', {'password': config.get('web_password', '')})
common = '''\n这是一个多级委派测试团队。请使用中文。仅使用 agent_delegate 完成成员间交接，不用 subagent 替代，不模拟队友发言。每次交办必须带上用户原始数据、要求和验收标准。对用户使用成员名字；内部调用使用准确的 Agent ID。如果委派工具不可用或调用失败，明确报告，禁止声称已经委派。不要修改项目文件、配置，不发送外部消息。不要执行首次身份引导。\n'''
roles = [
    ('hop-lead', '委派测试负责人', '多级委派测试总负责人；接收用户任务，委派分析员，汇总真实返回结果。',
     '你是委派测试负责人。接到测试任务后，必须先调用 agent_delegate，把完整任务交给 hop-analyst（委派测试分析员），并要求其先完成初步分析，再交给 hop-checker（委派测试核验员）独立核验，收到核验后再返回给你。你不能跳过分析员直接交给核验员。收到分析员结果后向用户汇总结论、核验意见和实际成功的交接链；失败时说明在哪一级失败。'),
    ('hop-analyst', '委派测试分析员', '承担初步分析，并必须继续委派核验员独立检查，构成第二级委派。',
     '你是委派测试分析员。收到负责人委派后，先独立计算或分析，再必须调用 agent_delegate，把原始任务、全部原始数据和你的初步结果交给 hop-checker（委派测试核验员），要求其独立复算和检查。不要把任务转回负责人。收到核验结果后，返回你的结论、核验员实际反馈以及是否修正。若核验失败，明确说明，不伪造核验。'),
    ('hop-checker', '委派测试核验员', '多级委派链的最终核验者；独立复算，指出错误，不再转交。',
     '你是委派测试核验员，是委派链最后一级。根据收到的原始数据独立计算和检查分析员结论，返回：核验通过或不通过、独立计算过程、发现的问题和修正值。不再调用 agent_delegate，不向上游发起新委派。只向委派方返回核验报告。'),
]
snapshot = api('/api/agents')
existing = {item['id']: item for item in snapshot.get('agents', [])}
for aid, name, description, instructions in roles:
    if aid not in existing:
        api('/api/agents', {'action': 'create', 'id': aid, 'name': name,
            'description': description, 'skills': [], 'knowledge': []})
    elif existing[aid].get('name') != name:
        raise RuntimeError('Agent ID already used: ' + aid)
    path = f'/api/agents/{aid}/files/AGENT.md'
    current = api(path)
    api(path, {'content': '# AGENT.md\n\n' + instructions + '\n' + common,
        'revision': current['revision']}, 'PUT')
    print('Configured:', aid, name)

api(f'/api/sessions/{SESSION}/settings', {
    'agent_id': 'hop-lead', 'members': ['hop-analyst', 'hop-checker']})
api(f'/api/sessions/{SESSION}/generate_title', {
    'agent_id': 'hop-lead', 'user_message': '多级委派测试：负责人、分析员、核验员'})
api(f'/api/sessions/{SESSION}', {'agent_id': 'hop-lead', 'title': '多级委派测试（负责人 → 分析员 → 核验员）'}, 'PUT')
state = api(f'/api/sessions/{SESSION}/settings?agent_id=hop-lead')
print(json.dumps({'session_id': SESSION, 'team': state.get('team')}, ensure_ascii=False))
