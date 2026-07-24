"""Build the routing-ablation dashboard: runs.jsonl -> one self-contained HTML file.

Brand-styled (white surface, ink #0a0a0a, hairline #ececec) with the chart palette validated
by the dataviz six-checks script (#0070f3 #b8770a #7928ca #0d9488 #ee0000, fixed variant
order; amber/teal are darkened chart steps of the brand hues - raw brand accents fail the
lightness/contrast checks on white). Sections: cost-quality Pareto per matrix (log-x), knob
frontier lines, model-mix stacked bars (top-4 + Other, 2px gaps), token breakdown, run table.
Every mark has a hover tooltip; the table is the accessibility fallback.

Usage: uv run python .agents/scripts/build_dashboard.py [--out .wmh/evals/dashboard.html]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RUNS = Path(".wmh/evals/runs.jsonl")

TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Routing ablations</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--ink:#0a0a0a;--muted:#6b6b6b;--grid:#ececec;--surface:#ffffff;
--c0:#0070f3;--c1:#b8770a;--c2:#7928ca;--c3:#0d9488;--c4:#ee0000;--other:#9a9a9a}
*{box-sizing:border-box}
body{margin:0;background:var(--surface);color:var(--ink);
font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;padding:32px 40px 80px}
h1{font-size:20px;margin:0 0 4px;text-align:left}
h2{font-size:15px;margin:36px 0 10px;text-align:left}
.sub{color:var(--muted);margin:0 0 20px}
.filters{display:flex;gap:8px;flex-wrap:wrap;margin:16px 0 8px}
.filters button{border:1px solid var(--grid);background:var(--surface);color:var(--ink);
padding:5px 12px;border-radius:6px;cursor:pointer;font:inherit}
.filters button.on{border-color:var(--ink);font-weight:600}
.legend{display:flex;gap:16px;flex-wrap:wrap;color:var(--muted);margin:6px 0 2px}
.legend span{display:inline-flex;align-items:center;gap:6px}
.swatch{width:10px;height:10px;border-radius:3px;display:inline-block}
svg text{font:11px -apple-system,sans-serif;fill:var(--muted)}
svg .tick line{stroke:var(--grid)}
.pop{position:fixed;pointer-events:none;background:var(--surface);border:1px solid var(--grid);border-radius:12px;box-shadow:0 10px 30px rgba(0,0,0,.10);padding:16px 18px;max-width:400px;z-index:10;opacity:0;transition:opacity .1s;font-size:12.5px;line-height:1.55}
.pop h4{margin:0 0 8px;font-size:13px}
.pop table{margin:0;width:100%}.pop td,.pop th{padding:2px 8px 2px 0;border:none;text-align:right;font-size:12px}
.pop td:first-child,.pop th:first-child{text-align:left}
.tip{position:fixed;pointer-events:none;background:var(--ink);color:#fff;padding:8px 10px;
border-radius:6px;font-size:12px;line-height:1.5;opacity:0;transition:opacity .08s;max-width:340px;z-index:9}
table{border-collapse:collapse;width:100%;margin-top:8px}
th,td{padding:6px 10px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}
th{color:var(--muted);font-weight:500;cursor:pointer}
.grid2{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:28px}
.note{border:1px solid var(--grid);border-radius:8px;padding:12px 16px;color:var(--muted);margin-top:10px}
.info{display:inline-flex;align-items:center;justify-content:center;width:14px;height:14px;border:1px solid var(--muted);border-radius:50%;color:var(--muted);font-size:10px;margin-left:5px;cursor:help;vertical-align:1px}
</style></head><body>
<h1>Routing ablations</h1>
<p class="sub">Cost, quality, and speed per router variant, on OUR model pool. Every dot is one
run on the held-out split; hover anything for the full record. Rebuild with --all to also see
the fitter-validation matrices (other papers' model pools).</p>
<div class="filters" id="matrixFilter"></div>
<div class="legend" id="variantLegend"></div>
<h2>Headlines: best routed run vs best single model</h2>
<div id="heads" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:14px"></div>
<h2>Cost vs quality (Pareto), per matrix</h2>
<div class="grid2" id="pareto"></div>
<h2>Model mix per run (all models)</h2>
<div class="legend" id="mixLegend"></div>
<div id="mix"></div>
<h2>Blended tokens by model</h2>
<div id="tokens"></div>
<h2>All runs</h2>
<table id="runs"></table>
<div class="note">Reading guide: cost and p50 latency often drop <b>together</b> when the mix
shifts toward cheaper models, because cheaper models are usually also faster - verify via the
per-model p50 in each tooltip and the token breakdown. Accuracy deltas smaller than the noise
floor (~1/sqrt(n) on the test count) are ties.</div>
<div class="tip" id="tip"></div>
<div class="pop" id="pop"></div>
<script>
const DATA = __DATA__;
const MATRIX_INFO = {
 'routerbench': 'RouterBench (2024): the classic public matrix. 36k prompts x 11 models from 2023 (gpt-4, claude-2 era) with measured per-call costs. Validates the fitter against published conventions; models are dated.',
 'llmrouterbench-flagship': 'LLMRouterBench (2026) flagship track: 13 modern flagship models (gemini-2.5, gpt-5, claude-sonnet-4, deepseek, qwen3-235b...) on hard datasets (AIME, GPQA, HLE...). Deduped by task text after a caught leak; small (809 scenarios), so noise floors are wide.',
 'routerbench-ours9': 'OUR 9-model pool (gpt-5.5/5.4-mini, fable/sonnet/haiku/opus, deepseek-v4-pro, kimi-k2.6, glm-5.2) run live on 1,199 gold-certified RouterBench MCQ prompts, exact-match graded (judge-free). Real latency per call.',
};
function matrixInfo(m){return MATRIX_INFO[m] || (m.startsWith('wm-') ?
 'Closed-loop world-model scenarios: our 9-model pool rolled as the agent against the '+m.slice(3)+' base world model (25 scenarios x 2 episodes, max 8 steps, judge pinned Opus 4.8, tool surface derived from the corpus traces). Small per-corpus test sides: read the cross-corpus aggregate, not one row.' : '');}
const COL_INFO = {
 'matrix':'Which outcome matrix (benchmark) the run was evaluated on. Hover the benchmark filters above for descriptions.',
 'variant':'Router variant + its knobs. lam = the cost knob: reward points paid per average-call-cost unit (0 = pure accuracy).',
 'acc':'Mean reward of the routed choices on HELD-OUT scenarios (exact-match or judge-scored depending on the matrix). Deltas below ~1/sqrt(n) are ties.',
 'cost/call':'Mean measured cost of the routed model per call, in USD (per-call usage x the real per-token price of that model; never list-price guesses).',
 'p50':'Median latency of the routed model calls themselves. "-" = this matrix carries no timings (frozen public benchmarks).',
 'p95':'95th-percentile call latency of the routed choices.',
 'vs best-single acc':'Accuracy delta (points) vs the strongest single model chosen on the fit split - the honest "why route at all" bar.',
 'vs cost':'Cost delta (%) vs that same best single model. Negative = cheaper.',
 'n':'Held-out test scenarios. Small n = wide noise floor (~+-1/sqrt(n)).',
};
const VC = {"best-single":"var(--c0)","rank":"var(--c1)","irt":"var(--c2)","jisi":"var(--c3)","static":"var(--c4)"};
const tip = document.getElementById('tip');
const pop = document.getElementById('pop');
function showPop(e, html){pop.innerHTML=html;pop.style.opacity=1;
 const w=Math.min(400,innerWidth-40);
 pop.style.left=Math.min(e.clientX+16,innerWidth-w-20)+'px';
 pop.style.top=Math.min(e.clientY+14,innerHeight-pop.offsetHeight-20)+'px';}
function hidePop(){pop.style.opacity=0;}
// One global color per model (fixed by overall token volume), shared by every section.
const MODEL_COLORS={};
{const vol={};
 DATA.forEach(r=>Object.entries(r.result.tokens_by_model||{}).forEach(([m,b])=>vol[m]=(vol[m]||0)+b.input+b.output));
 const GR=['#5c5c5c','#7d7d7d','#9a9a9a','#b3b3b3','#c9c9c9','#dedede','#6e6e6e','#8b8b8b'];
 Object.keys(vol).sort((a,b)=>vol[b]-vol[a]).forEach((m,i)=>MODEL_COLORS[m]=i<5?`var(--c${i})`:GR[(i-5)%GR.length]);}
function showTip(e, html){tip.innerHTML=html;tip.style.opacity=1;
 tip.style.left=Math.min(e.clientX+14,innerWidth-360)+'px';tip.style.top=(e.clientY+12)+'px';}
function hideTip(){tip.style.opacity=0;}
const fmt$ = v=>'$'+v.toFixed(5), fmtP=v=>(100*v).toFixed(1)+'%';
const matrices=[...new Set(DATA.map(r=>r.matrix))];
let active=new Set(matrices);
function tooltip(r){
 const res=r.result;
 let h=`<b>${r.variant}</b> ${JSON.stringify(r.params)}<br>${r.matrix} · ${res.scenarios} scenarios`+
 `<br>accuracy ${fmtP(res.accuracy)} · cost ${fmt$(res.cost_per_call)}`+
 (res.latency_p50_s?`<br>p50 ${res.latency_p50_s.toFixed(2)}s · p95 ${res.latency_p95_s.toFixed(2)}s`:'');
 const bs=r.baselines&&r.baselines.best_single;
 if(bs) h+=`<br>vs best-single: acc ${(res.accuracy-bs.accuracy>=0?'+':'')}${(100*(res.accuracy-bs.accuracy)).toFixed(1)}pt, cost ${(100*(res.cost_per_call/bs.cost_per_call-1)).toFixed(0)}%`;
 const mix=Object.entries(res.model_mix).sort((a,b)=>b[1]-a[1]).slice(0,4)
   .map(([m,s])=>{const p=res.per_model_latency_p50_s[m];
     return `${m} ${fmtP(s)}${p?` (p50 ${p.toFixed(2)}s)`:''}`}).join('<br>');
 return h+'<br><br><u>mix</u><br>'+mix;
}
function paretoChart(matrix){
 const rows=DATA.filter(r=>r.matrix===matrix);
 const W=440,H=280,L=54,R=12,T=16,B=40;
 const costs=rows.map(r=>r.result.cost_per_call), accs=rows.map(r=>r.result.accuracy);
 const x0=Math.min(...costs)/1.5, x1=Math.max(...costs)*1.5;
 const y0=Math.max(0,Math.min(...accs)-0.03), y1=Math.min(1,Math.max(...accs)+0.03);
 const X=v=>L+(Math.log(v)-Math.log(x0))/(Math.log(x1)-Math.log(x0))*(W-L-R);
 const Y=v=>T+(y1-v)/(y1-y0)*(H-T-B);
 let s=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="cost vs accuracy, ${matrix}">`;
 for(let i=0;i<=4;i++){const v=y0+(y1-y0)*i/4;
  s+=`<line x1="${L}" x2="${W-R}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--grid)"/>`+
     `<text x="${L-6}" y="${Y(v)+4}" text-anchor="end">${fmtP(v)}</text>`;}
 [x0*2,x0*8,x0*32,x0*128].filter(v=>v<x1).forEach(v=>{
  s+=`<text x="${X(v)}" y="${H-B+16}" text-anchor="middle">${fmt$(v)}</text>`;});
 s+=`<text x="${(L+W-R)/2}" y="${H-6}" text-anchor="middle">cost per call (log)</text>`;
 // knob frontier lines per variant (sorted by cost), then dots
 for(const variant of [...new Set(rows.map(r=>r.variant))]){
  const vr=rows.filter(r=>r.variant===variant).sort((a,b)=>a.result.cost_per_call-b.result.cost_per_call);
  if(vr.length>1){
   const d=vr.map((r,i)=>`${i?'L':'M'}${X(r.result.cost_per_call)},${Y(r.result.accuracy)}`).join('');
   s+=`<path d="${d}" fill="none" stroke="${VC[variant]||'var(--other)'}" stroke-width="2" opacity="0.5"/>`;}
 }
 rows.forEach((r,i)=>{
  const cx=X(r.result.cost_per_call), cy=Y(r.result.accuracy);
  s+=`<circle data-i="${DATA.indexOf(r)}" cx="${cx}" cy="${cy}" r="6" fill="${VC[r.variant]||'var(--other)'}"
   stroke="var(--surface)" stroke-width="2"/>`;});
 s+=`</svg>`;
 const div=document.createElement('div');
 div.innerHTML=`<h3 style="font-size:13px;margin:0 0 4px">${matrix}<span class="info" data-t="${matrixInfo(matrix).replace(/"/g,'&quot;')}">i</span></h3>`+s;
 const inf=div.querySelector('.info');
 inf.addEventListener('mousemove',e=>showTip(e,inf.dataset.t));
 inf.addEventListener('mouseleave',hideTip);
 div.querySelectorAll('circle').forEach(c=>{
  c.style.cursor='pointer';
  c.addEventListener('mousemove',e=>showTip(e,tooltip(DATA[+c.dataset.i])));
  c.addEventListener('mouseleave',hideTip);});
 return div;
}
function stacked(containerId, per, unit){ // per: run -> [ [label, value], ... ]
 const cont=document.getElementById(containerId); cont.innerHTML='';
 const rows=DATA.filter(r=>active.has(r.matrix));
 const share={};
 rows.forEach(r=>Object.entries(per(r)).forEach(([m,v])=>share[m]=(share[m]||0)+v));
 const models=Object.keys(share).sort((a,b)=>share[b]-share[a]);
 const GRAYS=['#5c5c5c','#7d7d7d','#9a9a9a','#b3b3b3','#c9c9c9','#dedede','#6e6e6e','#8b8b8b'];
 const MC={}; models.forEach((m,i)=>MC[m]=i<5?`var(--c${i})`:GRAYS[(i-5)%GRAYS.length]);
 const leg=document.getElementById('mixLegend');
 if(containerId==='mix'){leg.innerHTML=models.map(m=>`<span><i class="swatch" style="background:${MC[m]}"></i>${m}</span>`).join('');}
 for(const r of rows){
  const entries=Object.entries(per(r)); if(!entries.length) continue;
  const total=entries.reduce((a,[,v])=>a+v,0); if(!total) continue;
  const top=entries.sort((a,b)=>b[1]-a[1]);
  const W=560,H=22; let x=0, s=`<svg viewBox="0 0 ${W} ${H}" style="max-width:${W}px">`;
  const seg=(label,v,color)=>{const w=v/total*(W-2);
   s+=`<rect x="${x}" y="2" width="${Math.max(w-2,1)}" height="16" rx="3" fill="${color}"
    data-t="${label}: ${unit==='%'?fmtP(v/total):v.toLocaleString()}"/>`;x+=w;};
  top.forEach(([m,v])=>seg(m,v,MC[m]||'var(--other)'));
  s+='</svg>';
  const row=document.createElement('div');
  row.style.cssText='display:flex;gap:12px;align-items:center;margin:3px 0';
  row.innerHTML=`<span style="width:340px;color:var(--muted);font-size:12px;text-align:right">${r.matrix} · ${r.variant} ${JSON.stringify(r.params)}</span>`+s;
  row.querySelectorAll('rect').forEach(el=>{
   el.addEventListener('mousemove',e=>showTip(e,el.dataset.t));
   el.addEventListener('mouseleave',hideTip);});
  cont.appendChild(row);
 }
}
function table(){
 const t=document.getElementById('runs');
 const rows=DATA.filter(r=>active.has(r.matrix));
 const th=(label,key)=>`<th>${label}<span class="info" data-t="${(COL_INFO[key]||'').replace(/"/g,'&quot;')}">i</span></th>`;
 t.innerHTML='<tr>'+th('matrix','matrix')+th('variant · params','variant')+th('acc','acc')+
  th('cost/call','cost/call')+th('p50','p50')+th('p95','p95')+
  th('vs best-single acc','vs best-single acc')+th('vs cost','vs cost')+th('n','n')+'</tr>'+
  rows.map(r=>{const res=r.result,bs=r.baselines&&r.baselines.best_single;
   return `<tr><td>${r.matrix}</td><td>${r.variant} ${JSON.stringify(r.params)}</td>`+
   `<td>${fmtP(res.accuracy)}</td><td>${fmt$(res.cost_per_call)}</td>`+
   `<td>${res.latency_p50_s?res.latency_p50_s.toFixed(2)+'s':'-'}</td>`+
   `<td>${res.latency_p95_s?res.latency_p95_s.toFixed(2)+'s':'-'}</td>`+
   `<td>${bs?((res.accuracy-bs.accuracy>=0?'+':'')+(100*(res.accuracy-bs.accuracy)).toFixed(1)+'pt'):'-'}</td>`+
   `<td>${bs?((100*(res.cost_per_call/bs.cost_per_call-1)).toFixed(0)+'%'):'-'}</td>`+
   `<td>${res.scenarios}</td></tr>`}).join('');
 t.querySelectorAll('th .info').forEach(el=>{
  el.addEventListener('mousemove',e=>showTip(e,el.dataset.t));
  el.addEventListener('mouseleave',hideTip);});
}
function headlines(){
 const cont=document.getElementById('heads'); cont.innerHTML='';
 for(const m of matrices.filter(x=>active.has(x))){
  const rows=DATA.filter(r=>r.matrix===m);
  const bs=rows.find(r=>r.variant==='best-single');
  const routed=rows.filter(r=>r.variant!=='best-single');
  if(!bs||!routed.length) continue;
  const best=routed.slice().sort((a,b)=>(b.result.accuracy-a.result.accuracy)||(a.result.cost_per_call-b.result.cost_per_call))[0];
  const r=best.result, b=bs.result;
  const dAcc=(100*(r.accuracy-b.accuracy));
  const xCost=b.cost_per_call/r.cost_per_call;
  const xLat=(r.latency_p50_s&&b.latency_p50_s)?(b.latency_p50_s/r.latency_p50_s):null;
  const good=v=>v>=0?'var(--c3)':'var(--c4)';
  const div=document.createElement('div');
  div.style.cssText='border:1px solid var(--grid);border-radius:10px;padding:14px 16px;cursor:help';
  // Headline only; everything else lives on hover.
  div.innerHTML=`<div style="color:var(--muted);font-size:12px">${m}</div>
   <div style="font-size:24px;font-weight:650;margin:6px 0 2px">${fmtP(r.accuracy)} <span style="font-size:13px;color:${good(dAcc)};font-weight:600">${dAcc>=0?'+':''}${dAcc.toFixed(1)}pt</span></div>
   <div style="color:var(--muted)"><b style="color:${good(xCost-1)}">${xCost>=1?xCost.toFixed(1)+'x cheaper':(1/xCost).toFixed(1)+'x pricier'}</b>`+
   (xLat?` · <b style="color:${good(xLat-1)}">${xLat>=1?xLat.toFixed(1)+'x faster':(1/xLat).toFixed(1)+'x slower'}</b>`:'')+`</div>`;
  // Per-model economics for this run: routed calls, total cost, token share.
  const counts={}; Object.entries(r.model_mix).forEach(([mo,s])=>counts[mo]=Math.round(s*r.scenarios));
  const totCost={}; Object.entries(r.per_model_cost_per_call||{}).forEach(([mo,c])=>totCost[mo]=c*(counts[mo]||0));
  const costSum=Object.values(totCost).reduce((a,v)=>a+v,0)||1;
  const tokSum=Object.values(r.tokens_by_model||{}).reduce((a,b)=>a+b.input+b.output,0)||1;
  const models=Object.keys(r.model_mix).sort((a,b)=>(totCost[b]||0)-(totCost[a]||0));
  let seg='', x=0; const W=360;
  models.forEach(mo=>{const w=(totCost[mo]||0)/costSum*W;
   if(w>0.5) seg+=`<rect x="${x}" y="2" width="${Math.max(w-2,1)}" height="12" rx="3" fill="${MODEL_COLORS[mo]||'var(--other)'}"/>`; x+=w;});
  const rows2=models.map(mo=>{const bk=(r.tokens_by_model||{})[mo]||{input:0,output:0};
   return `<tr><td><i class="swatch" style="background:${MODEL_COLORS[mo]||'var(--other)'}"></i> ${mo}</td>`+
    `<td>${fmtP(r.model_mix[mo])}</td><td>${fmtP((bk.input+bk.output)/tokSum)}</td>`+
    `<td>${fmt$(totCost[mo]||0)}</td><td>${fmtP((totCost[mo]||0)/costSum)}</td></tr>`}).join('');
  const detail=`<h4>${best.variant} ${JSON.stringify(best.params)} <span style="color:var(--muted);font-weight:400">vs best-single ${bs.params.model}</span></h4>`+
   `<table><tr><th></th><th>acc</th><th>cost/call</th><th>p50</th></tr>`+
   `<tr><td>routed</td><td><b>${fmtP(r.accuracy)}</b></td><td><b>${fmt$(r.cost_per_call)}</b></td><td><b>${r.latency_p50_s?r.latency_p50_s.toFixed(2)+'s':'-'}</b></td></tr>`+
   `<tr><td>baseline</td><td>${fmtP(b.accuracy)}</td><td>${fmt$(b.cost_per_call)}</td><td>${b.latency_p50_s?b.latency_p50_s.toFixed(2)+'s':'-'}</td></tr></table>`+
   `<div style="color:var(--muted);margin:8px 0 10px">n = ${r.scenarios} held-out${r.scenarios<60?' · <b>small n - wide noise floor</b>':''} · unscored ${r.unscored} · split seed ${best.split_seed} · fit ${best.fit_scenarios}</div>`+
   `<div style="font-weight:600;margin-bottom:4px">Cost by model</div>`+
   `<svg viewBox="0 0 ${W} 16" style="width:100%;display:block">${seg}</svg>`+
   `<table style="margin-top:6px"><tr><th>model</th><th>calls</th><th>tok %</th><th>cost</th><th>cost %</th></tr>${rows2}</table>`;
  div.addEventListener('mousemove',e=>showPop(e,detail));
  div.addEventListener('mouseleave',hidePop);
  cont.appendChild(div);
 }
}
function render(){
 headlines();
 const p=document.getElementById('pareto'); p.innerHTML='';
 matrices.filter(m=>active.has(m)).forEach(m=>p.appendChild(paretoChart(m)));
 stacked('mix', r=>r.result.model_mix, '%');
 stacked('tokens', r=>Object.fromEntries(Object.entries(r.result.tokens_by_model).map(([m,b])=>[m,b.input+b.output])), 'tok');
 table();
}
const mf=document.getElementById('matrixFilter');
matrices.forEach(m=>{const b=document.createElement('button');b.className='on';
 b.innerHTML=m+`<span class="info" data-t="${matrixInfo(m).replace(/"/g,'&quot;')}">i</span>`;
 b.onclick=()=>{b.classList.toggle('on');b.classList.contains('on')?active.add(m):active.delete(m);render();};
 b.querySelector('.info').addEventListener('mousemove',e=>{e.stopPropagation();showTip(e,e.target.dataset.t)});
 b.querySelector('.info').addEventListener('mouseleave',hideTip);
 b.querySelector('.info').addEventListener('click',e=>e.stopPropagation());
 mf.appendChild(b);});
document.getElementById('variantLegend').innerHTML=Object.entries(VC)
 .map(([v,c])=>`<span><i class="swatch" style="background:${c}"></i>${v}</span>`).join('');
render();
</script></body></html>
"""


def main() -> None:
    out = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else Path(
        ".wmh/evals/dashboard.html"
    )
    runs = [json.loads(line) for line in RUNS.read_text(encoding="utf-8").splitlines()]
    # Default view = OUR models only (the 9-model pool on certified prompts + the wm corpora).
    # The fitter-validation matrices (RouterBench 2023 models, LLMRouterBench flagships) are
    # research plumbing; include them explicitly with --all.
    if "--all" not in sys.argv:
        ours = {"routerbench-ours9"}
        runs = [r for r in runs if r["matrix"] in ours or r["matrix"].startswith("wm-")]
    out.parent.mkdir(parents=True, exist_ok=True)
    html = TEMPLATE.replace("__DATA__", json.dumps(runs))
    # Syntax gate: a single bad quote blanks the whole page silently; check before shipping.
    import re
    import shutil
    import subprocess
    import tempfile

    if shutil.which("node"):
        script = re.search(r"<script>(.*)</script>", html, re.S)
        assert script is not None
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as handle:
            handle.write(script.group(1))
        subprocess.run(["node", "--check", handle.name], check=True)
    out.write_text(html, encoding="utf-8")
    sys.stderr.write(f"dashboard: {len(runs)} runs -> {out}\n")


if __name__ == "__main__":
    main()
