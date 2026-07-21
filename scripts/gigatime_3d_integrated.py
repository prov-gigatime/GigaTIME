"""
GigaTIME 3D Viewer — Gradio + Separate 3D Server (v4)
=======================================================
Two-server approach:
  • Gradio (port 7860): Model inference, 2D gallery, embeds 3D via iframe
  • HTTP  (port 7861): Serves standalone Three.js 3D viewer HTML

This bypasses all Gradio HTML sanitization issues.

Setup:
  1. Place this file in GigaTIME/scripts/ (next to archs.py)
  2. conda activate gigatime
  3. pip install gradio
  4. export HF_TOKEN=<your_huggingface_token>
  5. python gigatime_3d_integrated.py

  Both servers start automatically.
"""

import os, sys, json, base64, io, threading
import numpy as np
import torch
import gradio as gr
from PIL import Image
from huggingface_hub import snapshot_download
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import archs

# ── Static directory for the 3D viewer ────────────────────────────────────────
VIEWER_DIR = os.path.join(SCRIPT_DIR, "_gigatime_viewer")
os.makedirs(VIEWER_DIR, exist_ok=True)

VIEWER_PORT = 7861

# ── Constants ─────────────────────────────────────────────────────────────────
ALL_CHANNEL_NAMES = [
    'DAPI', 'TRITC', 'Cy5', 'PD-1', 'CD14', 'CD4', 'T-bet', 'CD34',
    'CD68', 'CD16', 'CD11c', 'CD138', 'CD20', 'CD3', 'CD8', 'PD-L1',
    'CK', 'Ki67', 'Tryptase', 'Actin-D', 'Caspase3-D', 'PHH3-B', 'Transgelin',
]
EXCLUDE = {'TRITC', 'Cy5'}
DISPLAY_CHANNELS = [n for n in ALL_CHANNEL_NAMES if n not in EXCLUDE]
DISPLAY_INDICES  = [i for i, n in enumerate(ALL_CHANNEL_NAMES) if n not in EXCLUDE]

MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])

NUM_CLASSES    = 23
INPUT_CHANNELS = 3
INPUT_SIZE     = 512
WINDOW_SIZE    = 256
DS = 4
DS_DIM = INPUT_SIZE // DS  # 128

CHANNEL_COLORS = [
    "#4169E1","#FF6B6B","#FFA726","#66BB6A","#AB47BC","#FFEE58","#FF1744",
    "#00E5FF","#FF9100","#E040FB","#00E676","#FF4081","#00BCD4","#FFD740",
    "#F48FB1","#B2FF59","#FF6E40","#18FFFF","#EA80FC","#CCFF90","#FFD180",
]

def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16)/255.0 for i in (0, 2, 4))

# ── Model ─────────────────────────────────────────────────────────────────────
def load_model():
    print("Loading GigaTIME model...")
    model = archs.gigatime(NUM_CLASSES, INPUT_CHANNELS)
    local_dir = snapshot_download(repo_id="prov-gigatime/GigaTIME")
    state_dict = torch.load(os.path.join(local_dir, "model.pth"), map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded on {device}")
    return model, device

MODEL, DEVICE = load_model()

# ── Inference ─────────────────────────────────────────────────────────────────
def preprocess(pil_img):
    img = pil_img.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).float()

def infer(tensor):
    b, c, h, w = tensor.shape
    out = torch.zeros(b, NUM_CLASSES, h, w, device=tensor.device)
    with torch.no_grad():
        for i in range(0, h, WINDOW_SIZE):
            for j in range(0, w, WINDOW_SIZE):
                out[:, :, i:i+WINDOW_SIZE, j:j+WINDOW_SIZE] = MODEL(
                    tensor[:, :, i:i+WINDOW_SIZE, j:j+WINDOW_SIZE])
    return torch.sigmoid(out).cpu().numpy()[0]

def pil_to_data_url(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ── Pipeline ──────────────────────────────────────────────────────────────────
def run_pipeline(input_image):
    if input_image is None:
        return [], make_placeholder("Upload an image and click Run")

    tensor = preprocess(input_image).to(DEVICE)
    probs  = infer(tensor)

    he_resized = input_image.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    he_data_url = pil_to_data_url(he_resized)

    # ── 2D Gallery ────────────────────────────────────────────────────────────
    gallery = [(he_resized, "H&E Input")]
    for ci, (ch_i, ch_name) in enumerate(zip(DISPLAY_INDICES, DISPLAY_CHANNELS)):
        prob = probs[ch_i]
        rc, gc, bc = hex_to_rgb(CHANNEL_COLORS[ci])
        intensity = prob.clip(0, 1)
        hmap = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
        hmap[:,:,0] = (intensity * rc * 255).clip(0,255).astype(np.uint8)
        hmap[:,:,1] = (intensity * gc * 255).clip(0,255).astype(np.uint8)
        hmap[:,:,2] = (intensity * bc * 255).clip(0,255).astype(np.uint8)
        gallery.append((Image.fromarray(hmap), ch_name))

    # ── 3D data → JSON file ──────────────────────────────────────────────────
    channel_grids = []
    for ch_i in DISPLAY_INDICES:
        full = probs[ch_i]
        small = full.reshape(DS_DIM, DS, DS_DIM, DS).mean(axis=(1, 3))
        q = (small * 255).clip(0, 255).astype(np.uint8)
        channel_grids.append(q.tolist())

    payload = {
        "he": he_data_url,
        "ch": channel_grids,
        "names": DISPLAY_CHANNELS,
        "dim": DS_DIM,
    }

    # Write data.json into the viewer directory
    data_path = os.path.join(VIEWER_DIR, "data.json")
    with open(data_path, "w") as f:
        json.dump(payload, f)

    # Return iframe pointing to the 3D viewer on port 7861
    # Add timestamp to bust cache
    import time
    ts = int(time.time() * 1000)
    iframe = (
        f'<iframe src="http://localhost:{VIEWER_PORT}/viewer.html?t={ts}" '
        f'style="width:100%;height:700px;border:none;border-radius:10px;" '
        f'allow="accelerometer;autoplay"></iframe>'
    )
    return gallery, iframe


def make_placeholder(msg):
    return (
        f"<div style='height:700px;display:flex;align-items:center;justify-content:center;"
        f"background:#08080f;border-radius:10px;color:#667;font-family:monospace;font-size:14px'>"
        f"<div style='text-align:center'>"
        f"<div style='font-size:48px;margin-bottom:12px'>&#x1F9EC;</div>"
        f"<div style='color:#8899aa'>{msg}</div>"
        f"</div></div>"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Write the standalone 3D viewer HTML
# ══════════════════════════════════════════════════════════════════════════════
VIEWER_HTML = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>GigaTIME 3D</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
html,body{width:100%;height:100%;overflow:hidden;background:#08080f;
  font-family:'JetBrains Mono','Fira Code','Courier New',monospace;color:#c8d6e5}
#root{display:flex;width:100%;height:100%}
#sb{width:250px;min-width:250px;height:100%;overflow-y:auto;
  background:rgba(12,15,22,.97);border-right:1px solid rgba(80,120,200,.12);
  display:flex;flex-direction:column}
#sb::-webkit-scrollbar{width:4px}
#sb::-webkit-scrollbar-thumb{background:rgba(100,140,200,.2);border-radius:2px}
#vw{flex:1;position:relative}
#vw canvas{display:block}
.sec{padding:9px 12px;border-bottom:1px solid rgba(80,120,200,.08)}
.tit{font-size:8px;text-transform:uppercase;letter-spacing:1.1px;color:#556;margin-bottom:4px}
.cr{display:flex;align-items:center;gap:6px;padding:4px 5px;border-radius:4px;
  cursor:pointer;margin-bottom:1px;transition:opacity .12s,background .1s}
.cr:hover{background:rgba(65,105,225,.07)}
.cr.off{opacity:.25}
input[type=range]{width:100%;accent-color:#4169E1;height:3px}
input[type=checkbox]{accent-color:#4169E1}
#badge{position:absolute;top:10px;left:10px;background:rgba(12,15,22,.75);
  border:1px solid rgba(80,120,200,.15);border-radius:8px;padding:5px 10px;z-index:10}
#hint{position:absolute;bottom:10px;left:50%;transform:translateX(-50%);
  background:rgba(12,15,22,.8);border:1px solid rgba(80,120,200,.12);
  border-radius:6px;padding:4px 12px;font-size:8px;color:#556;z-index:10;pointer-events:none}
#loading{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
  color:#7aa2f7;font-size:14px;z-index:20}
</style>
</head><body>
<div id="root">
  <div id="sb"></div>
  <div id="vw">
    <div id="badge">
      <div style="font-size:11px;font-weight:700;color:#e8f0fe">GigaTIME 3-D</div>
      <div style="font-size:7.5px;color:#4a5;letter-spacing:1px;text-transform:uppercase">Real model predictions</div>
    </div>
    <div id="loading">Loading 3D data...</div>
    <div id="hint">Drag to rotate · Scroll to zoom · Toggle channels in sidebar</div>
  </div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
// Fetch data.json from the same server, then build scene
fetch('data.json?t=' + Date.now())
  .then(function(r){ return r.json(); })
  .then(function(DATA){ init(DATA); })
  .catch(function(e){
    document.getElementById('loading').textContent = 'Error loading data: ' + e.message;
  });

function init(DATA) {
  document.getElementById('loading').style.display = 'none';

  var COLORS = [
    "#4169E1","#FF6B6B","#FFA726","#66BB6A","#AB47BC","#FFEE58","#FF1744",
    "#00E5FF","#FF9100","#E040FB","#00E676","#FF4081","#00BCD4","#FFD740",
    "#F48FB1","#B2FF59","#FF6E40","#18FFFF","#EA80FC","#CCFF90","#FFD180"
  ];
  var DESCS = [
    "Nuclear stain","Immune checkpoint","Monocytes","Helper T-cells",
    "Th1 transcription","Stem/endothelial","Macrophages","NK cells",
    "Dendritic cells","Plasma cells","B-cells","Pan T-cells",
    "Cytotoxic T-cells","Immune ligand","Cytokeratin","Proliferation",
    "Mast cells","Smooth muscle","Apoptosis","Mitosis","Stromal"
  ];

  var N=DATA.names.length, DIM=DATA.dim;
  var vis=[]; for(var i=0;i<N;i++) vis.push(true);
  var layerSp=0.12, elev=0.18, autoRot=true;

  // ── Three.js ─────────────────────────────────────────────────────────────
  var vw=document.getElementById('vw');
  var W=vw.clientWidth, H=vw.clientHeight;

  var scene=new THREE.Scene();
  scene.background=new THREE.Color(0x08080f);
  scene.fog=new THREE.FogExp2(0x08080f, 0.10);

  var camera=new THREE.PerspectiveCamera(42, W/H, 0.1, 100);
  var renderer=new THREE.WebGLRenderer({antialias:true});
  renderer.setSize(W, H);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
  renderer.toneMapping=THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure=1.3;
  vw.insertBefore(renderer.domElement, vw.firstChild);

  scene.add(new THREE.AmbientLight(0x445577, 0.6));
  var dl=new THREE.DirectionalLight(0xffffff,1.0);
  dl.position.set(3,5,3); scene.add(dl);
  var pl=new THREE.PointLight(0x4488ff,0.35,12);
  pl.position.set(-2,4,-1); scene.add(pl);

  var grp=new THREE.Group(); scene.add(grp);

  // Base H&E
  var tex=new THREE.TextureLoader().load(DATA.he);
  var bMat=new THREE.MeshStandardMaterial({map:tex,side:THREE.DoubleSide,roughness:0.55,metalness:0.08});
  var bMesh=new THREE.Mesh(new THREE.PlaneGeometry(2,2), bMat);
  bMesh.rotation.x=-Math.PI/2; grp.add(bMesh);

  // Wireframe
  var eg=new THREE.EdgesGeometry(new THREE.BoxGeometry(2.08,0.015,2.08));
  var wire=new THREE.LineSegments(eg,new THREE.LineBasicMaterial({color:0x445566}));
  wire.position.y=-0.008; grp.add(wire);

  // Labels
  function mkLbl(t,px,py,pz){
    var c=document.createElement('canvas');c.width=256;c.height=64;
    var x=c.getContext('2d');x.fillStyle='#667788';x.font='bold 26px monospace';
    x.textAlign='center';x.fillText(t,128,40);
    var s=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(c),transparent:true}));
    s.position.set(px,py,pz);s.scale.set(0.75,0.19,1);grp.add(s);
  }
  mkLbl('x: H&E slide',0,-0.14,1.28);
  mkLbl('y: mIF channels',1.28,-0.14,0);

  // ── Point clouds ─────────────────────────────────────────────────────────
  var chObjs=[];
  function buildCh(){
    for(var k=0;k<chObjs.length;k++) if(chObjs[k]) grp.remove(chObjs[k]);
    chObjs=[];
    for(var ch=0;ch<N;ch++){
      var grid=DATA.ch[ch], col=new THREE.Color(COLORS[ch]);
      var pos=[],cls=[];
      for(var y=0;y<DIM;y++){
        for(var x=0;x<DIM;x++){
          var val=grid[y][x]/255.0;
          if(val<0.08) continue;
          pos.push((x/DIM-0.5)*2, (ch+1)*layerSp+val*elev, (y/DIM-0.5)*2);
          var v=Math.min(val*1.8,1.0);
          cls.push(col.r*v, col.g*v, col.b*v);
        }
      }
      if(pos.length===0){chObjs.push(null);continue;}
      var geom=new THREE.BufferGeometry();
      geom.setAttribute('position',new THREE.Float32BufferAttribute(pos,3));
      geom.setAttribute('color',new THREE.Float32BufferAttribute(cls,3));
      var mat=new THREE.PointsMaterial({
        size:0.024, vertexColors:true, transparent:true, opacity:0.9,
        sizeAttenuation:true, blending:THREE.AdditiveBlending, depthWrite:false
      });
      var pts=new THREE.Points(geom,mat);
      pts.visible=vis[ch]; grp.add(pts); chObjs.push(pts);
    }
  }
  buildCh();

  // ── Orbit ────────────────────────────────────────────────────────────────
  var drag=false,prevX=0,prevY=0;
  var sT=Math.PI/4.2,sP=Math.PI/4.5,sR=3.6;
  function camUp(){
    camera.position.set(sR*Math.sin(sP)*Math.cos(sT),sR*Math.cos(sP),sR*Math.sin(sP)*Math.sin(sT));
    camera.lookAt(0,0.55,0);
  }
  camUp();

  var cvs=renderer.domElement;
  cvs.addEventListener('pointerdown',function(e){drag=true;prevX=e.clientX;prevY=e.clientY;cvs.setPointerCapture(e.pointerId);});
  cvs.addEventListener('pointermove',function(e){
    if(!drag)return;
    sT-=(e.clientX-prevX)*0.007;
    sP=Math.max(0.18,Math.min(1.5,sP+(e.clientY-prevY)*0.007));
    prevX=e.clientX;prevY=e.clientY;camUp();
  });
  cvs.addEventListener('pointerup',function(){drag=false;});
  cvs.addEventListener('wheel',function(e){
    e.preventDefault();sR=Math.max(1.4,Math.min(8,sR+e.deltaY*0.004));camUp();
  },{passive:false});

  // ── Animate ──────────────────────────────────────────────────────────────
  var t=0;
  function loop(){
    requestAnimationFrame(loop);
    if(autoRot&&!drag){t+=0.003;sT=Math.PI/4.2+Math.sin(t)*0.45;camUp();}
    renderer.render(scene,camera);
  }
  loop();

  window.addEventListener('resize',function(){
    var nw=vw.clientWidth,nh=vw.clientHeight;
    camera.aspect=nw/nh;camera.updateProjectionMatrix();renderer.setSize(nw,nh);
  });

  // ── Sidebar ──────────────────────────────────────────────────────────────
  var sb=document.getElementById('sb');
  function renderSB(){
    var h='';
    h+='<div class="sec" style="background:linear-gradient(180deg,rgba(65,105,225,.06),transparent)">';
    h+='<div style="display:flex;align-items:center;gap:6px;margin-bottom:3px">';
    h+='<div style="width:24px;height:24px;border-radius:5px;display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,#4169E1,#00BCD4);font-size:12px">&#x1F52C;</div>';
    h+='<div><div style="font-size:11.5px;font-weight:700;color:#e8f0fe">GigaTIME</div>';
    h+='<div style="font-size:7.5px;color:#4a5;letter-spacing:1.2px;text-transform:uppercase">Real Model · 3D</div></div></div>';
    h+='<div style="font-size:7.5px;color:#3a4556;line-height:1.3">21 mIF channels · Research only</div></div>';

    h+='<div class="sec">';
    h+='<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px">';
    h+='<span class="tit" style="margin:0">Controls</span>';
    h+='<label style="font-size:9px;color:#7aa2f7;cursor:pointer;display:flex;align-items:center;gap:3px">';
    h+='<input type="checkbox" id="ck-rot"'+(autoRot?' checked':'')+'>Auto-rotate</label></div>';
    h+='<div style="margin-bottom:4px"><div style="display:flex;justify-content:space-between;font-size:8px;color:#556;margin-bottom:1px"><span>Layer spacing</span><span id="lbl-sp">'+layerSp.toFixed(2)+'</span></div>';
    h+='<input type="range" id="sl-sp" min="0.03" max="0.35" step="0.005" value="'+layerSp+'"></div>';
    h+='<div><div style="display:flex;justify-content:space-between;font-size:8px;color:#556;margin-bottom:1px"><span>Height scale</span><span id="lbl-el">'+elev.toFixed(2)+'</span></div>';
    h+='<input type="range" id="sl-el" min="0.02" max="0.6" step="0.005" value="'+elev+'"></div></div>';

    var on=0; for(var j=0;j<N;j++) if(vis[j]) on++;
    h+='<div style="padding:5px 12px 2px;display:flex;justify-content:space-between;align-items:center">';
    h+='<span class="tit" style="margin:0">Channels ('+on+'/'+N+')</span>';
    h+='<span id="btn-tog" style="font-size:8px;color:#7aa2f7;cursor:pointer">'+(on===N?'Hide all':'Show all')+'</span></div>';

    h+='<div style="flex:1;overflow-y:auto;padding:0 8px 8px">';
    for(var i=0;i<N;i++){
      var o=vis[i];
      h+='<div class="cr'+(o?'':' off')+'" data-i="'+i+'">';
      h+='<div style="width:8px;height:8px;border-radius:2px;flex-shrink:0;background:'+COLORS[i]+';'+(o?'box-shadow:0 0 4px '+COLORS[i]+'60':'')+'"></div>';
      h+='<div style="flex:1;min-width:0"><div style="font-size:10px;font-weight:600;color:'+(o?'#e0e8f4':'#445')+'">'+DATA.names[i]+'</div>';
      h+='<div style="font-size:7px;color:#445;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">'+DESCS[i]+'</div></div>';
      h+='<div style="width:4px;height:4px;border-radius:50%;background:'+(o?COLORS[i]:'rgba(80,120,200,.18)')+'"></div></div>';
    }
    h+='</div>';
    h+='<div style="padding:6px 12px;border-top:1px solid rgba(80,120,200,.08);font-size:7px;color:#2a3546;text-align:center;line-height:1.4">GigaTIME · Microsoft/Providence/UW · Cell 2025</div>';
    sb.innerHTML=h;

    document.getElementById('ck-rot').addEventListener('change',function(e){autoRot=e.target.checked;});
    document.getElementById('sl-sp').addEventListener('input',function(e){
      layerSp=parseFloat(e.target.value);document.getElementById('lbl-sp').textContent=layerSp.toFixed(2);buildCh();
    });
    document.getElementById('sl-el').addEventListener('input',function(e){
      elev=parseFloat(e.target.value);document.getElementById('lbl-el').textContent=elev.toFixed(2);buildCh();
    });
    document.getElementById('btn-tog').addEventListener('click',function(){
      var all=true; for(var j=0;j<N;j++) if(!vis[j]) all=false;
      for(var j=0;j<N;j++) vis[j]=!all;
      for(var j=0;j<chObjs.length;j++) if(chObjs[j]) chObjs[j].visible=vis[j];
      renderSB();
    });
    var rows=document.querySelectorAll('.cr');
    for(var r=0;r<rows.length;r++){
      (function(row){
        row.addEventListener('click',function(){
          var idx=parseInt(row.getAttribute('data-i'));
          vis[idx]=!vis[idx];
          if(chObjs[idx]) chObjs[idx].visible=vis[idx];
          renderSB();
        });
      })(rows[r]);
    }
  }
  renderSB();
}
</script>
</body></html>"""

# Write viewer.html once at startup
viewer_html_path = os.path.join(VIEWER_DIR, "viewer.html")
with open(viewer_html_path, "w") as f:
    f.write(VIEWER_HTML)
print(f"3D viewer HTML written to {viewer_html_path}")


# ══════════════════════════════════════════════════════════════════════════════
# HTTP server for 3D viewer (port 7861)
# ══════════════════════════════════════════════════════════════════════════════
class CORSHandler(SimpleHTTPRequestHandler):
    """Serves files from VIEWER_DIR with CORS headers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=VIEWER_DIR, **kwargs)

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        super().end_headers()

    def log_message(self, format, *args):
        pass  # Quiet


def start_viewer_server():
    server = HTTPServer(('0.0.0.0', VIEWER_PORT), CORSHandler)
    print(f"3D viewer server running at http://localhost:{VIEWER_PORT}")
    server.serve_forever()


# ══════════════════════════════════════════════════════════════════════════════
# Gradio App
# ══════════════════════════════════════════════════════════════════════════════
CSS = """
#viewer-html { min-height: 720px; }
.gradio-container { max-width: 1400px !important; }
"""

PLACEHOLDER = (
    "<div style='height:700px;display:flex;align-items:center;justify-content:center;"
    "background:#08080f;border-radius:10px;color:#667;font-family:monospace;font-size:14px'>"
    "<div style='text-align:center'>"
    "<div style='font-size:48px;margin-bottom:12px'>&#x1F9EC;</div>"
    "<div style='color:#8899aa'>Upload an H&amp;E tile and click<br>"
    "<b style='color:#7aa2f7'>Run GigaTIME Inference</b></div>"
    "</div></div>"
)

with gr.Blocks(
    title="GigaTIME 3-D — Virtual mIF from H&E",
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("JetBrains Mono"),
    ),
    css=CSS,
) as demo:

    gr.Markdown("""
    # 🔬 GigaTIME: 3-D Virtual Multiplex Immunofluorescence
    Upload an H&E pathology tile → the **GigaTIME model** (NestedUNet, 23-channel output)
    runs real inference → results are rendered as a **rotatable 3-D point cloud** with each
    protein channel in a distinct color, stacked above the H&E base slide.

    > **Research use only.** Not for clinical decision-making.
    > [Paper (Cell)](https://aka.ms/gigatime-paper) ·
    > [Model Card](https://huggingface.co/prov-gigatime/GigaTIME) ·
    > [GitHub](https://github.com/prov-gigatime/GigaTIME)
    """)

    with gr.Row():
        with gr.Column(scale=1, min_width=280):
            input_image = gr.Image(type="pil", label="Upload H&E Tile", height=300)
            run_btn = gr.Button("⚡ Run GigaTIME Inference", variant="primary", size="lg")
            gr.Markdown("""
            **Predicted channels (21):**
            DAPI · PD-1 · CD14 · CD4 · T-bet · CD34 · CD68 · CD16 ·
            CD11c · CD138 · CD20 · CD3 · CD8 · PD-L1 · CK · Ki67 ·
            Tryptase · Actin-D · Caspase3-D · PHH3-B · Transgelin
            """)

        with gr.Column(scale=3):
            with gr.Tab("🧊 3-D Viewer"):
                html_viewer = gr.HTML(value=PLACEHOLDER, elem_id="viewer-html")
            with gr.Tab("🖼️ 2-D Channel Gallery"):
                gallery = gr.Gallery(
                    label="Virtual mIF Channels",
                    columns=4, rows=3, height=600,
                    object_fit="contain", preview=True,
                )

    run_btn.click(fn=run_pipeline, inputs=input_image, outputs=[gallery, html_viewer])

if __name__ == "__main__":
    # Start the 3D viewer HTTP server in a background thread
    t = threading.Thread(target=start_viewer_server, daemon=True)
    t.start()

    # Start Gradio
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)