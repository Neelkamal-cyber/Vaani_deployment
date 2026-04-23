---
title: VAANI Sign Language Recognition
emoji: 🤟
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: 1.32.2
python_version: 3.10.11
app_file: app.py
pinned: false
license: mit
---

# 🤟 VAANI — Real-Time Indian Sign Language → Speech

VAANI converts Indian Sign Language gestures into spoken words in real-time
using only a standard webcam. No special hardware. No app install.

## How it works
1. **MediaPipe Holistic** extracts 258 skeletal keypoints per frame (pose + both hands)
2. A rolling 60-frame window feeds an **Encoder-Decoder LSTM** model
3. Confirmed signs are spoken aloud via **gTTS** (plays directly in the browser)

## Usage
1. Click **START** on the camera widget and allow webcam access
2. Sign clearly with good, even lighting
3. The detected sign appears on the right panel and is spoken aloud
4. Press **🗑️ Clear** to reset the sentence

## Vocabulary (31 ISL signs)
ALL · BATHROOM · BUY · CANCEL · COLD · DIFFERENT · DOCTOR · DRINK · EAT ·
FIRE · HOW · Hello · KNOW · LATER · MAYBE · NEED · NO · OK · OLD · PLAY ·
POLICE MAN · SICK · STOP · THANK YOU · UNDER · WANT · WORK · big · money ·
please · water

## Performance
| Condition | Accuracy |
|-----------|----------|
| Isolated signs (good lighting) | **95%** |
| Continuous signing | 85–90% |
| End-to-end latency | < 20 ms |

## Files required in your Space
```
app.py
requirements.txt
packages.txt
label_map.json
README.md
vaani_endec_deploy.h5      ← upload via drag-and-drop or Git LFS
```

## Uploading a large model with Git LFS (if > 50 MB)
```bash
git lfs install
git lfs track "*.h5"
git add .gitattributes
git add vaani_endec_deploy.h5
git commit -m "add model via LFS"
git push
```

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Team
Neel Kamal (2203400100036) · Arun Chaudhary (2203400100010) · Chetan Sharma (2203400100015)  
Vivekananda College of Technology & Management, Aligarh  
B.Tech CSE Final Year Project · 2026
