from flask import Flask, render_template, request, jsonify
import os
import tempfile
import librosa
import numpy as np
from pydub import AudioSegment
from shutil import which  # ✅ 自動偵測 ffmpeg 或 avconv

# ✅ 自動設定 pydub converter
ffmpeg_path = which("ffmpeg") or which("avconv")
if ffmpeg_path:
    AudioSegment.converter = ffmpeg_path
else:
    print("⚠️ 找不到 ffmpeg 或 avconv，音檔轉換可能失敗")

app = Flask(__name__)

# --- 歌曲資料庫，可持續擴充 ---
SONGS = {
    "jianjia": {
        "title": "蒹葭",
        "segments": {
            "1": "static/jianjia/jianjia-1.mp3",
            "2": "static/jianjia/jianjia-2.mp3",
            "3": "static/jianjia/jianjia-3.mp3",
            "4": "static/jianjia/jianjia-4.mp3",
            "5": "static/jianjia/jianjia-5.mp3",
            "6": "static/jianjia/jianjia-6.mp3",
            "7": "static/jianjia/jianjia-7.mp3",
            "8": "static/jianjia/jianjia-8.mp3",
            "9": "static/jianjia/jianjia-9.mp3",
            "10": "static/jianjia/jianjia-10.mp3",
            "11": "static/jianjia/jianjia-11.mp3",
            "12": "static/jianjia/jianjia-12.mp3",
        }
    },
    "look": {
        "title": "望海潮",
        "segments": {
            "1": "static/look/look-1.mp3",
            "2": "static/look/look-2.mp3",
            "3": "static/look/look-3.mp3",
            "4": "static/look/look-4.mp3",
            "5": "static/look/look-5.mp3",
            "6": "static/look/look-6.mp3",
            "7": "static/look/look-7.mp3",
            "8": "static/look/look-8.mp3",
            "9": "static/look/look-9.mp3",
            "10": "static/look/look-10.mp3",
            "11": "static/look/look-11.mp3",
            "12": "static/look/look-12.mp3",
            "13": "static/look/look-13.mp3",
            "14": "static/look/look-14.mp3",
            "15": "static/look/look-15.mp3",
            "16": "static/look/look-16.mp3",
        }
    },
    "nian": {
        "title": "念奴嬌",
        "segments": {
            "1": "static/nian/nian-1.mp3",
            "2": "static/nian/nian-2.mp3",
            "3": "static/nian/nian-3.mp3",
            "4": "static/nian/nian-4.mp3",
            "5": "static/nian/nian-5.mp3",
            "6": "static/nian/nian-6.mp3",
            "7": "static/nian/nian-7.mp3",
            "8": "static/nian/nian-8.mp3",
            "9": "static/nian/nian-9.mp3",
            "10": "static/nian/nian-10.mp3",
            "11": "static/nian/nian-11.mp3",
            "12": "static/nian/nian-12.mp3",
            "13": "static/nian/nian-13.mp3",
            "14": "static/nian/nian-14.mp3",
            "15": "static/nian/nian-15.mp3",
        }
    },
    "nian": {
        "title": "武陵春",
        "segments": {
            "1": "static/spring/spring-1.mp3",
            "2": "static/spring/spring-2.mp3",
            "3": "static/spring/spring-3.mp3",
            "4": "static/spring/spring-4.mp3",
            "5": "static/spring/spring-5.mp3",
            "6": "static/spring/spring-6.mp3",
            "7": "static/spring/spring-7.mp3",
            "8": "static/spring/spring-8.mp3",
            "9": "static/spring/spring-9.mp3",
            "10": "static/spring/spring-10.mp3",
        }
    }
}

# ---- 音準分數 ----
def pitch_score_waveform(f0_ref, f0_user):
    min_len = min(len(f0_ref), len(f0_user))
    if min_len == 0:
        return 0
    f0_ref = f0_ref[:min_len]
    f0_user = f0_user[:min_len]
    mask = ~np.isnan(f0_ref) & ~np.isnan(f0_user)
    if np.sum(mask) == 0:
        return 0
    f0_ref = f0_ref[mask]
    f0_user = f0_user[mask]
    corr = np.corrcoef(f0_ref, f0_user)[0, 1]
    return int(max(0, min(100, corr * 100)))

# ---- 穩定度分數 ----
def stability_score(y_ref, y_user, sr):
    rms_ref = librosa.feature.rms(y=y_ref, frame_length=2048, hop_length=512)[0]
    rms_user = librosa.feature.rms(y=y_user, frame_length=2048, hop_length=512)[0]
    if len(rms_ref) < 5 or len(rms_user) < 5:
        return 0
    # 標準化
    rms_ref = (rms_ref - np.min(rms_ref)) / (np.max(rms_ref) - np.min(rms_ref) + 1e-6)
    rms_user = (rms_user - np.min(rms_user)) / (np.max(rms_user) - np.min(rms_user) + 1e-6)
    # 對齊
    x_ref = np.linspace(0, 1, len(rms_ref))
    x_user = np.linspace(0, 1, len(rms_user))
    rms_user_aligned = np.interp(x_ref, x_user, rms_user)
    corr = np.corrcoef(rms_ref, rms_user_aligned)[0, 1]
    return int(max(0, min(100, (corr ** 0.5) * 150)))

# ---- 音訊預處理 ----
def preprocess_audio(y, sr):
    y_trimmed, _ = librosa.effects.trim(y, top_db=25)
    return y_trimmed

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/practice')
def practice():
    return render_template('playlist.html')

@app.route('/practice/jianjia')
def jianjia():
    return render_template('reeds.html')

@app.route('/practice/look')
def look():
    return render_template('look.html')

@app.route('/practice/nian')
def nian():
    return render_template('nian.html')

@app.route('/practice/spring')
def spring():
    return render_template('spring.html')

@app.route('/evaluate_audio', methods=['POST'])
def evaluate_audio():
    try:
        file = request.files.get('audio')
        song = request.form.get('song', "jianjia").strip()
        segment = request.form.get('segment', "1").strip()

        if not file:
            return jsonify({'error': '沒有收到音訊檔案'}), 400

        # --- 檢查歌曲是否存在 ---
        if song not in SONGS:
            return jsonify({'error': f'歌曲不存在：{song}'}), 400

        ref_path_map = SONGS[song]["segments"]

        # --- 檢查段落是否存在 ---
        ref_audio_path = ref_path_map.get(segment)
        if not ref_audio_path:
            return jsonify({'error': f'{song} 沒有第 {segment} 段'}), 400

        # --- 存暫存錄音 ---
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_input:
            file.save(temp_input.name)
            input_path = temp_input.name

        # --- 轉換成 wav ---
        temp_wav = input_path.replace(".webm", ".wav")
        audio = AudioSegment.from_file(input_path, format="webm")
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(temp_wav, format="wav")

        # --- 載入音檔 ---
        y_ref, sr_ref = librosa.load(ref_audio_path, sr=16000)
        y_user, sr_user = librosa.load(temp_wav, sr=16000)

        # --- 預處理 ---
        y_ref = preprocess_audio(y_ref, sr_ref)
        y_user = preprocess_audio(y_user, sr_user)

        # --- F0 ---
        f0_ref, _, _ = librosa.pyin(y_ref, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        f0_user, _, _ = librosa.pyin(y_user, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))

        pitch = pitch_score_waveform(f0_ref, f0_user)
        stability = stability_score(y_ref, y_user, sr_ref)
        total = pitch + stability

        # --- 清除暫存 ---
        os.remove(input_path)
        os.remove(temp_wav)

        return jsonify({
            "song": song,
            "segment": segment,
            "pitch": pitch,
            "stability": stability,
            "total": total
        })

    except Exception as e:
        print("⚠️ Error:", e)
        return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     from waitress import serve
#     port = int(os.environ.get("PORT", 8080))
#     serve(app, host="0.0.0.0", port=port)

if __name__ == '__main__':
    app.run(debug=True)




