import matplotlib
matplotlib.use('Agg')   # must be set before anything imports pyplot

from Integration import *   # getStressed, createOutputFolders, ...

import os
import json
import time
import shutil
import threading
import subprocess
import traceback
from collections import Counter
from contextlib import contextmanager

from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024   # reject uploads over 50 MB

OUTDIR = app.instance_path
os.makedirs(OUTDIR, exist_ok=True)

SAMPLE_VIDEO_DIR = os.path.join(app.root_path, 'static', 'video')

# Uploaded clips are deleted after analysis (they are recordings of a person's face).
# Set KEEP_UPLOADS=1 while debugging to keep them.
KEEP_FILES = os.environ.get('KEEP_UPLOADS') == '1'

EMOTION_NAMES = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# The models and matplotlib's global state are not thread-safe: one analysis at a time.
ANALYSIS_LOCK = threading.Lock()


# ---------------------------------------------------------------- CORS
# The video page is served from a different origin (file://, Live Server, ...),
# so the browser blocks the response unless we allow it.
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = os.environ.get('CORS_ORIGIN', '*')
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response


# ---------------------------------------------------------------- helpers
def _safe_rmtree(path):
    """Delete a folder, but only if it lives inside the instance folder."""
    root = os.path.abspath(OUTDIR)
    target = os.path.abspath(path)
    if target != root and target.startswith(root + os.sep):
        shutil.rmtree(target, ignore_errors=True)


@contextmanager
def workspace():
    userdir, framesdir, plotsdir = createOutputFolders(OUTDIR)
    try:
        yield userdir, framesdir, plotsdir
    finally:
        if not KEEP_FILES:
            for folder in {userdir, framesdir, plotsdir}:
                _safe_rmtree(folder)


def convert_to_mp4(src, dst):
    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-i', src,
        '-an',                                   # audio is not needed
        '-filter:v', 'fps=30',
        '-b:v', '3M', '-minrate', '3M', '-maxrate', '3M', '-bufsize', '3M',
        dst,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except FileNotFoundError:
        raise RuntimeError('ffmpeg is not installed or not on the PATH of the server.')
    except subprocess.TimeoutExpired:
        raise RuntimeError('Converting the recording took too long.')
    if proc.returncode != 0 or not os.path.exists(dst):
        print(proc.stderr)
        raise RuntimeError('ffmpeg could not convert the recording.')


def _label(value, high, medium, labels):
    if value >= high:
        return labels[2]
    if value >= medium:
        return labels[1]
    return labels[0]


def summarize(final_stress_score, heart_rates, emotions, facial_movements):
    """Turn the raw per-sample lists into the values the video page displays."""
    scores = [float(v) for v in final_stress_score]
    bpms = [float(v) for v in heart_rates]
    faces = [float(v) for v in facial_movements]
    codes = [int(v) for v in emotions]

    # converter() forces the first score to 0 (warm-up), so leave it out of the average.
    usable = scores[1:] or scores
    avg_stress = sum(usable) / len(usable) if usable else 0.0
    avg_face = sum(faces) / len(faces) if faces else 0.0
    valid_bpm = [b for b in bpms if b > 0]

    dominant = None
    if codes:
        code = Counter(codes).most_common(1)[0][0]
        if 0 <= code < len(EMOTION_NAMES):
            dominant = EMOTION_NAMES[code]

    return {
        # raw series (same keys as before)
        'StressScore': scores,
        'HeartRates': bpms,
        'Emotions': codes,
        'EmotionNames': [EMOTION_NAMES[c] if 0 <= c < len(EMOTION_NAMES) else 'Unknown' for c in codes],
        'FacialMovements': faces,
        # summary fields used by the video page
        'emotion': dominant,
        'heart_rate': round(sum(valid_bpm) / len(valid_bpm), 1) if valid_bpm else None,
        'tension': _label(avg_face, 0.75, 0.65, ('Low', 'Moderate', 'High')),
        'stress_level': _label(avg_stress, 2.0, 1.2, ('Low', 'Moderate', 'Elevated')),
    }


def run_analysis(video_path, frames_dir, plots_dir):
    with ANALYSIS_LOCK:
        t0 = time.time()
        result = getStressed(video_path, frames_dir, plots_dir)
        print('*** PREDICTION TIME ***', round(time.time() - t0, 1), 'seconds')
    return summarize(*result)


def analysis_error(exc):
    """Map an exception to a JSON error the front end can show."""
    if isinstance(exc, RuntimeError):
        return jsonify(error=str(exc)), 500
    if isinstance(exc, (ZeroDivisionError, IndexError, ValueError)):
        traceback.print_exc()
        return jsonify(error="Couldn't read enough from this clip. Check the lighting and that your face is in view."), 422
    traceback.print_exc()
    return jsonify(error='Analysis failed. Check the server log.'), 500


# ---------------------------------------------------------------- routes
@app.route('/')
def home():
    return jsonify(status='ok', service='umeed-stress-analysis')


@app.route('/stress', methods=['POST'])
def detect_stress():
    print('** NEW VIDEO UPLOADED **')

    upload = request.files.get('file')
    if upload is None or not upload.filename:
        return jsonify(error="No video received (expected a form field named 'file')."), 400

    ext = os.path.splitext(secure_filename(upload.filename))[1].lower()
    if ext not in ('.webm', '.mp4', '.mov', '.mkv'):
        ext = '.webm'

    try:
        with workspace() as (userdir, framesdir, plotsdir):
            raw_path = os.path.join(userdir, 'uploaded' + ext)
            mp4_path = os.path.join(userdir, 'converted.mp4')
            upload.save(raw_path)

            if os.path.getsize(raw_path) < 1024:
                return jsonify(error='The recording was empty.'), 400

            convert_to_mp4(raw_path, mp4_path)
            data = run_analysis(mp4_path, framesdir, plotsdir)

            if KEEP_FILES:
                with open(os.path.join(userdir, 'results.json'), 'w') as fh:
                    json.dump(data, fh)

            return jsonify(data)
    except Exception as exc:
        return analysis_error(exc)


@app.route('/stresstest', methods=['POST'])
def stresstest():
    """Run the analysis on a sample video kept in static/video (for testing)."""
    name = secure_filename(request.form.get('videos', ''))
    video_path = os.path.join(SAMPLE_VIDEO_DIR, name)
    if not name or not os.path.isfile(video_path):
        return jsonify(error='Sample video not found in static/video.'), 404

    try:
        with workspace() as (userdir, framesdir, plotsdir):
            return jsonify(run_analysis(video_path, framesdir, plotsdir))
    except Exception as exc:
        return analysis_error(exc)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, threaded=True)